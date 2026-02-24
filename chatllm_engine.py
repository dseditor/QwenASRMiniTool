"""
chatllm_engine.py — ChatLLM.cpp + Vulkan 推理後端

兩種執行模式：
  1. DLL 模式（優先）：ctypes 直接呼叫 libchatllm.dll，模型常駐記憶體
     - 每 chunk 約 0.23s（GPU shader 暖機後），免去 subprocess 啟動 overhead
  2. Subprocess 模式（後備）：每 chunk 啟動 main.exe 子程序
     - 模型每次重載，但不需要 DLL

輸出格式：language {lang}<asr_text>{transcription}
"""
from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

# ── 共用常數（與 app.py 保持同步）─────────────────────────────────────
SAMPLE_RATE   = 16000
VAD_CHUNK     = 512
VAD_THRESHOLD = 0.5
MAX_GROUP_SEC = 20
MAX_CHARS     = 20
MIN_SUB_SEC   = 0.6
GAP_SEC       = 0.08

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# Windows 旗標：建立子程序時不彈出主控台視窗（防止辨識時畫面閃爍）
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# STARTUPINFO：額外強制隱藏子程序視窗（搭配 CREATE_NO_WINDOW 雙重保護）
# CREATE_NO_WINDOW 阻止 console 分配，STARTF_USESHOWWINDOW+SW_HIDE 隱藏主視窗
_STARTUP_INFO: "subprocess.STARTUPINFO | None" = None
if sys.platform == "win32":
    _STARTUP_INFO = subprocess.STARTUPINFO()
    _STARTUP_INFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _STARTUP_INFO.wShowWindow = 0  # SW_HIDE


def _to_path_bytes(path: "str | Path") -> bytes:
    """路徑轉 bytes，適合傳給 Windows C DLL（ANSI API 相容）。

    Windows C 函式庫的 fopen() / LoadLibrary() 等 ANSI 函式期望系統碼頁
    （CP936/CP950）編碼的路徑，而 Python 預設 .encode() 是 UTF-8，
    兩者在中文路徑上不相容。

    解法優先順序：
      1. GetShortPathNameW → 8.3 短路徑（純 ASCII，任何 C API 都能處理）
      2. 若 8.3 短路徑仍含非 ASCII → 改用 GetACP() 系統碼頁編碼
      3. 最後回退 UTF-8
    """
    p = str(path)
    if sys.platform != "win32":
        return p.encode("utf-8")
    # 嘗試 GetShortPathNameW 取得 ASCII 8.3 短路徑
    try:
        n = ctypes.windll.kernel32.GetShortPathNameW(p, None, 0)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n)
            if ctypes.windll.kernel32.GetShortPathNameW(p, buf, n) > 0:
                try:
                    return buf.value.encode("ascii")
                except UnicodeEncodeError:
                    p = buf.value   # 短路徑仍有非 ASCII → 繼續往下
    except Exception:
        pass
    # 回退：系統 ANSI 碼頁（C fopen/CreateFileA 期望的編碼）
    try:
        cp = ctypes.windll.kernel32.GetACP()
        return p.encode(f"cp{cp}")
    except (UnicodeEncodeError, LookupError):
        return p.encode("utf-8")


def _short_path_str(path: "str | Path") -> str:
    """回傳 8.3 短路徑字串（盡量 ASCII，用於嵌入 DLL 訊息字串中）。"""
    p = str(path)
    if sys.platform != "win32":
        return p
    try:
        n = ctypes.windll.kernel32.GetShortPathNameW(p, None, 0)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n)
            if ctypes.windll.kernel32.GetShortPathNameW(p, buf, n) > 0:
                return buf.value
    except Exception:
        pass
    return p

# 語系名稱 → ISO 639-1 語言代碼（Qwen3-ASR 輸出格式 "language {code}<asr_text>..."）
_LANG_CODE: dict[str, str] = {
    "Chinese":    "zh",
    "English":    "en",
    "Japanese":   "ja",
    "Korean":     "ko",
    "Cantonese":  "yue",
    "French":     "fr",
    "German":     "de",
    "Spanish":    "es",
    "Portuguese": "pt",
    "Russian":    "ru",
    "Arabic":     "ar",
    "Thai":       "th",
    "Vietnamese": "vi",
    "Indonesian": "id",
    "Malay":      "ms",
    # 中文 UI 標籤（OpenVINO 路線帶來的標籤相容）
    "中文":  "zh",
    "英文":  "en",
    "日文":  "ja",
    "韓文":  "ko",
    "法文":  "fr",
    "德文":  "de",
    "西班牙文": "es",
    "葡萄牙文": "pt",
    "俄文":  "ru",
    "阿拉伯文": "ar",
    "泰文":  "th",
    "越南文": "vi",
}

SRT_DIR = BASE_DIR / "subtitles"


# ══════════════════════════════════════════════════════
# Vulkan 裝置偵測
# ══════════════════════════════════════════════════════

def detect_vulkan_devices(chatllm_dir: str | Path) -> list[dict]:
    """執行 main.exe --show_devices，解析 Vulkan GPU 清單。

    回傳: [{'id': 0, 'name': 'NVIDIA RTX 5090', 'vram_free': 32944160768}, ...]
    失敗時回傳空清單。
    """
    exe = Path(chatllm_dir) / "main.exe"
    if not exe.exists():
        return []
    try:
        result = subprocess.run(
            [str(exe), "--show_devices"],
            capture_output=True, stdin=subprocess.DEVNULL, text=True, timeout=10,
            cwd=str(chatllm_dir),
            creationflags=_CREATE_NO_WINDOW,
            startupinfo=_STARTUP_INFO,
        )
        output = result.stdout + result.stderr
        devices: list[dict] = []
        current: dict = {}

        for line in output.splitlines():
            m = re.match(r"\s*(\d+):\s*\S+\s*-\s*\S+\s*\((.+)\)", line)
            if m:
                if current and current.get("is_gpu"):
                    devices.append({
                        "id":        current["id"],
                        "name":      current["name"],
                        "vram_free": current.get("vram_free", 0),
                    })
                current = {
                    "id":     int(m.group(1)),
                    "name":   m.group(2).strip(),
                    "is_gpu": False,
                }
            elif "type: GPU" in line and current:
                current["is_gpu"] = True
            elif "memory free" in line and current:
                mf = re.search(r"(\d+)\s*B", line)
                if mf:
                    current["vram_free"] = int(mf.group(1))

        if current and current.get("is_gpu"):
            devices.append({
                "id":        current["id"],
                "name":      current["name"],
                "vram_free": current.get("vram_free", 0),
            })

        return devices
    except Exception:
        return []


# ══════════════════════════════════════════════════════
# main.exe 子程序包裝
# ══════════════════════════════════════════════════════

class _ChatLLMRunner:
    """
    以一次性模式執行 main.exe（每個音訊 chunk 一次呼叫）。

    使用 `-mgl main N` 而非 `-ngl N`：
      - Transformer 放 GPU（Vulkan 加速）
      - 音訊 encoder（FFmpeg + GGML audio）留在 CPU（Vulkan 不支援 audio encoder）

    輸出格式：language {lang}<asr_text>{transcription}
    """

    def __init__(
        self,
        model_path:   str | Path,
        chatllm_dir:  str | Path,
        n_gpu_layers: int = 99,
    ):
        self._model_path   = Path(model_path).resolve()   # 必須解析為絕對路徑
        self._chatllm_dir  = Path(chatllm_dir).resolve()
        self._n_gpu_layers = n_gpu_layers
        self._lock         = threading.Lock()

        exe = self._chatllm_dir / "main.exe"
        if not exe.exists():
            raise FileNotFoundError(f"main.exe 不存在：{exe}")
        self._exe = exe

        # 驗證：執行 --show 確認模型可載入
        # 注意：用 -ngl 0 驗證（不上 GPU），避免驗證步驟佔用顯存
        r = subprocess.run(
            [str(exe), "-m", str(self._model_path), "-ngl", "0",
             "--hide_banner", "--show"],
            capture_output=True, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            timeout=30, cwd=str(self._chatllm_dir),
            creationflags=_CREATE_NO_WINDOW,
            startupinfo=_STARTUP_INFO,
        )
        output = r.stdout + r.stderr
        if "Qwen3-ASR" not in output:
            raise RuntimeError(f"模型驗證失敗（rc={r.returncode}）：{output[:300]}")

    def transcribe(self, wav_path: str, sys_prompt: str | None = None) -> str:
        """送入 WAV 路徑（絕對路徑），回傳轉錄文字。"""
        # -ngl all = -ngl 99999,prolog,epilog：把全部 layer（含 audio encoder Conv2D）放 GPU
        # 比 -mgl main N 快 2.7×（GPU 加速 audio encoder + Transformer 兩段）
        gpu_args = ["-ngl", "all"] if self._n_gpu_layers > 0 else ["-ngl", "0"]
        cmd = [
            str(self._exe),
            "-m",    str(self._model_path),
            *gpu_args,
            "--hide_banner",
            "-p",    wav_path,
        ]
        if sys_prompt:
            cmd += ["-s", sys_prompt]

        with self._lock:
            r = subprocess.run(
                cmd,
                capture_output=True, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                timeout=120, cwd=str(self._chatllm_dir),
                creationflags=_CREATE_NO_WINDOW,
                startupinfo=_STARTUP_INFO,
            )
        output = r.stdout + r.stderr

        # 解析 language {lang}<asr_text>{text}
        if "<asr_text>" in output:
            return output.split("<asr_text>", 1)[1].strip()
        return output.strip()


# ══════════════════════════════════════════════════════
# DLL 模式包裝（ctypes，模型常駐記憶體）
# ══════════════════════════════════════════════════════

class _DLLASRRunner:
    """
    libchatllm.dll ctypes 包裝，模型常駐 GPU 記憶體。

    每 chunk 呼叫 transcribe()：
      chatllm_restart → 寫 WAV → chatllm_user_input("{{audio:path}}")
      第一次因 Vulkan shader 編譯約 8s；後續 ~0.23s（43× 實時）
    """

    def __init__(
        self,
        model_path:   str | Path,
        chatllm_dir:  str | Path,
        n_gpu_layers: int = 99,
        cb=None,
    ):
        self._chatllm_dir = Path(chatllm_dir).resolve()
        self._lock        = threading.Lock()

        # ── 凍結視窗模式：預配置隱藏主控台（防止 DLL 閃出黑色視窗）────
        # 問題根源：--windowed PyInstaller EXE 沒有主控台，
        # libchatllm.dll 的 MSVC C runtime 每次呼叫 chatllm_restart() /
        # chatllm_user_input() 寫 stderr/stdout 時，發現 handle 無效，
        # 就會自行呼叫 AllocConsole() 建立主控台視窗（黑色視窗閃爍）。
        # 解法：在 LoadLibrary 前搶先 AllocConsole() 並立即隱藏，
        # 讓 DLL C runtime 找到合法 handle，不再自行建立可見視窗。
        # source 模式（python app.py）從 cmd.exe 繼承主控台，不觸發此問題。
        if getattr(sys, "frozen", False) and sys.platform == "win32":
            _k32 = ctypes.windll.kernel32
            _u32 = ctypes.windll.user32
            if not _k32.GetConsoleWindow():          # 目前無主控台
                if _k32.AllocConsole():              # 分配一個
                    _hwnd = _k32.GetConsoleWindow()
                    if _hwnd:
                        _u32.ShowWindow(_hwnd, 0)    # SW_HIDE：立即隱藏

        dll_path = self._chatllm_dir / "libchatllm.dll"
        if not dll_path.exists():
            raise FileNotFoundError(f"libchatllm.dll 不存在：{dll_path}")

        # ── DLL 相依解析修復（PyInstaller EXE 關鍵）──────────────
        # libchatllm.dll 內部用 plain LoadLibrary("ggml-vulkan.dll")
        # （不帶 LOAD_LIBRARY_SEARCH_* 旗標），走傳統 DLL 搜尋順序：
        #   模組目錄 → CWD → System32 → PATH
        # AddDllDirectory()（os.add_dll_directory）只影響有旗標的 LoadLibraryEx，
        # 對傳統搜尋無效。EXE 的 CWD ≠ chatllm/，PATH 也不含 chatllm/，
        # 所以 ggml-vulkan.dll 等找不到 → DLL 初始化失敗 → fallback subprocess。
        # 解法：暫時把 chatllm_dir 插到 PATH 最前面，chatllm_start() 後還原。
        _saved_path = os.environ.get("PATH", "")
        _chatllm_dir_str = str(self._chatllm_dir)
        os.environ["PATH"] = _chatllm_dir_str + os.pathsep + _saved_path

        os.add_dll_directory(_chatllm_dir_str)
        lib = ctypes.windll.LoadLibrary(str(dll_path))

        # ── 函式原型 ─────────────────────────────────────────────
        PRINTFUNC = ctypes.WINFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)
        ENDFUNC   = ctypes.WINFUNCTYPE(None, ctypes.c_void_p)

        lib.chatllm_append_init_param.argtypes = [ctypes.c_char_p]
        lib.chatllm_append_init_param.restype  = None
        lib.chatllm_init.argtypes              = []
        lib.chatllm_init.restype               = ctypes.c_int
        lib.chatllm_create.argtypes            = []
        lib.chatllm_create.restype             = ctypes.c_void_p
        lib.chatllm_append_param.argtypes      = [ctypes.c_void_p, ctypes.c_char_p]
        lib.chatllm_append_param.restype       = None
        lib.chatllm_start.argtypes             = [ctypes.c_void_p, PRINTFUNC, ENDFUNC, ctypes.c_void_p]
        lib.chatllm_start.restype              = ctypes.c_int
        lib.chatllm_restart.argtypes           = [ctypes.c_void_p, ctypes.c_char_p]
        lib.chatllm_restart.restype            = None
        lib.chatllm_user_input.argtypes        = [ctypes.c_void_p, ctypes.c_char_p]
        lib.chatllm_user_input.restype         = ctypes.c_int

        self._lib       = lib
        self._PRINTFUNC = PRINTFUNC
        self._ENDFUNC   = ENDFUNC

        # ── chatllm 全域初始化（--ggml_dir 告知後端 DLL 位置）────
        # 必須用 ANSI/ASCII 相容的路徑；DLL 的 C 函式庫用 ANSI fopen/LoadLibrary，
        # 若傳 UTF-8 中文路徑會找不到 ggml-*.dll，使用 _to_path_bytes() 解決此問題。
        lib.chatllm_append_init_param(b"--ggml_dir")
        lib.chatllm_append_init_param(_to_path_bytes(self._chatllm_dir))
        r = lib.chatllm_init()
        if r != 0:
            raise RuntimeError(f"chatllm_init() failed: {r}")

        # ── 建立 LLM object ───────────────────────────────────────
        chat = lib.chatllm_create()
        if not chat:
            raise RuntimeError("chatllm_create() returned NULL")
        self._chat = chat

        # ── 模型參數：必須加 --multimedia_file_tags {{ }} ─────────
        # 若缺少此參數，chat->history 的 mm_opening/closing 為空字串，
        # Content::push_back() 會把 {{audio:path}} 當純文字儲存，不做音訊解析。
        # 模型路徑同樣需要 ANSI/ASCII 相容編碼（GetShortPathNameW）。
        gpu_arg = "all" if n_gpu_layers > 0 else "0"
        model_path_bytes = _to_path_bytes(Path(model_path).resolve())
        for p_b in [
            b"-m", model_path_bytes,
            b"-ngl", gpu_arg.encode(),
            b"--multimedia_file_tags", b"{{", b"}}",
        ]:
            lib.chatllm_append_param(chat, p_b)

        # ── 回呼（必須存為 instance attribute 防止 GC 回收）────────
        self._chunks: list[str] = []
        self._error:  str | None = None

        @PRINTFUNC
        def on_print(user_data, print_type, s_ptr):
            text = s_ptr.decode("utf-8", errors="replace") if s_ptr else ""
            if print_type == 0:       # PRINT_CHAT_CHUNK
                self._chunks.append(text)
            elif print_type == 2:     # PRINTLN_ERROR
                self._error = text

        @ENDFUNC
        def on_end(user_data):
            pass

        self._on_print = on_print
        self._on_end   = on_end

        # ── 載入模型（Vulkan 全層 GPU）───────────────────────────
        if cb:
            cb("載入 chatllm 模型（Vulkan GPU，-ngl all）…")
        r = lib.chatllm_start(chat, on_print, on_end, ctypes.c_void_p(0))
        # chatllm_start() 後 DLL 已完全初始化，相依 DLL 也已載入記憶體，可還原 PATH
        os.environ["PATH"] = _saved_path
        if r != 0:
            raise RuntimeError(f"chatllm_start() failed: {r}")

    def transcribe(self, wav_path: str, sys_prompt: str | None = None) -> str:
        """送入 WAV 路徑（絕對路徑），回傳轉錄文字。"""
        # 取得 8.3 短路徑（ASCII），避免中文路徑無法被 DLL 的 C fopen 開啟
        # 例：C:\Users\陳小明\AppData\Local\Temp\xxx.wav
        #   → C:\Users\CHEN~1\AppData\Local\Temp\xxx.wav（純 ASCII）
        safe_path = _short_path_str(str(Path(wav_path).resolve()))
        fwd = safe_path.replace("\\", "/")
        # 若 _short_path_str 仍有非 ASCII（8.3 名稱停用），改用 ANSI 碼頁
        try:
            path_b = fwd.encode("ascii")
        except UnicodeEncodeError:
            cp = ctypes.windll.kernel32.GetACP() if sys.platform == "win32" else 65001
            path_b = fwd.encode(f"cp{cp}", errors="replace")
        msg = b"{{audio:" + path_b + b"}}"
        sys_bytes = sys_prompt.encode("utf-8") if sys_prompt else None

        with self._lock:
            self._lib.chatllm_restart(
                self._chat,
                ctypes.c_char_p(sys_bytes) if sys_bytes else ctypes.c_char_p(None),
            )
            self._chunks.clear()
            self._error = None

            r = self._lib.chatllm_user_input(self._chat, msg)

        if r != 0:
            raise RuntimeError(f"chatllm_user_input() failed: {r}")
        if self._error:
            raise RuntimeError(f"DLL 錯誤：{self._error}")

        full = "".join(self._chunks)
        if "<asr_text>" in full:
            return full.split("<asr_text>", 1)[1].strip()
        return full.strip()


# ══════════════════════════════════════════════════════
# 輔助函式（從 app.py 複製，避免循環 import）
# ══════════════════════════════════════════════════════

def _detect_speech_groups(audio: np.ndarray, vad_sess, max_group_sec: int = MAX_GROUP_SEC):
    h  = np.zeros((2, 1, 64), dtype=np.float32)
    c  = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    n  = len(audio) // VAD_CHUNK
    probs = []
    for i in range(n):
        chunk = audio[i*VAD_CHUNK:(i+1)*VAD_CHUNK].astype(np.float32)[np.newaxis, :]
        out, h, c = vad_sess.run(None, {"input": chunk, "h": h, "c": c, "sr": sr})
        probs.append(float(out[0, 0]))
    if not probs:
        return [(0.0, len(audio) / SAMPLE_RATE, audio)]

    MIN_CH = 16; PAD = 5; MERGE = 16
    raw: list[tuple[int, int]] = []
    in_sp = False; s0 = 0
    for i, p in enumerate(probs):
        if p >= VAD_THRESHOLD and not in_sp:
            s0 = i; in_sp = True
        elif p < VAD_THRESHOLD and in_sp:
            if i - s0 >= MIN_CH:
                raw.append((max(0, s0-PAD), min(n, i+PAD)))
            in_sp = False
    if in_sp and n - s0 >= MIN_CH:
        raw.append((max(0, s0-PAD), n))
    if not raw:
        return []

    merged = [list(raw[0])]
    for s, e in raw[1:]:
        if s - merged[-1][1] <= MERGE:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    mx_samp = max_group_sec * SAMPLE_RATE
    groups: list[tuple[int, int]] = []
    gs = merged[0][0] * VAD_CHUNK
    ge = merged[0][1] * VAD_CHUNK
    for seg in merged[1:]:
        s = seg[0] * VAD_CHUNK; e = seg[1] * VAD_CHUNK
        if e - gs > mx_samp:
            groups.append((gs, ge)); gs = s
        ge = e
    groups.append((gs, ge))

    result = []
    for gs, ge in groups:
        ns = max(1, int((ge - gs) // SAMPLE_RATE))
        ch = audio[gs: gs + ns * SAMPLE_RATE].astype(np.float32)
        if len(ch) < SAMPLE_RATE:
            continue
        result.append((gs / SAMPLE_RATE, gs / SAMPLE_RATE + ns, ch))
    return result


def _split_to_lines(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"[。！？，、；：…—,.!?;:]+", text)
    lines = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        while len(p) > MAX_CHARS:
            lines.append(p[:MAX_CHARS]); p = p[MAX_CHARS:]
        lines.append(p)
    return [l for l in lines if l.strip()]


def _srt_ts(s: float) -> str:
    ms = int(round(s * 1000))
    hh = ms // 3_600_000; ms %= 3_600_000
    mm = ms // 60_000;    ms %= 60_000
    ss = ms // 1_000;     ms %= 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _assign_ts(lines: list[str], g0: float, g1: float) -> list[tuple[float, float, str]]:
    if not lines:
        return []
    total = sum(len(l) for l in lines)
    if total == 0:
        return []
    dur = g1 - g0; res = []; cur = g0
    for i, line in enumerate(lines):
        end = cur + max(MIN_SUB_SEC, dur * len(line) / total)
        if i == len(lines) - 1:
            end = max(end, g1)
        res.append((cur, end, line))
        cur = end + GAP_SEC
    return res


# ══════════════════════════════════════════════════════
# ChatLLMASREngine
# ══════════════════════════════════════════════════════

class ChatLLMASREngine:
    """
    chatllm.cpp + Vulkan 推理後端。

    優先使用 DLL 模式（_DLLASRRunner）：
      - 模型常駐 GPU 記憶體，每 chunk ~0.23s（vs subprocess ~2.5s）
      - 若 libchatllm.dll 不存在，自動回退到 subprocess 模式（_ChatLLMRunner）

    與 ASREngine / ASREngine1p7B 介面相容：
      - max_chunk_secs = 30
      - processor = None（chatllm 不使用 LightProcessor）
      - diar_engine    ← CPU ONNX，與後端無關，照常初始化
      - vad_sess       ← CPU ONNX
      - cc             ← opencc 簡→繁轉換
      - ready
    """

    max_chunk_secs = 30
    processor      = None   # chatllm 不用 LightProcessor，UI 偵測此為 None

    def __init__(self):
        self.ready       = False
        self.vad_sess    = None
        self.diar_engine = None
        self.cc          = None
        self._runner: _DLLASRRunner | _ChatLLMRunner | None = None
        self._use_dll    = False   # 記錄目前使用哪種模式

    # ── 載入 ──────────────────────────────────────────────────────────

    def load(
        self,
        model_path:   str | Path,
        chatllm_dir:  str | Path,
        n_gpu_layers: int = 99,
        cb=None,
    ):
        """從背景執行緒呼叫。cb(msg) 更新 UI 狀態。"""
        import onnxruntime as ort

        def _s(msg):
            if cb:
                cb(msg)

        self._model_path   = Path(model_path)
        self._chatllm_dir  = Path(chatllm_dir)
        self._n_gpu_layers = n_gpu_layers

        # ── VAD ──────────────────────────────────────────────────────
        _s("載入 VAD 模型…")
        vad_candidates = [
            BASE_DIR / "ov_models" / "silero_vad_v4.onnx",
            BASE_DIR / "GPUModel"  / "silero_vad_v4.onnx",
        ]
        # PyInstaller onedir 模式：bundled 資源在 _internal/（sys._MEIPASS）
        import sys as _sys
        if getattr(_sys, "frozen", False) and hasattr(_sys, "_MEIPASS"):
            vad_candidates.insert(0, Path(_sys._MEIPASS) / "ov_models" / "silero_vad_v4.onnx")
        vad_path = next((p for p in vad_candidates if p.exists()), None)
        if vad_path is None:
            raise FileNotFoundError("找不到 silero_vad_v4.onnx")
        self.vad_sess = ort.InferenceSession(
            str(vad_path), providers=["CPUExecutionProvider"]
        )

        # ── 說話者分離（CPU ONNX，與後端無關）───────────────────────
        _s("載入說話者分離模型…")
        try:
            from diarize import DiarizationEngine
            diar_candidates = [
                BASE_DIR / "ov_models" / "diarization",
                BASE_DIR / "GPUModel"  / "diarization",
            ]
            diar_dir = next((p for p in diar_candidates if p.exists()), None)
            if diar_dir:
                eng = DiarizationEngine(diar_dir)
                self.diar_engine = eng if eng.ready else None
        except Exception:
            self.diar_engine = None

        # ── OpenCC 簡→繁轉換 ──────────────────────────────────────
        try:
            import opencc
            self.cc = opencc.OpenCC("s2twp")
        except Exception:
            self.cc = None

        # ── 驗證路徑 ─────────────────────────────────────────────────
        if not self._model_path.exists():
            raise FileNotFoundError(f"模型不存在：{self._model_path}")

        # ── 建立 Runner：優先 DLL，後備 subprocess ───────────────
        dll_path = self._chatllm_dir / "libchatllm.dll"
        if dll_path.exists():
            try:
                _s("載入 chatllm 模型（DLL 模式，Vulkan 全層 GPU）…")
                self._runner = _DLLASRRunner(
                    model_path   = model_path,
                    chatllm_dir  = chatllm_dir,
                    n_gpu_layers = n_gpu_layers,
                    cb           = cb,
                )
                self._use_dll = True
                self.ready = True
                _s("ChatLLM DLL 載入完成（模型常駐 GPU，每 chunk ~0.23s）")
                return
            except Exception as e:
                _s(f"DLL 模式失敗（{e}），改用 subprocess 模式…")

        _s("驗證 chatllm 模型（subprocess 模式）…")
        self._runner = _ChatLLMRunner(
            model_path   = model_path,
            chatllm_dir  = chatllm_dir,
            n_gpu_layers = n_gpu_layers,
        )
        self._use_dll = False
        self.ready = True
        _s("ChatLLM 載入完成（subprocess 模式，Vulkan GPU）")

    # ── 單段轉錄 ──────────────────────────────────────────────────────

    def transcribe(
        self,
        audio:      np.ndarray,
        sr:         int = SAMPLE_RATE,
        language:   str | None = None,
        context:    str | None = None,
        max_tokens: int = 300,
    ) -> str:
        """16kHz float32 → 轉錄文字。"""
        import soundfile as sf

        # 語系 → system prompt
        # Qwen3-ASR 輸出格式：language {code}<asr_text>{text}
        # 透過 sys_prompt 明確指定語言代碼，引導模型用正確語言輸出。
        sys_prompt: str | None = None
        if language and language != "自動偵測":
            code = _LANG_CODE.get(language, language.lower()[:2])
            sys_prompt = (
                f"The audio language is {language}. "
                f"Transcribe it and output strictly in this format: "
                f"language {code}<asr_text>[transcription]. "
                f"Output only {language} text after <asr_text>, no translation."
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
            text = self._runner.transcribe(tmp_path, sys_prompt=sys_prompt)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        # OpenCC 簡→繁轉換（模型預設輸出簡體中文）
        if self.cc and text:
            text = self.cc.convert(text)

        return text

    # ── chunk 長度限制 ─────────────────────────────────────────────────

    def _enforce_chunk_limit(
        self,
        groups: list[tuple[float, float, np.ndarray, "str | None"]],
    ) -> list[tuple[float, float, np.ndarray, "str | None"]]:
        max_samples = self.max_chunk_secs * SAMPLE_RATE
        result = []
        for t0, t1, chunk, spk in groups:
            if len(chunk) <= max_samples:
                result.append((t0, t1, chunk, spk))
            else:
                pos = 0
                while pos < len(chunk):
                    piece = chunk[pos: pos + max_samples]
                    if len(piece) < SAMPLE_RATE:
                        break
                    piece_t0 = t0 + pos / SAMPLE_RATE
                    piece_t1 = min(t1, piece_t0 + len(piece) / SAMPLE_RATE)
                    result.append((piece_t0, piece_t1, piece, spk))
                    pos += max_samples
        return result

    # ── 音檔轉 SRT ─────────────────────────────────────────────────────

    def process_file(
        self,
        audio_path: Path,
        progress_cb=None,
        language:   str | None = None,
        context:    str | None = None,
        diarize:    bool = False,
        n_speakers: int | None = None,
    ) -> Path | None:
        import librosa

        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

        # ── 分段策略：說話者分離 vs 傳統 VAD（與 ASREngine 一致）────
        use_diar = diarize and self.diar_engine is not None and self.diar_engine.ready
        if use_diar:
            diar_segs = self.diar_engine.diarize(audio, n_speakers=n_speakers)
            if not diar_segs:
                return None
            groups_spk = [
                (t0, t1,
                 audio[int(t0 * SAMPLE_RATE): int(t1 * SAMPLE_RATE)],
                 spk)
                for t0, t1, spk in diar_segs
            ]
        else:
            vad_groups = _detect_speech_groups(audio, self.vad_sess, self.max_chunk_secs)
            if not vad_groups:
                return None
            groups_spk = [(g0, g1, chunk, None) for g0, g1, chunk in vad_groups]

        groups_spk = self._enforce_chunk_limit(groups_spk)

        all_subs: list[tuple[float, float, str, str | None]] = []
        total = len(groups_spk)
        for i, (g0, g1, chunk, spk) in enumerate(groups_spk):
            if progress_cb:
                spk_info = f" [{spk}]" if spk else ""
                progress_cb(i, total, f"[{i+1}/{total}] {g0:.1f}s~{g1:.1f}s{spk_info}")
            text = self.transcribe(chunk, language=language, context=context)
            if not text:
                continue
            lines = _split_to_lines(text)
            all_subs.extend(
                (s, e, line, spk) for s, e, line in _assign_ts(lines, g0, g1)
            )

        if not all_subs:
            return None

        if progress_cb:
            progress_cb(total, total, "寫入 SRT…")

        SRT_DIR.mkdir(exist_ok=True)
        out = SRT_DIR / (audio_path.stem + ".srt")
        with open(out, "w", encoding="utf-8") as f:
            for idx, (s, e, line, spk) in enumerate(all_subs, 1):
                prefix = f"{spk}：" if spk else ""
                f.write(f"{idx}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{prefix}{line}\n\n")
        return out

    def __del__(self):
        pass   # DLL runner 由 GC 自然回收（ctypes callback 會被 GC 清理）
