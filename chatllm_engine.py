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
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ── 輸出語系旗標（由 app.py / app-gpu.py 切換時同步設定）──────────────
# True = 直接輸出模型原始簡體；False = 經 OpenCC 轉為繁體
_output_simplified: bool = False

# 繁體輸出時是否啟用「簡繁詞彙轉換」：
#   True  → OpenCC "s2twp"（含台灣慣用詞）；False → "s2t"（僅字形）
_vocab_convert: bool = True


def _opencc_config() -> str:
    return "s2twp" if _vocab_convert else "s2t"

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

def probe_vulkan_devices(chatllm_dir: str | Path) -> dict:
    """執行 main.exe --show_devices 並回傳「完整診斷」，供 UI 明確告知使用者。

    與 detect_vulkan_devices() 不同：本函式不吞錯誤，而是把整個探測過程
    的結果都帶回來，讓上層能分辨「真的沒有 GPU」與「chatllm 列舉失敗
    （超時 / 崩潰 / 卡在內顯）」這兩種完全不同的情況。

    輸出格式（每裝置兩行）：
      0: Vulkan - VulkanO (AMD Radeon(TM) Graphics)
         type: ACCEL
         memory free: 7957908736 B
      1: CPU - CPU (AMD Ryzen 5 9600X 6-Core Processor)
         type: CPU

    回傳 dict：
      {
        "devices":     [...],     # 非 CPU 計算裝置（供推理選用）
        "all_devices": [...],     # 全部列舉到的裝置（含 CPU/iGPU，供診斷）
        "raw":         str,       # --show_devices 原始輸出（stdout+stderr）
        "error":       str|None,  # 例外/超時訊息（None 表示正常結束）
        "exit_code":   int|None,  # main.exe 結束碼
        "exe_found":   bool,      # 是否找得到 main.exe
      }
    每個裝置 dict：{'id', 'name', 'vram_free', 'backend', 'is_cpu'}
    """
    exe = Path(chatllm_dir) / "main.exe"
    diag: dict = {
        "devices": [], "all_devices": [], "raw": "",
        "error": None, "exit_code": None, "exe_found": exe.exists(),
    }
    if not exe.exists():
        diag["error"] = f"找不到 main.exe：{exe}"
        return diag
    try:
        result = subprocess.run(
            [str(exe), "--show_devices"],
            capture_output=True, stdin=subprocess.DEVNULL, text=True, timeout=10,
            cwd=str(chatllm_dir),
            creationflags=_CREATE_NO_WINDOW,
            startupinfo=_STARTUP_INFO,
        )
        output = result.stdout + result.stderr
        diag["raw"] = output
        diag["exit_code"] = result.returncode

        pending: list[dict] = []
        current: dict | None = None
        for line in output.splitlines():
            # 裝置標頭行：「0: Vulkan - VulkanO (AMD Radeon(TM) Graphics)」
            m = re.match(r"\s*(\d+):\s*(\S+)\s+-\s+\S+\s+\((.+)\)", line)
            if m:
                backend = m.group(2).upper()   # "VULKAN", "CPU", "METAL" …
                current = {
                    "id":        int(m.group(1)),
                    "name":      m.group(3).strip(),
                    "vram_free": 0,
                    "backend":   backend,
                    "is_cpu":    backend == "CPU",
                }
                pending.append(current)
            elif "memory free" in line and current is not None:
                mf = re.search(r"(\d+)\s*B", line)
                if mf:
                    current["vram_free"] = int(mf.group(1))

        diag["all_devices"] = pending
        diag["devices"] = [d for d in pending if not d["is_cpu"]]
    except subprocess.TimeoutExpired:
        diag["error"] = (
            "main.exe --show_devices 逾時（10 秒）。"
            "常見於筆電多 GPU 環境，Vulkan 初始化卡在內建顯示卡。"
        )
    except Exception as e:
        diag["error"] = f"{type(e).__name__}: {e}"
    return diag


def detect_vulkan_devices(chatllm_dir: str | Path) -> list[dict]:
    """執行 main.exe --show_devices，解析所有非 CPU 的計算裝置（薄包裝）。

    保留舊簽名以維持向後相容；完整診斷請改用 probe_vulkan_devices()。
    回傳: [{'id', 'name', 'vram_free'}, ...]，失敗時回傳空清單。
    """
    diag = probe_vulkan_devices(chatllm_dir)
    return [
        {"id": d["id"], "name": d["name"], "vram_free": d["vram_free"]}
        for d in diag["devices"]
    ]


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
        device_id:    int = 0,
    ):
        self._model_path   = Path(model_path).resolve()   # 必須解析為絕對路徑
        self._chatllm_dir  = Path(chatllm_dir).resolve()
        self._n_gpu_layers = n_gpu_layers
        self._device_id    = device_id
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
        # -ngl {id}:all = 指定裝置 id + 全部 layer（含 audio encoder Conv2D）放 GPU
        # 比 -mgl main N 快 2.7×（GPU 加速 audio encoder + Transformer 兩段）
        gpu_args = ["-ngl", f"{self._device_id}:all"] if self._n_gpu_layers > 0 else ["-ngl", "0"]
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

        # 正常輸出必含 <asr_text>；若缺失代表裝置錯誤，立即中止，不回傳垃圾字幕
        if "<asr_text>" not in output:
            preview = output.strip()[:300] or "(無輸出)"
            raise RuntimeError(
                f"GPU 推理失敗，未取得語音輸出。\n"
                f"可能原因：裝置不相容、模型錯誤或記憶體不足。\n"
                f"chatllm 輸出：{preview}"
            )
        text = output.split("<asr_text>", 1)[1].strip()
        deg = detect_degenerate_asr(text)
        if deg:
            raise RuntimeError(
                f"GPU 推理退化：{deg}。\n"
                f"此裝置／核心可能不相容，建議改用 CPU(OpenVINO) 核心。\n"
                f"模型輸出：{text[:200]}"
            )
        return text


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
        device_id:    int = 0,
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
        # 帶入 device_id：「1:all」= 裝置 1 全層 GPU，「0:all」= 預設裝置 0
        # chatllm.cpp -ngl 語法：one_spec ::= [id:]spec
        gpu_arg = f"{device_id}:all" if n_gpu_layers > 0 else "0"
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
        if "<asr_text>" not in full:
            preview = full.strip()[:300] or "(無輸出)"
            raise RuntimeError(
                f"GPU 推理失敗，未取得語音輸出。\n"
                f"可能原因：裝置不相容、模型錯誤或記憶體不足。\n"
                f"DLL 輸出：{preview}"
            )
        text = full.split("<asr_text>", 1)[1].strip()
        deg = detect_degenerate_asr(text)
        if deg:
            raise RuntimeError(
                f"GPU 推理退化：{deg}。\n"
                f"此裝置／核心可能不相容，建議改用 CPU(OpenVINO) 核心。\n"
                f"模型輸出：{text[:200]}"
            )
        return text


# ══════════════════════════════════════════════════════
# 輔助函式（從 app.py 複製，避免循環 import）
# ══════════════════════════════════════════════════════

def _detect_speech_groups(audio: np.ndarray, vad_sess, max_group_sec: int = MAX_GROUP_SEC,
                          stats: dict | None = None):
    h  = np.zeros((2, 1, 64), dtype=np.float32)
    c  = np.zeros((2, 1, 64), dtype=np.float32)
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    n  = len(audio) // VAD_CHUNK
    probs = []
    for i in range(n):
        chunk = audio[i*VAD_CHUNK:(i+1)*VAD_CHUNK].astype(np.float32)[np.newaxis, :]
        out, h, c = vad_sess.run(None, {"input": chunk, "h": h, "c": c, "sr": sr})
        probs.append(float(out[0, 0]))
    if stats is not None:
        stats["n_chunks"]  = len(probs)
        stats["max_prob"]  = max(probs) if probs else 0.0
        stats["mean_prob"] = (sum(probs) / len(probs)) if probs else 0.0
        stats["threshold"] = VAD_THRESHOLD
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
    if stats is not None:
        stats["n_segments"] = len(raw)
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
    _tail = int(0.35 * SAMPLE_RATE)   # 多含尾段 0.35s：補捉 Silero 漏掉的句尾輕聲字（的/了/嗎）
    for gs, ge in groups:
        # 取完整語音段並多含一點尾巴（mel 會補零到固定長度）；舊版 floor 會丟尾段
        # 近 1 秒，句尾輕聲字也常被 Silero 的 ge 排除，故補 0.35s 尾段。
        ch = audio[gs: min(len(audio), ge + _tail)].astype(np.float32)
        if len(ch) < SAMPLE_RATE // 2:      # 最小 0.5 秒
            continue
        result.append((gs / SAMPLE_RATE, ge / SAMPLE_RATE, ch))
    return result


def format_vad_diag(stats: dict) -> str:
    """把 _detect_speech_groups 的 stats 轉成「為什麼沒偵測到人聲」的明確說明。

    區分三種情況：音檔過短 / 全段機率過低（純背景音或門檻太高）/ 有訊號但段落太短。
    """
    n    = stats.get("n_chunks", 0)
    mp   = stats.get("max_prob", 0.0)
    th   = stats.get("threshold", VAD_THRESHOLD)
    nseg = stats.get("n_segments", 0)
    if n == 0:
        return "⚠ 未偵測到人聲：音訊過短或為空檔。"
    if mp < th:
        return (f"⚠ 未偵測到人聲：全段人聲機率偏低（最高 {mp:.2f} < 門檻 {th:.2f}）。"
                f"可能為純背景音／音樂，或門檻過高 —— 可至『設定 → VAD 靈敏度』調低後重試。")
    if nseg == 0:
        return (f"⚠ 未偵測到人聲：偵測到語音訊號（最高 {mp:.2f}）但每段都過短（< 0.5 秒）未成段。")
    return "⚠ 未偵測到人聲：無有效語音分段。"


def detect_degenerate_asr(text: str) -> str | None:
    """偵測 ASR 退化輸出（模型複誦提示範本 / 高度重複），回傳原因字串；正常回 None。

    chatllm 在部分不相容 GPU（常見 AMD Vulkan）上會吐出複誦 system prompt 的垃圾，
    例如 'language en<asr_text>[transcription]' 或同字無限重複。這類內容不該被當成
    字幕寫出，需明確判為推理失敗（issue #30）。
    """
    t = (text or "").strip()
    if not t:
        return None   # 空字串交由上層「未偵測到人聲／跳過」邏輯處理
    low = t.lower()
    # 1) 殘留 prompt 範本 token（最明確的退化訊號）
    for tok in ("<asr_text", "asr_text>", "[transcription]", "[asr_text]"):
        if tok in low:
            return "模型輸出殘留提示範本（疑似未正常辨識）"
    # 2) 整段就是語言標記 'language xx'（exact，避免誤判 'Language is power'）
    if re.fullmatch(r"language\s+[a-z]{2}", low):
        return "模型僅複誦語言標記，未產生轉錄內容"
    if low.count("language ") >= 3:
        return "模型反覆複誦語言標記（疑似推理退化）"
    # 3) 單一字元高度重複
    compact = re.sub(r"\s+", "", t)
    if len(compact) >= 12 and max(Counter(compact).values()) / len(compact) >= 0.85:
        return "模型輸出高度重複（疑似推理退化）"
    return None


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
        self.aligner     = None   # 相容旗標（chatllm FA 就緒時為 True）
        self.use_aligner = False  # 是否啟用時間軸對齊
        self._fa         = None   # ChatLLMAligner（與 OpenVINO 引擎共用）
        self._fa_bin     = None   # FA .bin 路徑（就緒時非 None）

    # ── 載入 ──────────────────────────────────────────────────────────

    def load(
        self,
        model_path:   str | Path,
        chatllm_dir:  str | Path,
        n_gpu_layers: int = 99,
        device_id:    int = 0,
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
        self._device_id    = device_id   # FA 對齊的 -ngl {id}:all 需要

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
            self.cc = opencc.OpenCC(_opencc_config())
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
                    device_id    = device_id,
                    cb           = cb,
                )
                self._use_dll = True
                self.ready = True
                _s("ChatLLM DLL 載入完成（模型常駐 GPU，每 chunk ~0.23s）")

                # ── ForcedAligner（可選，CPU PyTorch，不需 CUDA）────────
                self._load_aligner(cb=cb)
                return
            except Exception as e:
                _s(f"DLL 模式失敗（{e}），改用 subprocess 模式…")

        _s("驗證 chatllm 模型（subprocess 模式）…")
        self._runner = _ChatLLMRunner(
            model_path   = model_path,
            chatllm_dir  = chatllm_dir,
            n_gpu_layers = n_gpu_layers,
            device_id    = device_id,
        )
        self._use_dll = False
        self.ready = True
        _s("ChatLLM 載入完成（subprocess 模式，Vulkan GPU）")

        # ── ForcedAligner（可選，CPU PyTorch，不需 CUDA）──────────────
        self._load_aligner(cb=cb)

    # ── 單段轉錄 ──────────────────────────────────────────────────────

    def rebuild_cc(self):
        """依目前的詞彙轉換旗標重建 OpenCC 轉換器（免重新載入模型）。"""
        try:
            import opencc
            self.cc = opencc.OpenCC(_opencc_config())
        except Exception:
            pass

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

        # OpenCC 簡→繁轉換（模型預設輸出簡體中文；簡體模式則跳過）
        if self.cc and text and not _output_simplified:
            text = self.cc.convert(text)

        return text

    # ── ForcedAligner 載入 ────────────────────────────────────────────

    # FA chatllm 模型檔名（與 ASR .bin 同資料夾）
    FA_BIN_NAME = "qwen3-focedaligner-0.6b.bin"

    def _load_aligner(self, cb=None):
        """偵測 chatllm 版 ForcedAligner .bin（委派共用 fa_aligner，無需 torch）。

        與 OpenVINO 引擎共用 fa_aligner.ChatLLMAligner：給「參考文字 + 音訊」
        即可輸出字級毫秒時間軸。GPU 用 -ngl {id}:all、CPU 用 -ngl 0。
        缺 .bin 或驗證失敗時靜默退回比例估算。
        """
        from fa_aligner import ChatLLMAligner
        self.aligner     = None
        self.use_aligner = False
        self._fa_bin     = None
        # FA .bin 與 ASR .bin 放同一資料夾（尊重使用者指定的 model_dir）
        # ── FA 強制 CPU 子程序（n_gpu_layers=0）──────────────────────────
        # 致命關鍵：ASR 走常駐 libchatllm.dll，Vulkan context 永不釋放。
        # 若 FA 也用 -ngl {id}:all 上 GPU，main.exe 子程序會在 DLL 仍持有
        # context 時建立「第二個 Vulkan context」→ 同卡雙 context 並存 →
        # 驅動 TDR/reset → 整機當機（1.0.8 起的潛伏 bug，見 vulkan-dual-context-crash）。
        # FA 是 0.6B，CPU 對齊本就夠快；改 CPU 後全程僅一個 GPU context（DLL）。
        self._fa = ChatLLMAligner(
            fa_dir       = self._model_path.parent,
            chatllm_dir  = self._chatllm_dir,
            n_gpu_layers = 0,
            device_id    = self._device_id,
        )
        if self._fa.load(cb=cb):
            self._fa_bin     = self._fa._fa_bin
            self.aligner     = True   # 相容旗標（沿用 use_aligner 判斷流程）
            self.use_aligner = True

    def _align_chunk(
        self,
        wav_path:  str,
        ref_text:  str,
        language:  str = "Chinese",
    ) -> list[tuple[str, float, float]]:
        """委派 chatllm FA 對齊，回傳字級 [(word, start_s, end_s), ...]（失敗回 []）。"""
        if self._fa is None or not self._fa_bin:
            return []
        return self._fa.align_chunk(wav_path, ref_text, language)

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
        original_path: Path | None = None,
        out_format: str | None = None,
    ) -> Path | None:
        from audio_io import load_audio_16k_mono
        from subtitle_lines import write_transcript

        audio, _ = load_audio_16k_mono(audio_path, SAMPLE_RATE)

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
            if self.vad_sess is None:
                self._last_vad_diag = "⚠ VAD 模型（silero_vad）未載入，無法分段"
                if progress_cb:
                    progress_cb(0, 1, self._last_vad_diag)
                return None
            _vad_stats: dict = {}
            vad_groups = _detect_speech_groups(
                audio, self.vad_sess, self.max_chunk_secs, stats=_vad_stats)
            if not vad_groups:
                self._last_vad_diag = format_vad_diag(_vad_stats)
                if progress_cb:
                    progress_cb(0, 1, self._last_vad_diag)
                return None
            groups_spk = [(g0, g1, chunk, None) for g0, g1, chunk in vad_groups]

        groups_spk = self._enforce_chunk_limit(groups_spk)

        # ── 導入 chatllm 版斷句函式（避免循環 import，延遲導入）─────────
        _ts_fn = None
        if self.use_aligner and self._fa_bin is not None:
            try:
                from app import _ts_chatllm_to_subtitle_lines
                _ts_fn = _ts_chatllm_to_subtitle_lines
            except ImportError:
                pass

        all_subs: list[tuple[float, float, str, str | None]] = []
        total = len(groups_spk)
        for i, (g0, g1, chunk, spk) in enumerate(groups_spk):
            if progress_cb:
                spk_info = f" [{spk}]" if spk else ""
                progress_cb(i, total, f"[{i+1}/{total}] {g0:.1f}s~{g1:.1f}s{spk_info}")

            # ── 轉錄（取原始簡體輸出，對齊後再繁化）─────────────────────
            import soundfile as sf
            sys_prompt: str | None = None
            if language and language != "自動偵測":
                code = _LANG_CODE.get(language, language.lower()[:2])
                sys_prompt = (
                    f"The audio language is {language}. "
                    f"Transcribe it and output strictly in this format: "
                    f"language {code}<asr_text>[transcription]. "
                    f"Output only {language} text after <asr_text>, no translation."
                )

            import tempfile as _tempfile
            with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_path = tf.name
            aligned  = False
            raw_text = ""
            try:
                sf.write(tmp_path, chunk, SAMPLE_RATE, subtype="PCM_16")
                raw_text = self._runner.transcribe(tmp_path, sys_prompt=sys_prompt)

                if not raw_text:
                    continue

                # ── ForcedAligner 精確時間軸對齊（chatllm .bin，無需 torch）──
                #    重用同一個暫存 wav，避免重複寫檔。
                if self.use_aligner and self._fa_bin is not None and _ts_fn is not None:
                    try:
                        align_lang = (language
                                      if (language and language != "自動偵測")
                                      else "Chinese")
                        ts_items = self._align_chunk(tmp_path, raw_text, align_lang)
                        if ts_items:
                            subs = _ts_fn(
                                ts_items, raw_text, g0, spk,
                                self.cc, _output_simplified,
                            )
                            if subs:
                                all_subs.extend(subs)
                                aligned = True
                    except Exception:
                        aligned = False
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if not raw_text:
                continue

            if not aligned:
                # ── 比例估算 Fallback ──────────────────────────────────────
                text = raw_text
                if self.cc and not _output_simplified:
                    text = self.cc.convert(raw_text)
                lines = _split_to_lines(text)
                all_subs.extend(
                    (s, e, line, spk) for s, e, line in _assign_ts(lines, g0, g1)
                )

        if not all_subs:
            return None

        if progress_cb:
            progress_cb(total, total, "寫入字幕…")

        # 以原始檔案的目錄與檔名輸出（影片抽音軌時 audio_path 是暫存路徑）
        # 共用寫出層：依全域設定（或 out_format 覆寫）產出 .srt 或 .txt。
        ref = original_path if original_path is not None else audio_path
        return write_transcript(ref, all_subs, out_format)

    def __del__(self):
        pass   # DLL runner 由 GC 自然回收（ctypes callback 會被 GC 清理）

