"""
crisp_engine.py — CrispASR (ggml / Vulkan) Whisper 推理後端

定位：把 CrispASR 的 whisper backend 當作本專案的第三個推理引擎（與
ASREngine[OpenVINO] / ChatLLMASREngine[chatllm] 並列），用來載入 Whisper
類模型（預設 Breeze-ASR-26，繁中/台語特化）。

設計重點：
  • CrispASR 是 whisper.cpp fork，crispasr.exe 自己會輸出 SRT（含分段時間軸），
    因此本引擎不需重造 VAD/chunk/FA，本質是「呼叫 crispasr.exe → 讀 SRT →
    OpenCC 繁化 → 寫回」。
  • GPU 後端由 load(gpu_backend=) 決定，對應「使用者選的 CrispASR 加速版本」
    （downloader.crispasr_gpu_backend）：
      - vulkan（預設）：Windows 通吃 Intel/AMD 內顯與 NVIDIA 獨顯，體積最小。
      - cuda：NVIDIA 專用，需下載 CUDA build（0.8.32 起在 Blackwell 實測正常；
        早期 0.7.1 的 fattn crash 為 nemotron 後端專屬問題）。
      - None：CPU build，改掛 -ng（該 build 沒有 ggml-vulkan/cuda DLL）。
  • 介面對齊 app.py 既有引擎契約：load / ready / transcribe / process_file /
    processor / diar_engine / use_aligner / aligner / _lock / rebuild_cc。

對外介面（app.py 的 _load_models[crispasr 分支] 會這樣呼叫）：
    eng = CrispWhisperEngine()
    eng.load(model_path=..., crispasr_dir=..., device_id=0, cb=set_status)
    srt = eng.process_file(audio_path, progress_cb=..., language=..., ...)
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

# 與 Qwen 路徑統一的字幕分行（字級時間軸 → 字幕行，全引擎共用）
from subtitle_lines import (_ts_chatllm_to_subtitle_lines, _srt_ts, write_transcript,
                            _ZH_CLAUSE_END, _EN_SENT_END)

# 中文/英文標點集合（與斷句器一致）：Qwen 模式下標點只當切點、不進 items
_PUNCT = _ZH_CLAUSE_END | _EN_SENT_END

# ── 輸出語系旗標（由 app.py 切換時同步設定，與 chatllm_engine 行為一致）──
_output_simplified: bool = False   # True=輸出模型原始；False=OpenCC 繁化
_vocab_convert:     bool = True    # True=s2twp(含台灣詞)；False=s2t(僅字形)


def _opencc_config() -> str:
    return "s2twp" if _vocab_convert else "s2t"


# ── 共用常數 ──────────────────────────────────────────────────────────
SAMPLE_RATE = 16000

if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

# Windows：隱藏子程序主控台視窗（避免辨識時畫面閃爍）
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_STARTUP_INFO: "subprocess.STARTUPINFO | None" = None
if sys.platform == "win32":
    _STARTUP_INFO = subprocess.STARTUPINFO()
    _STARTUP_INFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _STARTUP_INFO.wShowWindow = 0  # SW_HIDE

# 預設 crispasr 目錄（仿 chatllm/ 慣例）與模型檔名
_DEFAULT_CRISPASR_DIR = BASE_DIR / "crispasr"
_DEFAULT_MODEL_NAME   = "ggml-model-q5_0.bin"   # Breeze-ASR-26 Q5 標準（HF 原名）
# qwen3 ForcedAligner GGUF（FA 對齊器）檔名樣式，置於 crispasr_dir 同層
_ALIGNER_GLOB = "qwen3-forced-aligner-0.6b-*.gguf"


def _find_aligner_gguf(crispasr_dir: Path) -> Path | None:
    """在 crispasr 目錄尋找 qwen3 ForcedAligner GGUF（取第一個符合者）。"""
    for g in sorted(crispasr_dir.glob(_ALIGNER_GLOB)):
        if g.is_file():
            return g
    return None


def _infer_backend(model_path: Path | str) -> str:
    """依模型檔名推斷 crispasr 後端（--backend）。

    CrispASR 是多後端 runtime（whisper / qwen3 / parakeet …），同一支
    crispasr.exe 靠 `--backend` 切換模型架構。Breeze-ASR-26 屬 whisper 架構
    （預設，免旗標）；Qwen3-ASR GGUF 檔名含 'qwen3' → 走 qwen3 後端，1.7B
    用 'qwen3-1.7b'。判斷錯誤時 crispasr 會自報模型不符，不致默默產生垃圾。
    """
    name = Path(model_path).name.lower()
    if "qwen3" in name:
        return "qwen3-1.7b" if "1.7b" in name else "qwen3"
    # TEA-ASR = Qwen3-ASR-1.7B 的台灣國語微調，架構相同故共用 qwen3-1.7b 後端
    if "tea-asr" in name or "tea_asr" in name:
        return "qwen3-1.7b"
    return "whisper"

# app.py 傳入的 language（如 "Chinese" / "自動偵測"）→ whisper 語言代碼
_LANG_MAP = {
    "自動偵測": None, "auto": None, "": None,
    "Chinese": "zh", "中文": "zh", "Mandarin": "zh",
    "English": "en", "Japanese": "ja", "Korean": "ko",
    "Cantonese": "yue", "French": "fr", "German": "de",
    "Spanish": "es", "Portuguese": "pt", "Russian": "ru",
    "Arabic": "ar", "Thai": "th", "Vietnamese": "vi",
    "Indonesian": "id", "Malay": "ms",
}

# 本引擎對外宣告的常用語系清單（app.py 在 processor 為 None 時改用此清單）
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Japanese", "Korean", "Cantonese",
    "French", "German", "Spanish", "Portuguese", "Russian",
    "Arabic", "Thai", "Vietnamese", "Indonesian", "Malay",
]


def _find_exe(crispasr_dir: Path) -> Path | None:
    """在 crispasr 目錄（含一層子資料夾）尋找 crispasr.exe。"""
    direct = crispasr_dir / "crispasr.exe"
    if direct.exists():
        return direct
    for sub in crispasr_dir.glob("**/crispasr.exe"):
        return sub
    return None


def probe_crispasr_devices(crispasr_dir: str | Path) -> dict:
    """執行 `crispasr.exe --diagnostics` 並解析計算裝置清單（含 GPU 記憶體）。

    這是 chatllm `probe_vulkan_devices()` 的對應物，但走 CrispASR 自帶的
    `--diagnostics`（會列舉 ggml 後端與裝置後立即退出，不需模型/音檔）。
    讓裝置偵測**不再依賴 chatllm**（main.exe 未隨安裝包提供時也能偵測 GPU）。

    解析目標是 diagnostics 輸出中的「registered devices」區塊，例如：
        registered devices : 2
          [0] gpu    name=Vulkan0 desc=NVIDIA GeForce RTX 5090 mem=31418/32186 MiB id=0000:01:00.0
          [1] cpu    name=CPU desc=13th Gen Intel(R) Core(TM) i5-13500 mem=79978/98045 MiB id=?
    每行 → {'id','name','vram_free'(bytes),'backend','is_cpu'}；mem 為 free/total MiB。

    回傳 dict 與 probe_vulkan_devices() 同結構：
      {devices(非CPU), all_devices(全部), raw, error, exit_code, exe_found}
    """
    crispasr_dir = Path(crispasr_dir)
    exe = _find_exe(crispasr_dir)
    diag: dict = {
        "devices": [], "all_devices": [], "raw": "",
        "error": None, "exit_code": None, "exe_found": bool(exe),
    }
    if not exe:
        diag["error"] = f"找不到 crispasr.exe：{crispasr_dir}"
        return diag
    try:
        result = subprocess.run(
            [str(exe), "--diagnostics"],
            capture_output=True, stdin=subprocess.DEVNULL, text=True,
            encoding="utf-8", errors="replace", timeout=15,
            cwd=str(exe.parent),
            creationflags=_CREATE_NO_WINDOW,
            startupinfo=_STARTUP_INFO,
        )
        output = (result.stdout or "") + (result.stderr or "")
        diag["raw"] = output
        diag["exit_code"] = result.returncode

        # 「registered devices」逐行：[idx] <type> name=.. desc=<名稱> mem=free/total MiB id=..
        dev_re = re.compile(
            r"^\s*\[(\d+)\]\s+(\w+)\s+name=(\S+)\s+desc=(.+?)\s+mem=(\d+)/(\d+)\s*MiB",
            re.IGNORECASE)
        pending: list[dict] = []
        for line in output.splitlines():
            m = dev_re.match(line)
            if not m:
                continue
            kind = m.group(2).lower()            # gpu / cpu
            is_cpu = kind == "cpu"
            pending.append({
                "id":        int(m.group(1)),
                "name":      m.group(4).strip(),                 # 人類可讀名稱
                "vram_free": int(m.group(5)) * 1024 * 1024,      # MiB → bytes
                "backend":   "CPU" if is_cpu else "VULKAN",
                "is_cpu":    is_cpu,
            })

        # 備援：若無 registered devices 區塊，改解析 ggml_vulkan 列舉行
        #   「ggml_vulkan: 0 = NVIDIA GeForce RTX 5090 (NVIDIA) | uma: 0 | ...」
        if not pending:
            vk_re = re.compile(r"ggml_vulkan:\s*(\d+)\s*=\s*(.+?)\s*\((.+?)\)\s*\|")
            for line in output.splitlines():
                m = vk_re.match(line)
                if m:
                    pending.append({
                        "id": int(m.group(1)), "name": m.group(2).strip(),
                        "vram_free": 0, "backend": "VULKAN", "is_cpu": False,
                    })

        diag["all_devices"] = pending
        diag["devices"] = [d for d in pending if not d["is_cpu"]]
    except subprocess.TimeoutExpired:
        diag["error"] = ("crispasr.exe --diagnostics 逾時（15 秒）。"
                         "Vulkan 初始化可能卡在內建顯示卡。")
    except Exception as e:
        diag["error"] = f"{type(e).__name__}: {e}"
    return diag


# ── 與 CTC 對齊器不相容的模型 ──────────────────────────────────────────
#   `qwen3-forced-aligner-0.6b` 是「把 vocabulary head 換成 5000 類 timestamp
#   head」的對齊器，其類別表是配著 Qwen3-ASR 的**簡體**輸出建立的。我們的管線
#   一直沒事，是因為既有模型都吐簡體、OpenCC 繁化發生在對齊**之後** —— 對齊器
#   永遠只看到簡體。
#
#   TEA-ASR 打破了這個隱含假設：它**原生輸出繁體**，`詢`／`會`／`灣` 這類字落在
#   對齊器的類別表外，對齊器就輸出未映射的類別 ID，crispasr 直接把它渲染成私用
#   區字元（實測 `U+E000 + class_id`）→ 字幕出現亂碼，時間軸也跟著對歪。
#   故這類模型一律不掛 `-falign`，改用後端自帶的時間戳（較粗但正確）。
_FA_INCOMPATIBLE = ("tea-asr", "tea_asr")


def _fa_incompatible(model_path) -> bool:
    name = Path(model_path).name.lower()
    return any(k in name for k in _FA_INCOMPATIBLE)


class CrispWhisperEngine:
    """CrispASR(whisper backend) 推理引擎。子程序模式呼叫 crispasr.exe。"""

    # use_aligner 做成 property：對齊器與模型不相容時，**任何**呼叫端把它設成
    # True 都會被擋下（webview_backend.transcribe 每次轉錄都會依 UI 勾選重設，
    # 只在 load() 關掉是攔不住的）。
    @property
    def use_aligner(self) -> bool:
        return self._use_aligner

    @use_aligner.setter
    def use_aligner(self, v: bool):
        self._use_aligner = bool(v) and not getattr(self, "_fa_blocked", False)

    def __init__(self):
        self._use_aligner = False
        self._fa_blocked  = False
        self.ready       = False
        self._lock       = threading.Lock()
        # 介面相容旗標
        self.processor   = None    # None → app.py 改用 SUPPORTED_LANGUAGES
        self.diar_engine = None
        # 時間軸對齊（FA）：用 qwen3 ForcedAligner GGUF + crispasr `-am <gguf> -falign`，
        # 以 CTC 對齊器的字級時間軸覆蓋 whisper 自帶（較粗）的時間戳。gguf 存在才啟用；
        # 否則退回 whisper native 時間軸（仍可用，僅較不精確）。
        # 旗標真偽性供 app.py 共用 FA 流程判斷（use_aligner=勾選、_fa_bin=模型就緒）。
        self.use_aligner   = False
        self.aligner       = False
        self._fa           = False
        self._fa_bin       = None   # FA gguf 路徑（真值＝就緒）
        self._aligner_path = None   # Path: qwen3-forced-aligner-*.gguf
        # 執行狀態
        self._exe        = None    # Path: crispasr.exe
        self._model_path = None    # Path: Breeze .bin / Qwen3-ASR .gguf
        self._backend    = "whisper"  # crispasr 後端：whisper / qwen3 / qwen3-1.7b …
        self._device_id  = 0       # GPU device id
        self._gpu_backend = "vulkan"  # --gpu-backend；None＝純 CPU（掛 -ng）
        self.cc          = None    # OpenCC 轉換器

    # ══ 載入 ══════════════════════════════════════════════════════════

    def load(self, model_path: Path = None, crispasr_dir: Path = None,
             device_id: int = 0, cb=None, aligner_path: Path = None,
             backend: str | None = None, gpu_backend: str | None = "vulkan"):
        """驗證 crispasr.exe 與模型存在即可（子程序模式無常駐載入）。

        aligner_path：qwen3 ForcedAligner GGUF；None 時自動在 crispasr_dir 內尋找。
        找到 → 啟用 FA（process_file 加 `-am <gguf> -falign`）；找不到 → 退回
        whisper native 時間軸。

        backend：crispasr 後端名（whisper / qwen3 / qwen3-1.7b …）。None 時依
        模型檔名自動推斷（Breeze→whisper、qwen3*.gguf→qwen3-1.7b）。

        gpu_backend：crispasr.exe 的 `--gpu-backend`（vulkan / cuda）。傳 None
        代表這份核心是 CPU 版或使用者選了純 CPU 推理 → 改掛 `-ng`。呼叫端應
        依「實際安裝的加速版本」給值（downloader.crispasr_gpu_backend）。
        """
        def _s(msg):
            if cb:
                cb(msg)

        crispasr_dir = Path(crispasr_dir) if crispasr_dir else _DEFAULT_CRISPASR_DIR
        _s("尋找 CrispASR 執行檔…")
        exe = _find_exe(crispasr_dir)
        if exe is None:
            raise FileNotFoundError(
                f"找不到 crispasr.exe（已搜尋 {crispasr_dir}）。\n"
                "請將 CrispASR Vulkan 版解壓到 crispasr/ 目錄。"
            )
        self._exe = exe

        model_path = Path(model_path) if model_path else (_DEFAULT_CRISPASR_DIR / _DEFAULT_MODEL_NAME)
        if not model_path.exists():
            raise FileNotFoundError(
                f"找不到 Whisper 模型：{model_path}\n"
                f"請放入 {_DEFAULT_MODEL_NAME}（Breeze-ASR-26 GGML）。"
            )
        self._model_path = model_path
        self._device_id  = int(device_id)
        self._gpu_backend = gpu_backend or None
        # 後端：顯式指定優先，否則依模型檔名自動推斷
        self._backend    = backend or _infer_backend(model_path)

        # ── 偵測 FA 對齊器 GGUF（有則預設啟用）──────────────────────────
        #   先判斷這顆模型能不能用對齊器（見 _FA_INCOMPATIBLE 的說明）；不能用
        #   就整條 FA 關掉，避免對齊器吐出私用區亂碼並把時間軸對歪。
        self._fa_blocked = _fa_incompatible(model_path)
        if self._fa_blocked:
            _s("此模型原生輸出繁體，與 CTC 對齊器不相容 → 停用字級對齊，"
               "改用後端自帶時間軸")
        ap = Path(aligner_path) if aligner_path else _find_aligner_gguf(crispasr_dir)
        if ap and ap.exists() and not self._fa_blocked:
            self._aligner_path = ap
            self._fa_bin   = ap          # 真值＝FA 就緒
            self.use_aligner = True      # 預設開（缺檔則維持 False，退回 native）
            self.aligner   = True
            self._fa       = True
            _s(f"FA 對齊器就緒（{ap.name}）")
        else:
            self._aligner_path = None
            self._fa_bin   = None
            self.use_aligner = False
            self.aligner   = False
            self._fa       = False

        import opencc
        self.cc = opencc.OpenCC(_opencc_config())
        self.ready = True
        _fa_tag = "  + FA" if self._aligner_path else ""
        _s(f"就緒（CrispASR / Vulkan / {self._backend}  {model_path.name}{_fa_tag}）")

    def rebuild_cc(self):
        """依目前詞彙轉換旗標重建 OpenCC（免重新載入）。"""
        try:
            import opencc
            self.cc = opencc.OpenCC(_opencc_config())
        except Exception:
            pass

    # ══ 子程序命令建構（★ 由你定奪：speed/quality 取捨）═════════════════

    def _build_cmd(self, audio_path: Path, out_base: Path,
                   language: str | None, word_level: bool = True) -> list[str]:
        """組合 crispasr.exe 命令列引數。

        ───────────────────────────────────────────────────────────────
        ★ 學習貢獻點：這裡是整個引擎唯一有「設計取捨」的地方。
          我們在 2026-06-17 的實測（Breeze-ASR-26 q5_0 / Vulkan / 126s 台語
          新聞）數據如下，請據此決定預設參數：

            設定                 fallback  時間    RTF     品質
            預設(-bo5+fallback)   129      142s   0.84x   人名最準
            -bo 2(保留fallback)    50       47.5s  2.65x   品質平平
            -nf -bo 1(關fallback)   0        3.6s  ~35x    幾乎不損(推薦)

          可用的相關旗標：
            -nf            關閉溫度 fallback（避免難段反覆重解碼，最大提速）
            -bo N          best-of 候選數（1=貪婪最快；預設 5）
            -bs N          beam size（預設 greedy）
            -fa            flash attention（Vulkan 上安全，建議開）

        固定部分（已幫你寫好）：模型、語言、Vulkan 裝置、輸出格式、靜音。
        word_level=True 時加 `-ml 1` → 字元級時間軸（餵共享分行器，與 Qwen
        路徑統一）；word_level=False（即時模式）→ 純文字。
        你只需把速度/品質相關旗標填入 `tuning`（約 1–4 個元素）。
        ───────────────────────────────────────────────────────────────
        """
        cmd = [
            str(self._exe),
            "-m", str(self._model_path),
        ]
        # 非 whisper 架構（Qwen3-ASR 等）需顯式指定後端，crispasr 才用對解碼路徑
        if self._backend and self._backend != "whisper":
            cmd += ["--backend", self._backend]
        # 加速版本：Vulkan／CUDA 走 --gpu-backend + 裝置 id；CPU 版掛 -ng
        # （CPU 版核心沒有 ggml-vulkan/cuda DLL，硬給 --gpu-backend 會啟動失敗）
        if self._gpu_backend:
            cmd += ["--gpu-backend", self._gpu_backend,
                    "-dev", str(self._device_id)]
        else:
            cmd += ["-ng"]
        cmd += [
            "-of", str(out_base),         # 輸出檔名（不含副檔名）
            "-np",                        # 不印多餘訊息
        ]
        if word_level:
            # -ml 1 → 每段一字（字元級時間軸），等價於 FA 的字級輸出
            # -pp → 印出「progress = NN% (x/N slices)」，供串流解析驅動進度條
            cmd += ["-osrt", "-ml", "1", "-pp"]
        else:
            cmd += ["-otxt"]

        lang_code = _LANG_MAP.get(language, None) if language else None
        if lang_code:
            cmd += ["-l", lang_code]

        # ── 時間軸對齊（FA）：用 qwen3 ForcedAligner GGUF 覆蓋 whisper native 時間戳 ──
        # 僅檔案模式（word_level）需要精確字級時間軸；即時模式只取文字故不掛。
        #   -am <gguf>  載入 CTC 對齊器模型
        #   -falign     強制改用對齊器的字級時間軸（即使 whisper backend 自帶）
        # 注意：此處 -fa 是 flash-attn、-falign 才是 force-aligner，兩者並存不衝突。
        if word_level and self.use_aligner and self._aligner_path:
            cmd += ["-am", str(self._aligner_path), "-falign"]

        # 速度/品質取捨（實測決策，2026-06-17）：
        #   -fa  flash attention（Vulkan 安全）；-nf 關閉溫度 fallback
        #   → 126s 台語新聞 142s→3.6s（0.84x→~35x），品質幾乎不損。
        #   保留 fallback 雖人名略準但慢 40 倍、無法批次，故預設關閉。
        tuning: list[str] = ["-fa", "-nf"]

        cmd += tuning
        cmd.append(str(audio_path))
        return cmd

    # ══ 串流執行 crispasr.exe，解析 -pp 進度 → 驅動進度條 ═══════════════════
    def _run_streaming(self, cmd: list[str], progress_cb=None) -> int:
        """以 Popen 串流讀 crispasr 輸出，解析「progress = NN%」即時回報進度。

        crispasr 一次處理整檔（內部切 N 個 slice），`-pp` 會輸出
        「crispasr: progress = NN% (x/N slices)」。據此把單一阻塞子程序變成
        會動的進度條（granularity = slice 數，長檔較細、短檔較粗）。
        stderr 併入 stdout 一起讀；無 `-pp` 或解析不到時，行為等同舊版（只是
        進度條不動）——故對 Breeze/即時模式皆安全。
        """
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            bufsize=1, creationflags=_CREATE_NO_WINDOW, startupinfo=_STARTUP_INFO,
        )
        pat = re.compile(r"progress\s*=\s*(\d+)\s*%")
        try:
            for line in proc.stdout:
                m = pat.search(line)
                if m and progress_cb:
                    p = int(m.group(1))
                    # 上限 99：留「完成」給寫檔後的最終回報，避免 100% 卻還在寫字幕
                    progress_cb(min(99, p), 100, f"CrispASR 轉錄中… {p}%")
        finally:
            proc.wait()
        return proc.returncode

    # ══ 檔案轉錄 → SRT（字級時間軸 → 共享分行器，與 Qwen 路徑統一）════════

    def process_file(self, audio_path: Path, progress_cb=None,
                     language: str | None = None, context: str | None = None,
                     diarize: bool = False, n_speakers: int | None = None,
                     original_path: Path | None = None,
                     out_format: str | None = None) -> Path | None:
        """音檔 → SRT，回傳 SRT 路徑（None=無輸出）。

        流程：crispasr `-ml 1` 取字元級時間軸 → 解析 → 與 OpenVINO/chatllm
        共用的 `_ts_chatllm_to_subtitle_lines` 分行（含 OpenCC 繁化、孤兒合併）。

        說話者分離（diarize）：用與後端無關的外部 ONNX（diar_engine，CPU）取得
        說話者段落，再依時間把每行字幕指派給對應說話者。whisper/qwen 後端皆適用
        —— crispasr 單次出字級時間軸 × diar 段落，零額外子程序。需先由上層
        （webview_backend._ensure_diarization / app.py）把 diar_engine 掛上。
        context（hint）目前仍未支援，靜默忽略。
        """
        audio_path = Path(audio_path)
        if progress_cb:
            progress_cb(0, 1, "CrispASR 轉錄中…")

        with tempfile.TemporaryDirectory() as td:
            out_base = Path(td) / "crisp_out"
            cmd = self._build_cmd(audio_path, out_base, language, word_level=True)
            with self._lock:
                self._run_streaming(cmd, progress_cb)
            srt_tmp = out_base.with_suffix(".srt")
            if not srt_tmp.exists():
                return None
            raw = srt_tmp.read_text(encoding="utf-8", errors="replace")

        # ── 依後端選對「斷句契約」（使用者要求：Qwen 要走 Qwen 邏輯，非 Whisper）──
        #   Qwen3-ASR：模型自帶中文標點 → 標點剔出 items、只留 raw_text 當切點，
        #              break_on_space=False → 與 chatllm/OpenVINO **完全相同**的標點切行。
        #   Whisper(Breeze)：無標點 → 保留空白語句邊界、break_on_space=True 逼近 Qwen。
        is_qwen = (self._backend or "").startswith("qwen3")
        ts_items, raw_text = _parse_srt_words(raw, drop_punct=is_qwen)
        if not ts_items:
            return None
        if getattr(self, "_fa_blocked", False):
            # 沒有對齊器時 `-ml 1` 對 qwen3 後端無效，每個 item 會是一整段長文字，
            # 共用分行器便無從斷句（實測整段可達 60 秒）。這裡把長 item 依字元
            # **等時內插**攤成字元級，讓分行器照常做標點切行／MAX_CHARS 保護。
            # 內插時間軸當然不如 CTC 對齊精準，但在「粗到不能看」與「約略正確」
            # 之間，後者對字幕明顯有用；且段落起訖仍是後端給的真實時間。
            ts_items = _interpolate_chars(ts_items)

        # 日語：跳過 OpenCC 繁化。s2twp/s2t 是「簡體中文→繁體中文」轉換，套在日文上
        # 會把日文漢字（会/静/図/学）誤轉成繁中字形（會/靜/圖/學），對日語讀者是錯的
        # （ja-anime 模型本就針對日語，尤其該保留原生字形）。中文/台語維持繁化不變；
        # 自動偵測無法預知語言，保守仍走繁化。
        cc_use = None if _LANG_MAP.get(language) == "ja" else self.cc

        self._last_segments_rich = None   # 本次字級結果（卡拉OK側通道）
        lines5 = _ts_chatllm_to_subtitle_lines(
            ts_items, raw_text, 0.0, None, cc_use, _output_simplified,
            break_on_space=(not is_qwen), with_words=True,
        )
        if not lines5:
            return None
        if getattr(self, "_fa_blocked", False):
            lines5 = _fix_leading_punct(lines5)
        lines = [(s, e, t, sp) for (s, e, t, sp, _w) in lines5]

        # 說話者分離（外部 ONNX，與 whisper/qwen 後端無關）：依時間中點指派每行說話者。
        # diarize 只重新指派 speaker、不增減行，故與 lines5 平行 → 可 zip 併回字級。
        if diarize and self.diar_engine is not None and getattr(self.diar_engine, "ready", False):
            lines = self._apply_diarization(audio_path, lines, n_speakers, progress_cb)

        # 字級結果掛上實例：speaker 取 diarize 後的 lines，words 取 lines5
        self._last_segments_rich = [
            {"start": a[0], "end": a[1], "text": a[2], "speaker": a[3], "words": b[4]}
            for a, b in zip(lines, lines5)
        ]

        # 共用寫出層：依全域設定（或 out_format 覆寫）產出 .srt 或 .txt。
        ref = original_path if original_path is not None else audio_path
        out = write_transcript(ref, lines, out_format)
        if progress_cb:
            progress_cb(1, 1, "完成")
        return out

    def _apply_diarization(self, audio_path, lines, n_speakers, progress_cb=None):
        """把每行字幕依時間中點指派給 diar_engine 偵測到的說話者。

        lines:  [(start, end, text, None), ...]（_ts_chatllm_to_subtitle_lines 產出）
        回傳：  [(start, end, text, "說話者N"), ...]；失敗/無段落時原樣回傳。
        """
        try:
            from audio_io import load_audio_16k_mono
            if progress_cb:
                progress_cb(1, 1, "說話者分離中…")
            audio, _ = load_audio_16k_mono(Path(audio_path), SAMPLE_RATE)
            segs = self.diar_engine.diarize(audio, n_speakers=n_speakers)
        except Exception:
            return lines
        if not segs:
            return lines

        def _spk_at(t: float) -> str:
            for (t0, t1, lab) in segs:           # 中點落在哪個說話者段落
                if t0 <= t < t1:
                    return lab
            return min(segs, key=lambda g: abs((g[0] + g[1]) / 2.0 - t))[2]  # 否則取最近段

        return [(s, e, txt, _spk_at((s + e) / 2.0)) for (s, e, txt, _spk) in lines]

    # ══ 即時/單段轉錄 → 純文字 ═════════════════════════════════════════

    def transcribe(self, audio: np.ndarray, max_tokens: int = 300,
                   language: str | None = None, context: str | None = None) -> str:
        """16kHz float32 音訊 → 文字（即時模式用，不需字級時間軸）。"""
        import soundfile as sf
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "seg.wav"
            sf.write(str(wav), audio, SAMPLE_RATE)
            out_base = Path(td) / "seg_out"
            cmd = self._build_cmd(wav, out_base, language, word_level=False)
            with self._lock:
                subprocess.run(
                    cmd, capture_output=True, stdin=subprocess.DEVNULL,
                    creationflags=_CREATE_NO_WINDOW, startupinfo=_STARTUP_INFO,
                )
            txt = out_base.with_suffix(".txt")
            text = txt.read_text(encoding="utf-8", errors="replace").strip() if txt.exists() else ""
        # 日語跳過繁化（同 process_file 理由）；中文/台語維持 s2twp/s2t。
        if _output_simplified or _LANG_MAP.get(language) == "ja" or self.cc is None:
            return text
        return self.cc.convert(text)


# ── SRT 解析 / 寫入（模組層工具）──────────────────────────────────────

def _srt_ts_to_sec(ts: str) -> float:
    """'00:00:01,560' → 1.56 秒。"""
    ts = ts.strip().replace(".", ",")
    hms, _, ms = ts.partition(",")
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s) + (int(ms) / 1000.0 if ms else 0.0)


def _fix_leading_punct(lines5: list[tuple]) -> list[tuple]:
    """把「以標點開頭」的字幕行，其開頭標點移回前一行行尾。

    只發生在無對齊器的內插路徑：crispasr 的每個 SRT 區塊各自結束，區塊邊界剛好
    落在標點前時，該標點就成了下一行的開頭（「，直詢臺上」）。標點屬於前一句，
    移回去即可。首行的開頭標點沒有前一行可歸，直接去掉。
    """
    out: list[tuple] = []
    for (s, e, t, spk, w) in lines5:
        lead = ""
        while t and t[0] in _PUNCT:
            lead, t = lead + t[0], t[1:]
        if lead and out:
            ps, pe, pt, pspk, pw = out[-1]
            out[-1] = (ps, pe, pt + lead, pspk, pw)
        if t:
            out.append((s, e, t, spk, w))
    return out


def _interpolate_chars(items: list[tuple[str, float, float]]
                       ) -> list[tuple[str, float, float]]:
    """把「一段長文字 + 起訖時間」攤成字元級（等時內插）。

    只處理長度 > 1 的 item；已是單字元的原樣保留，故對本來就有字級時間軸的
    路徑（掛了 FA）呼叫也不會有副作用。
    """
    out: list[tuple[str, float, float]] = []
    for text, s, e in items:
        n = len(text)
        if n <= 1:
            out.append((text, s, e))
            continue
        step = (e - s) / n if e > s else 0.0
        for i, ch in enumerate(text):
            out.append((ch, s + step * i, s + step * (i + 1)))
    return out


def _parse_srt_words(srt_text: str, drop_punct: bool = False
                     ) -> tuple[list[tuple[str, float, float]], str]:
    """解析 `-ml 1` 字元級 SRT。

    回傳 (ts_items, raw_text)：
        ts_items : [(char, start_s, end_s), ...]（不含空白項）
        raw_text : 字元串接，並在 whisper 的「空白語句邊界」處保留一個空白，
                   供分行器 break_on_space 在語句邊界切行。

    drop_punct=True（Qwen 模式）：標點字元只保留在 raw_text 當「切點」，**不**進
    ts_items —— 這正是 chatllm/OpenVINO Qwen 路徑的契約（items=內容詞、標點僅在
    raw_text）。crispasr `-ml 1` 會把標點也輸出成獨立字元段，若放進 items 會讓
    斷句器的 raw_text 指標雙重前進、標點淪為下一行開頭，故需在此剔除。
    """
    items: list[tuple[str, float, float]] = []
    raw_parts: list[str] = []
    # 正規化換行（crispasr SRT 為 CRLF），再以空行切塊
    blocks = srt_text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n\n")
    for block in blocks:
        rows = block.split("\n")
        tl_idx = next((i for i, r in enumerate(rows) if "-->" in r), None)
        if tl_idx is None:
            continue
        a, _, b = rows[tl_idx].partition("-->")
        try:
            s = _srt_ts_to_sec(a)
            e = _srt_ts_to_sec(b)
        except (ValueError, IndexError):
            continue
        raw = "".join(rows[tl_idx + 1:])    # 不 strip，保留前導空白
        if raw.strip() == "":
            raw_parts.append(" ")           # 純空白塊 = 語句邊界
            continue
        if raw[:1].isspace():               # 字元前帶空白 = 邊界 + 字元
            raw_parts.append(" ")
        token = raw.strip()
        raw_parts.append(token)             # 標點一律進 raw_text（供切點偵測）
        if drop_punct and token and all(ch in _PUNCT for ch in token):
            continue                        # Qwen：標點不進 items
        items.append((token, s, e))
    return items, "".join(raw_parts)


def _write_srt_lines(out: Path, lines: list[tuple[float, float, str, str | None]]):
    """[(start,end,text,spk), ...] → SRT 檔（與 Qwen 路徑相同格式）。"""
    with open(out, "w", encoding="utf-8") as f:
        for idx, (s, e, text, spk) in enumerate(lines, 1):
            prefix = f"{spk}：" if spk else ""
            f.write(f"{idx}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{prefix}{text}\n\n")
