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
import time
from pathlib import Path

import numpy as np

# 與 Qwen 路徑統一的字幕分行（字級時間軸 → 字幕行，全引擎共用）
from subtitle_lines import (_ts_chatllm_to_subtitle_lines, _srt_ts, write_transcript,
                            _ZH_CLAUSE_END, _EN_SENT_END, is_spaced_script,
                            to_traditional)

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


# crispasr 內建的 VAD 別名（`-vm` 可直接吃這些字串，會自動去 HF 抓模型到
# ~/.cache/crispasr）。本專案預設自己下載 firered-vad.gguf 放進 crispasr/，
# 別名只作為「檔案不在但使用者硬指定」時的退路。
_VAD_ALIASES = {"auto", "silero", "firered", "marblenet", "webrtc", "whisper-vad"}


def _normalize_vad_model(v: "Path | str | None") -> "Path | str | None":
    """把 vad_model 參數正規化：不存在的檔案路徑 → None（退回固定視窗切段）。

    接受三種輸入：Path/str 檔案路徑（存在才採用）、crispasr 別名字串
    （如 "firered"）、None。回傳 None 代表「這次不掛 --vad」。
    """
    if not v:
        return None
    if isinstance(v, str) and v.lower() in _VAD_ALIASES:
        return v.lower()
    pv = Path(v)
    return pv if pv.exists() else None


def _vad_flags(threshold: float, max_speech_sec: float | None = None) -> list[str]:
    """VAD 切段參數（crispasr `--vad` 系列旗標）。

    ───────────────────────────────────────────────────────────────
    ★ 學習貢獻點：切段策略是這條路線唯一需要「調」的地方。
      crispasr 可用的相關旗標與預設：
        -vt   N  閾值（0.50）        越低越敏感，漏字少但更容易把雜訊當人聲
        -vspd N  最短語音 ms（250）  低於此長度的人聲片段直接丟掉
        -vsd  N  最短靜音 ms（100）  靜音短於此不切段 → 值大＝段落較長、較完整
        -vmsd N  最長語音 s（無限）  超過就硬切；這條直接決定「段落天花板」
        -vp   N  語音前後補 ms（30）  補太少會吃掉字頭字尾
        -vo   N  段間重疊 s（0.10）   重疊大＝邊界字較不易掉，但會重複解碼

      與本專案的關聯（見 chunk-secs-drift）：字級時間軸在「每段內」是單調
      對齊的，段越長、段尾越容易累積漂移。VAD 切段的好處正是段落邊界落在
      真正的靜音處，而不是機械式的 30s；但若一段人聲連續超過 30s（歌曲、
      連珠炮講話），whisper 視窗仍會硬切，此時 -vmsd 就是你的控制點。
    ───────────────────────────────────────────────────────────────
    """
    flags = ["-vt", f"{max(0.05, min(0.95, float(threshold))):.2f}"]
    # 最長語音長度：沿用「每段最長秒數」設定（webview 設定頁滑桿）；未指定時
    # 用 30s——whisper 的視窗就是 30s，超過一定會被再切，不如讓 VAD 切在靜音處。
    ms = float(max_speech_sec or 0)
    flags += ["-vmsd", f"{ms if ms > 0 else 30.0:.1f}"]
    return flags


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
        # ── VAD 路線（crispasr 原生）───────────────────────────────────
        # None＝不掛 --vad，crispasr 走自身固定視窗（-ck，預設 30s）切段；
        # Path/別名字串＝掛 `--vad -vm <model>`，由 crispasr 在 C++ 端做語音端點
        # 偵測後按人聲邊界切段。與 OpenVINO 路徑的 silero onnx 是兩條獨立路線
        # （那條只服務 Qwen，crispasr.exe 看不到它）。
        self.vad_model     = None  # Path: firered-vad.gguf（或 "firered" 等別名）
        self.vad_threshold = 0.5   # -vt；由設定頁的 VAD 滑桿同步
        self.vad_max_speech_sec = 0  # -vmsd；0＝用 _vad_flags 的預設（30s）
        self._crispasr_dir = None    # Path: 核心所在目錄（找 Silero VAD 用）
        # Whisper／Breeze 要不要掛 VAD。預設不掛：Silero 會把歌聲當成非人聲切掉
        # （實測兩首歌 140→7 字、39→0 字，調低閾值也救不回來），而 Breeze 不掛
        # VAD 在一般素材上本來就正常。只有使用者明確選「原生 VAD」才設 True，
        # 換取長靜音不幻覺。Qwen／TEA 的 FireRed 不受此旗標影響。
        self.vad_for_whisper = False
        # 上一次 process_file 實際採用的切段方式（"firered"/"silero"/None），
        # 供上層寫進轉錄紀錄，不必讓使用者猜這次有沒有套 VAD。
        self._last_vad_used = None
        self._last_segments_rich = None   # 上一次轉錄的字級結果（卡拉OK側通道）
        self.cc          = None    # OpenCC 轉換器

    # ══ 載入 ══════════════════════════════════════════════════════════

    def load(self, model_path: Path = None, crispasr_dir: Path = None,
             device_id: int = 0, cb=None, aligner_path: Path = None,
             backend: str | None = None, gpu_backend: str | None = "vulkan",
             vad_model: "Path | str | None" = None, vad_threshold: float = 0.5):
        """驗證 crispasr.exe 與模型存在即可（子程序模式無常駐載入）。

        aligner_path：qwen3 ForcedAligner GGUF；None 時自動在 crispasr_dir 內尋找。
        找到 → 啟用 FA（process_file 加 `-am <gguf> -falign`）；找不到 → 退回
        whisper native 時間軸。

        backend：crispasr 後端名（whisper / qwen3 / qwen3-1.7b …）。None 時依
        模型檔名自動推斷（Breeze→whisper、qwen3*.gguf→qwen3-1.7b）。

        gpu_backend：crispasr.exe 的 `--gpu-backend`（vulkan / cuda）。傳 None
        代表這份核心是 CPU 版或使用者選了純 CPU 推理 → 改掛 `-ng`。呼叫端應
        依「實際安裝的加速版本」給值（downloader.crispasr_gpu_backend）。

        vad_model：FireRed VAD GGUF 路徑（或 crispasr 別名字串如 "firered"）。
        給值 → 掛 `--vad -vm <model>`，改用語音端點偵測切段；None → 維持
        crispasr 自身的固定視窗切段。載入後仍可直接改 eng.vad_model 即時切換
        （子程序模式，下一次轉錄就生效，不需重載模型）。
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
        self._crispasr_dir = Path(crispasr_dir)

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
        # VAD 路線：檔案不存在（且非別名字串）就當沒給，靜默退回固定視窗切段
        self.vad_model     = _normalize_vad_model(vad_model)
        self.vad_threshold = float(vad_threshold or 0.5)

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
        _mode = self._vad_plan(True)[0]
        if _mode == "legacy":
            _vad_tag = "  + Silero VAD"
        elif _mode == "unified":
            _vad_tag = "  + FireRed VAD"
        elif self.vad_model and self._backend == "whisper" and self.vad_for_whisper:
            # 選了 VAD 但這個組合套不上 → 明說，不要讓使用者以為切段已換
            _vad_tag = "（VAD 未套用：缺 Silero VAD 模型）"
        else:
            _vad_tag = ""
        _s(f"就緒（CrispASR / Vulkan / {self._backend}  {model_path.name}{_fa_tag}{_vad_tag}）")

    def rebuild_cc(self):
        """依目前詞彙轉換旗標重建 OpenCC（免重新載入）。"""
        try:
            import opencc
            self.cc = opencc.OpenCC(_opencc_config())
        except Exception:
            pass

    # ══ 子程序命令建構（★ 由你定奪：speed/quality 取捨）═════════════════

    def _silero_vad_path(self) -> "Path | None":
        """核心目錄裡的 Silero VAD GGML（Whisper／Breeze 專用）；沒有就 None。"""
        d = self._crispasr_dir
        if not d:
            return None
        for g in sorted(Path(d).glob("ggml-silero*.bin")):
            if g.stat().st_size > 100_000:        # 排除 LFS pointer／下載殘檔
                return g
        return None

    def _vad_plan(self, word_level: bool) -> "tuple[str | None, Path | str | None]":
        """這一次轉錄的 VAD 方案 → (mode, model)。

            mode="unified"  顯式 `--backend X` ＋ `--vad -vm <FireRed>`
            mode="legacy"   不掛 `--backend`（whisper 舊路徑）＋ `--vad -vm <Silero>`
            mode=None       不掛 VAD，crispasr 走固定視窗切段

        2026-11-15 實測（crispasr 0.8.54，News.mp3 與「人聲＋120s 靜音＋人聲」）：

            後端            路徑     VAD      結果
            qwen3 / TEA     unified  FireRed  ✅ 有無 FA 皆正常，段落貼齊人聲邊界
            whisper(Breeze) legacy   無       ❌ 長靜音段大量幻覺（120s → 2000+ 字）
            whisper(Breeze) legacy   FireRed  ❌ 被忽略（legacy 路徑不支援）
            whisper(Breeze) legacy   Silero   ✅ 靜音乾淨，且快很多（39s → 4.5s）
                                              ❌ 但歌曲的歌聲幾乎全被切掉
            whisper(Breeze) unified  FireRed  ⚠ 無 FA 可用；掛 `-am -falign` 時間倒退

        兩件事決定了 whisper 的走法：
        1. legacy 路徑**只吃 Silero**，FireRed 會被整個忽略。
        2. legacy 路徑上 `-am -falign` **本來就沒有作用**（0.8.32 與 0.8.54 掛不掛
           輸出逐位元組相同，時間軸一直是 whisper 自帶的）。所以先前「Breeze 為了
           保住 FA 而略過 VAD」其實沒有保護到任何東西，改走 Silero 不損失精度。
           unified 路徑才會真的套 FA，但 whisper 模型在那條路徑上 slice 合併會
           倒退（上游 bug），故不採用。

        因此 whisper：有 Silero → legacy＋Silero；沒有 Silero 時，未掛 FA 才退而
        用 unified＋FireRed，其餘不掛 VAD。Qwen／TEA 一律 unified＋FireRed。
        """
        if not (self.vad_model and word_level):
            return (None, None)
        if self._backend == "whisper":
            if not self.vad_for_whisper:
                return (None, None)       # 預設：Whisper 不掛 VAD（見 __init__ 說明）
            silero = self._silero_vad_path()
            if silero is not None:
                return ("legacy", silero)
            if self.use_aligner and self._aligner_path:
                return (None, None)
        return ("unified", self.vad_model)

    def _vad_enabled(self, word_level: bool) -> bool:
        """這一次轉錄有沒有掛 VAD（細節見 `_vad_plan`）。"""
        return self._vad_plan(word_level)[0] is not None

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
        # 非 whisper 架構（Qwen3-ASR 等）需顯式指定後端，crispasr 才用對解碼路徑。
        # whisper 架構本可省略 —— 但掛 --vad 時**不能**省：省略會走 crispasr 的
        # legacy whisper 路徑，那條路徑不支援 FireRedVAD（實測 0.8.32 會印
        # 「FireRedVAD is not supported in the legacy whisper path」然後整個
        # 忽略 VAD、照樣固定視窗切段）。顯式 `--backend whisper` 才進 unified
        # dispatch，VAD 才真的生效。
        _vad_mode, _vad_model = self._vad_plan(word_level)
        _vad_on = _vad_mode is not None
        if self._backend and (self._backend != "whisper" or _vad_mode == "unified"):
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

        # ── VAD 切段路線（可選）：crispasr 原生語音端點偵測 ────────────────
        #   未啟用（vad_model=None）→ 不掛任何 --vad 旗標，行為與舊版完全相同
        #   （crispasr 以固定視窗 -ck 切段）。啟用 → 段落邊界改落在真正的靜音
        #   處，長檔的空白段不再被硬切開，段尾字級漂移也較不易累積。
        #   只在檔案模式掛：即時模式（word_level=False）送進來的每一小段，前端
        #   已用瀏覽器端 RMS 靜音偵測切過，再切一次只會削掉字頭字尾。
        #   Whisper／Breeze 掛 Silero、Qwen／TEA 掛 FireRed（見 _vad_plan 的實測矩陣）。
        if _vad_on:
            cmd += ["--vad", "-vm", str(_vad_model)]
            cmd += _vad_flags(self.vad_threshold,
                              getattr(self, "vad_max_speech_sec", 0))

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
        # crispasr 在印出第一個進度之前要先載入模型、跑完整個檔案的 VAD，這段
        # 時間沒有任何輸出。實測 2.3 小時的檔案總共 683 秒，前 382 秒進度條都
        # 停在 0%，看起來就像卡死（GitHub issue #28）。這裡每兩秒回報一次已經
        # 等了多久，讓使用者知道程式還在跑。
        started = threading.Event()

        def _heartbeat():
            t0 = time.monotonic()
            while not started.wait(2.0) and proc.poll() is None:
                if progress_cb:
                    sec = int(time.monotonic() - t0)
                    try:
                        progress_cb(0, 100, f"準備中：載入模型、分析語音段落…（已 {sec} 秒，長檔需要幾分鐘）")
                    except Exception:
                        pass

        threading.Thread(target=_heartbeat, name="crisp-heartbeat", daemon=True).start()
        try:
            for line in proc.stdout:
                m = pat.search(line)
                if m:
                    started.set()
                if m and progress_cb:
                    p = int(m.group(1))
                    # 上限 99：留「完成」給寫檔後的最終回報，避免 100% 卻還在寫字幕
                    progress_cb(min(99, p), 100, f"CrispASR 轉錄中… {p}%")
        finally:
            started.set()
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
            _vad_mode = self._vad_plan(True)[0]
            self._last_vad_used = {"legacy": "silero", "unified": "firered"}.get(_vad_mode)
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
        # 這次有沒有真的掛對齊器：模型不相容（TEA）、使用者關掉、或缺模型都算沒有。
        no_fa = is_qwen and not (self.use_aligner and self._aligner_path)
        # 英文／韓文等以空白分詞的語言：共用分行器會改走 spaced_subtitle_lines，
        # 那條路徑要的是「帶標點的整個詞」，不能再逐字元拆開。
        spaced = is_qwen and is_spaced_script(ts_items)
        if no_fa and spaced:
            ts_items = _interpolate_words(ts_items)
        elif no_fa:
            # 沒有對齊器時 `-ml 1` 對 qwen3 後端無效，每個 item 會是一整段長文字，
            # 共用分行器便無從斷句（實測整段可達 60 秒）。這裡把長 item 依字元
            # **等時內插**攤成字元級，讓分行器照常做標點切行／MAX_CHARS 保護。
            # 內插時間軸當然不如 CTC 對齊精準，但在「粗到不能看」與「約略正確」
            # 之間，後者對字幕明顯有用；且段落起訖仍是後端給的真實時間。
            ts_items = _interpolate_chars(ts_items)
        elif _vad_mode == "legacy":
            # Whisper＋Silero：VAD 把時間軸映回原音檔時，跨越靜音的那個字會被拉長
            # 成整段靜音的長度（實測單字 96 秒）。收回到貼著鄰字的正常長度。
            ts_items = _clamp_stretched(ts_items)
        if is_qwen and not no_fa and not spaced:
            # CrispASR 0.8.54 起對齊器把標點黏在前一個字後面（`中、`、`呢。`），不再
            # 是獨立的一段。中文／日文的分行契約是「項目不帶標點」，在這裡剝掉。
            ts_items = _strip_item_punct(ts_items)

        # 日語：跳過 OpenCC 繁化。s2twp/s2t 是「簡體中文→繁體中文」轉換，套在日文上
        # 會把日文漢字（会/静/図/学）誤轉成繁中字形（會/靜/圖/學），對日語讀者是錯的
        # （ja-anime 模型本就針對日語，尤其該保留原生字形）。中文/台語維持繁化不變。
        # 語言是「自動」時由共用分行器依內容判斷（subtitle_lines.looks_japanese）。
        cc_use = None if _LANG_MAP.get(language) == "ja" else self.cc

        self._last_segments_rich = None   # 本次字級結果（卡拉OK側通道）
        lines5 = _ts_chatllm_to_subtitle_lines(
            ts_items, raw_text, 0.0, None, cc_use, _output_simplified,
            break_on_space=(not is_qwen), with_words=True,
        )
        if not lines5:
            return None
        if no_fa and not spaced:
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
        if _LANG_MAP.get(language) == "ja":
            return text
        return to_traditional(self.cc, text, _output_simplified)


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


# 內插單位：一串半形英數（含撇號，如 don't）算一個，其餘每個字元各算一個。
_INTERP_UNIT = re.compile(r"[A-Za-z0-9']+|.", re.S)


def _clamp_stretched(items: list[tuple[str, float, float]],
                     max_dur: float = 2.0, keep: float = 0.3
                     ) -> list[tuple[str, float, float]]:
    """把「被拉長跨過靜音」的字級項目收回正常長度。

    一個字不可能唸超過 `max_dur` 秒；超過的一律是 VAD 時間軸映射的產物。收法是
    貼向「正常長度且緊鄰」的那一側鄰字：後一項正常且相接、前一項不是 → 保留
    終點；其餘情況保留起點。收完長度為 `keep` 秒。
    """
    def _ok(k: int) -> bool:
        return 0 <= k < len(items) and (items[k][2] - items[k][1]) <= max_dur

    out: list[tuple[str, float, float]] = []
    for k, (text, s, e) in enumerate(items):
        if e - s <= max_dur:
            out.append((text, s, e))
            continue
        prev_touch = _ok(k - 1) and abs(items[k - 1][2] - s) < 0.05
        next_touch = _ok(k + 1) and abs(items[k + 1][1] - e) < 0.05
        if next_touch and not prev_touch:
            out.append((text, e - keep, e))
        else:
            out.append((text, s, s + keep))
    return out


def _interpolate_words(items: list[tuple[str, float, float]]
                       ) -> list[tuple[str, float, float]]:
    """`_interpolate_chars` 的詞級版：只在空白處拆，標點留在詞尾。

    給英文／韓文用。無對齊器時 crispasr 多半已經一項一個詞（此時原樣通過）；
    偶爾一項是一整段文字，才依各詞字元數等比例分配時間。
    """
    out: list[tuple[str, float, float]] = []
    for text, s, e in items:
        units = text.split()
        if len(units) <= 1:
            if units:
                out.append((units[0], s, e))
            continue
        total = sum(len(u) for u in units)
        step = (e - s) / total if e > s else 0.0
        pos = 0
        for u in units:
            out.append((u, s + step * pos, s + step * (pos + len(u))))
            pos += len(u)
    return out


def _strip_item_punct(items: list[tuple[str, float, float]]
                      ) -> list[tuple[str, float, float]]:
    """剝掉字級項目頭尾的標點（`中、`→`中`）；只剩標點的項目整個丟掉。

    標點仍留在 raw_text 裡當切點，這裡只是讓項目回到「不帶標點」的契約。
    項目中間的標點（`3.5`）不動。
    """
    strip = "".join(_PUNCT)
    out: list[tuple[str, float, float]] = []
    for text, s, e in items:
        core = text.strip(strip)
        if core:
            out.append((core, s, e))
    return out


def _interpolate_chars(items: list[tuple[str, float, float]]
                       ) -> list[tuple[str, float, float]]:
    """把「一段長文字 + 起訖時間」攤成字元級（等時內插）。

    只處理長度 > 1 的 item；已是單字元的原樣保留，故對本來就有字級時間軸的
    路徑（掛了 FA）呼叫也不會有副作用。

    半形英數連續字元（`golf`、`resent`、`2026`）視為一個單位、不逐字母拆開，
    否則分行器會把它當成一串單字母詞，輸出成「g o l f」甚至從中間斷行。每個
    單位分到的時間與其字元數成正比；空白不佔單位。
    """
    out: list[tuple[str, float, float]] = []
    for text, s, e in items:
        if len(text) <= 1:
            out.append((text, s, e))
            continue
        units = [u for u in _INTERP_UNIT.findall(text) if not u.isspace()]
        total = sum(len(u) for u in units)
        if total == 0:
            continue
        step = (e - s) / total if e > s else 0.0
        pos = 0
        for u in units:
            out.append((u, s + step * pos, s + step * (pos + len(u))))
            pos += len(u)
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
