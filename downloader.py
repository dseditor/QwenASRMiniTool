"""
模型完整性檢查與自動下載工具

使用純標準庫（urllib）直接下載，支援斷點續傳。
不依賴 huggingface_hub / torch / transformers。

用法（命令列）：
    python downloader.py            ← 檢查後自動下載缺少的模型
    python downloader.py --check    ← 只檢查，不下載
"""
from __future__ import annotations

import hashlib
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _ssl_ctx() -> ssl.SSLContext:
    """建立 SSL Context，優先序：certifi bundle → 系統預設 → 不驗證（fallback）。

    PyInstaller EXE 中 Python 的 CA bundle 路徑常失效，
    certifi 套件自帶 Mozilla cacert.pem，是最可靠的修法。
    若兩者都不可用，才退回「不驗證」模式（只用於可信任的 HuggingFace URL）。
    """
    # 優先：certifi 套件的 CA bundle
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        pass
    # 次選：系統預設（開發環境通常正常）
    try:
        return ssl.create_default_context()
    except Exception:
        pass
    # 最後降級：不驗證（frozen EXE CA bundle 完全缺失時）
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode    = ssl.CERT_NONE
    return ctx

# ── 路徑（PyInstaller 凍結時指向 EXE 旁邊）────────────────────────────
import sys as _sys
if getattr(_sys, "frozen", False):
    BASE_DIR = Path(_sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent

_DEFAULT_MODEL_DIR = BASE_DIR / "ov_models"

# ── HuggingFace 倉庫 ───────────────────────────────────────────────────
# 主要來源（dseditor 備份倉庫）；失敗時自動切換至備用來源
_HF_REPO_PRIMARY  = "dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO"
_HF_REPO_FALLBACK = "Echo9Zulu/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO"
_HF_BASE_PRIMARY  = f"https://huggingface.co/{_HF_REPO_PRIMARY}/resolve/main"
_HF_BASE_FALLBACK = f"https://huggingface.co/{_HF_REPO_FALLBACK}/resolve/main"
_HF_REPO  = _HF_REPO_PRIMARY   # 相容舊版引用
_HF_BASE  = _HF_BASE_PRIMARY   # 相容舊版引用
_VAD_URL  = "https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx"
_UA       = "Mozilla/5.0 (compatible; QwenASR-downloader)"

# ── HuggingFace 鏡像站 ─────────────────────────────────────────────────
# 中國大陸等地直連 huggingface.co 常逾時；可改走鏡像（如 hf-mirror.com）。
# set_mirror() 設定後，_download_file() 會把所有 huggingface.co 的網址
# 改寫到鏡像網域；空字串代表使用官方來源（預設）。
HF_OFFICIAL = "https://huggingface.co"
_MIRROR_BASE: str = ""   # 例："https://hf-mirror.com"


def set_mirror(base: str | None):
    """設定 HuggingFace 鏡像站台基底網址（如 'https://hf-mirror.com'）。

    傳入空字串或 None 表示恢復使用官方 huggingface.co。
    僅改寫 huggingface.co 來源；GitHub 等其他網域不受影響。
    """
    global _MIRROR_BASE
    base = (base or "").strip().rstrip("/")
    _MIRROR_BASE = base


def get_mirror() -> str:
    """回傳目前生效的鏡像基底網址（空字串＝官方）。"""
    return _MIRROR_BASE


def _apply_mirror(url: str) -> str:
    """若已設定鏡像且 url 指向 huggingface.co，改寫為鏡像網域。"""
    if _MIRROR_BASE and url.startswith(HF_OFFICIAL):
        return _MIRROR_BASE + url[len(HF_OFFICIAL):]
    return url

# ── 1.7B INT8 KV-cache 模型倉庫 ───────────────────────────────────────
_HF_1P7B_REPO = "dseditor/Qwen3-ASR-1.7B-INT8_OpenVINO"
_HF_1P7B_BASE = f"https://huggingface.co/{_HF_1P7B_REPO}/resolve/main"

_1P7B_REQUIRED_BIN: list[str] = [
    "audio_encoder_model.bin",
    "thinker_embeddings_model.bin",
    "decoder_prefill_kv_model.bin",
    "decoder_kv_model.bin",
]
_1P7B_REQUIRED_OTHER: list[str] = [
    "audio_encoder_model.xml",
    "thinker_embeddings_model.xml",
    "decoder_prefill_kv_model.xml",
    "decoder_kv_model.xml",
    "prompt_template.json",
    "config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "chat_template.json",
]

# ── 說話者分離模型（直接 URL，非 HF API）──────────────────────────────
_DIAR_BASE = "https://huggingface.co/altunenes/speaker-diarization-community-1-onnx/resolve/main"
DIAR_FILES: dict[str, str] = {
    "segmentation-community-1.onnx": f"{_DIAR_BASE}/segmentation-community-1.onnx",
    "embedding_model.onnx":          f"{_DIAR_BASE}/embedding_model.onnx",
}

# ── 必要檔案清單 ───────────────────────────────────────────────────────
# 大型 .bin 附 SHA256；小型設定檔只檢查存在即可。
REQUIRED_BIN: dict[str, str] = {
    "audio_encoder_model.bin":      "d892464d9b6986719dd6e5c3962b880a2708d874c2c9bdead8958581be2dacb9",
    "decoder_model.bin":            "cc4363c401f5faf41e2bfcb4aea80c72144b8ea66d13ca5ca62cf49421a25778",
    "thinker_embeddings_model.bin": "a7818fcbd77240fb8705bc47c2a15da98498056cdd419742b7685719b5dc2a44",
}
REQUIRED_OTHER: list[str] = [
    "audio_encoder_model.xml",
    "thinker_embeddings_model.xml",
    "decoder_model.xml",
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
]


def _get_paths(model_dir: Path) -> tuple[Path, Path]:
    """回傳 (ov_dir, vad_path)。"""
    return model_dir / "qwen3_asr_int8", model_dir / "silero_vad_v4.onnx"


# ── Git LFS 指標檔偵測 ────────────────────────────────────────────────
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

def _file_is_real(path: Path) -> bool:
    """回傳 True 表示檔案存在且不是 Git LFS pointer。

    當使用者以「git clone」取得 HuggingFace 模型倉庫但未安裝
    git-lfs 時，所有 LFS 追蹤的檔案（*.bin、*.onnx、*.npy 等）
    在磁碟上會是約 130 bytes 的 pointer 文字檔：
        version https://git-lfs.github.com/spec/v1
        oid sha256:<hash>
        size <bytes>
    Path.exists() 對 pointer 回傳 True，導致下載器誤以為
    檔案已完整下載而跳過，最終模型無法載入。
    本函式以讀取前 43 bytes 來識別並拒絕 LFS pointer。
    """
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            header = f.read(len(_LFS_MAGIC))
        return header != _LFS_MAGIC
    except OSError:
        return False


# ── ForcedAligner（chatllm .bin，單檔，無需 torch）────────────────────
_FA_BIN_NAME = "qwen3-focedaligner-0.6b.bin"
_FA_BIN_URL  = (
    "https://huggingface.co/dseditor/Collection/resolve/main/"
    "qwen3-focedaligner-0.6b.bin"
)


def quick_check_aligner(model_dir: Path) -> bool:
    """快速檢查 chatllm ForcedAligner .bin 是否存在（非 LFS pointer）。"""
    return _file_is_real(Path(model_dir) / _FA_BIN_NAME)


def download_aligner(model_dir: Path, progress_cb=None):
    """下載 chatllm ForcedAligner .bin 至 model_dir（約 939 MB）。

    progress_cb(pct: float, msg: str)   pct ∈ [0, 1]
    下載失敗時拋出例外。
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    dest = model_dir / _FA_BIN_NAME
    if _file_is_real(dest):
        if progress_cb:
            progress_cb(1.0, "時間軸對齊模型已存在")
        return

    if progress_cb:
        progress_cb(0.0, "下載時間軸對齊模型…")

    def _cb(done: int, total: int):
        if progress_cb and total > 0:
            progress_cb(
                done / total,
                f"下載時間軸對齊模型… {done/1_048_576:.0f} / {total/1_048_576:.0f} MB",
            )

    _download_file(_FA_BIN_URL, dest, progress_cb=_cb)
    if progress_cb:
        progress_cb(1.0, "時間軸對齊模型下載完成！")


def quick_check_diarization(model_dir: Path) -> bool:
    """快速檢查說話者分離模型是否存在且非 LFS pointer。"""
    diar_dir = model_dir / "diarization"
    return all(_file_is_real(diar_dir / fname) for fname in DIAR_FILES)


def download_diarization(diar_dir: Path, progress_cb=None):
    """
    下載說話者分離 ONNX 模型至 diar_dir。
    progress_cb(pct: float, msg: str)   pct ∈ [0, 1]
    下載失敗時拋出例外。
    """
    diar_dir.mkdir(parents=True, exist_ok=True)
    total_tasks = len(DIAR_FILES)

    for idx, (fname, url) in enumerate(DIAR_FILES.items()):
        dest = diar_dir / fname
        if _file_is_real(dest):
            if progress_cb:
                progress_cb((idx + 1) / total_tasks, f"✅ {fname}（已存在）")
            continue

        base_pct = idx / total_tasks
        span_pct = 1.0 / total_tasks
        if progress_cb:
            progress_cb(base_pct, f"下載 {fname}…")

        def _file_cb(done: int, total: int,
                     _b=base_pct, _s=span_pct, _f=fname):
            if progress_cb and total > 0:
                progress_cb(
                    _b + _s * done / total,
                    f"下載 {_f}…  {done/1_048_576:.1f} / {total/1_048_576:.1f} MB",
                )

        _download_file(url, dest, progress_cb=_file_cb)
        if progress_cb:
            progress_cb(base_pct + span_pct, f"✅ {fname}")

    if progress_cb:
        progress_cb(1.0, "說話者分離模型下載完成！")


def quick_check_1p7b(model_dir: Path) -> bool:
    """快速檢查 1.7B KV-cache INT8 模型是否完整（非 LFS pointer）。"""
    kv_dir = model_dir / "qwen3_asr_1p7b_kv_int8"
    for fname in _1P7B_REQUIRED_BIN + _1P7B_REQUIRED_OTHER:
        if not _file_is_real(kv_dir / fname):
            return False
    return True


def download_1p7b(model_dir: Path, progress_cb=None):
    """
    從 HuggingFace 下載 1.7B KV-cache INT8 模型至 model_dir/qwen3_asr_1p7b_kv_int8/。
    progress_cb(pct: float, msg: str)   pct ∈ [0, 1]
    下載失敗時拋出例外。
    """
    kv_dir = model_dir / "qwen3_asr_1p7b_kv_int8"
    kv_dir.mkdir(parents=True, exist_ok=True)

    all_files = _1P7B_REQUIRED_BIN + _1P7B_REQUIRED_OTHER
    tasks = [f for f in all_files if not _file_is_real(kv_dir / f)]

    if not tasks:
        if progress_cb:
            progress_cb(1.0, "所有 1.7B 檔案已存在")
        return

    total = len(tasks)
    for idx, fname in enumerate(tasks):
        dest     = kv_dir / fname
        base_pct = idx / total
        span_pct = 1.0 / total

        if progress_cb:
            progress_cb(base_pct, f"下載 {fname}…")

        def _file_cb(done: int, total_b: int,
                     _b=base_pct, _s=span_pct, _f=fname):
            if progress_cb and total_b > 0:
                progress_cb(
                    _b + _s * done / total_b,
                    f"下載 {_f}…  {done/1_048_576:.1f} / {total_b/1_048_576:.1f} MB",
                )

        url = f"{_HF_1P7B_BASE}/{fname}"
        _download_file(url, dest, progress_cb=_file_cb)

        if progress_cb:
            progress_cb(base_pct + span_pct, f"✅ {fname}")

    if progress_cb:
        progress_cb(1.0, "1.7B 模型下載完成！")


# ══════════════════════════════════════════════════════════════════════
# CrispASR(Whisper) 核心 + Breeze-ASR-26 GGML 模型（按需下載，不進 EXE）
# ══════════════════════════════════════════════════════════════════════

# 已測試完成的 CrispASR Windows Vulkan 版（v0.7.2）
_CRISPASR_ZIP_URL = ("https://github.com/CrispStrobe/CrispASR/releases/"
                     "download/v0.7.2/crispasr-windows-x86_64-vulkan.zip")

# Breeze-ASR-26 GGML（phate334）三種量化
_BREEZE_REPO  = "phate334/Breeze-ASR-26-GGML"
_BREEZE_BASE  = f"https://huggingface.co/{_BREEZE_REPO}/resolve/main"
_BREEZE_FILES = {
    "q4": "ggml-model-q4_0.bin",   # 輕量 ~889 MB
    "q5": "ggml-model-q5_0.bin",   # 標準 ~1.08 GB
    "q8": "ggml-model-q8_0.bin",   # 精確 ~1.66 GB
}


def breeze_filename(quant: str) -> str:
    """量化代碼（q4/q5/q8）→ Breeze GGML 檔名（未知時回傳 q5 標準）。"""
    return _BREEZE_FILES.get(quant, _BREEZE_FILES["q5"])


def quick_check_crispasr(crispasr_dir: Path) -> bool:
    """CrispASR 核心是否就緒（crispasr.exe 存在，含一層子資料夾）。"""
    if (crispasr_dir / "crispasr.exe").exists():
        return True
    return any(crispasr_dir.glob("**/crispasr.exe"))


def download_crispasr_core(crispasr_dir: Path, progress_cb=None):
    """下載 CrispASR Vulkan zip 並解壓（扁平化）至 crispasr_dir。

    progress_cb(pct: float, msg: str)。GitHub 來源不套用 HF 鏡像。
    """
    import shutil
    import tempfile
    import zipfile

    crispasr_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0.0, "下載 CrispASR 核心（Vulkan）…")

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                0.9 * done / total_b,
                f"下載核心…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "crispasr-vulkan.zip"
        _download_file(_CRISPASR_ZIP_URL, zpath, progress_cb=_cb)
        if progress_cb:
            progress_cb(0.92, "解壓 CrispASR 核心…")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(td)
        # 扁平化：所有 .exe / .dll 移到 crispasr_dir 根目錄
        for f in Path(td).glob("**/*"):
            if f.is_file() and f.suffix.lower() in (".exe", ".dll"):
                shutil.copy(f, crispasr_dir / f.name)

    if not quick_check_crispasr(crispasr_dir):
        raise RuntimeError("CrispASR 核心解壓後仍找不到 crispasr.exe")
    if progress_cb:
        progress_cb(1.0, "CrispASR 核心就緒")


# ── ffmpeg（按需下載，不隨安裝包附帶）──────────────────────────────────
#   桌面 CTk 版走 BtbN essentials（ffmpeg_utils.FFmpegDownloadDialog）；webview /
#   EXE 版改抓我們自備的精簡 zip（只含 ffmpeg.exe[/ffprobe.exe]），體積小、來源穩定。
_FFMPEG_ZIP_URL = "https://huggingface.co/dseditor/Collection/resolve/main/ffmpeg.zip"


def quick_check_ffmpeg(dest_dir: Path) -> bool:
    """ffmpeg.exe 是否已就緒（dest_dir 根目錄或子資料夾）。"""
    dest_dir = Path(dest_dir)
    if (dest_dir / "ffmpeg.exe").exists():
        return True
    return any(dest_dir.glob("**/ffmpeg.exe"))


def download_ffmpeg(dest_dir: Path, progress_cb=None):
    """下載 ffmpeg.zip（HF）並解壓 ffmpeg.exe/ffprobe.exe 至 dest_dir（扁平化）。

    progress_cb(pct: float, msg: str)。仿 download_crispasr_core 的流程。
    """
    import shutil
    import tempfile
    import zipfile

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if progress_cb:
        progress_cb(0.0, "下載 ffmpeg（影片音軌提取）…")

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                0.9 * done / total_b,
                f"下載 ffmpeg…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    with tempfile.TemporaryDirectory() as td:
        zpath = Path(td) / "ffmpeg.zip"
        _download_file(_FFMPEG_ZIP_URL, zpath, progress_cb=_cb)
        if progress_cb:
            progress_cb(0.92, "解壓 ffmpeg…")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(td)
        # 扁平化：把 ffmpeg.exe / ffprobe.exe 移到 dest_dir 根目錄（忽略 zip 內層級）
        for f in Path(td).glob("**/*"):
            if f.is_file() and f.name.lower() in ("ffmpeg.exe", "ffprobe.exe"):
                shutil.copy(f, dest_dir / f.name)

    if not quick_check_ffmpeg(dest_dir):
        raise RuntimeError("ffmpeg 解壓後仍找不到 ffmpeg.exe")
    if progress_cb:
        progress_cb(1.0, "ffmpeg 就緒")


def quick_check_breeze(crispasr_dir: Path, quant: str) -> bool:
    """指定量化的 Breeze 模型是否存在（排除 LFS pointer）。"""
    return _file_is_real(crispasr_dir / breeze_filename(quant))


def download_breeze(crispasr_dir: Path, quant: str, progress_cb=None):
    """下載指定量化的 Breeze-ASR-26 GGML 模型至 crispasr_dir。

    progress_cb(pct: float, msg: str)。支援斷點續傳與 HF 鏡像。
    """
    crispasr_dir.mkdir(parents=True, exist_ok=True)
    fname = breeze_filename(quant)
    dest  = crispasr_dir / fname
    if _file_is_real(dest):
        if progress_cb:
            progress_cb(1.0, f"{fname} 已存在")
        return

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                done / total_b,
                f"下載 {fname}…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    _download_file(f"{_BREEZE_BASE}/{fname}", dest, progress_cb=_cb)
    if progress_cb:
        progress_cb(1.0, f"✅ {fname}")


# ── qwen3 ForcedAligner GGUF（CrispASR -am 對齊器，Whisper 核心專用 FA）──
# crispasr.exe -am <gguf> -falign：用 CTC 對齊器的字級時間軸覆蓋 whisper 自帶
# 的（較粗）時間戳。同作者(cstr)上傳，與 crispasr 的 -am 介面相容。
_ALIGNER_GGUF_REPO  = "cstr/qwen3-forced-aligner-0.6b-GGUF"
_ALIGNER_GGUF_BASE  = f"https://huggingface.co/{_ALIGNER_GGUF_REPO}/resolve/main"
_ALIGNER_GGUF_FILES = {
    "q4": "qwen3-forced-aligner-0.6b-q4_k.gguf",   # 輕量 ~529 MB
    "q5": "qwen3-forced-aligner-0.6b-q5_0.gguf",   # 標準 ~643 MB
    "q8": "qwen3-forced-aligner-0.6b-q8_0.gguf",   # 精確 ~986 MB
}
# 對齊（CTC）對量化不敏感，標準 q5 即足夠精確，預設用之。
_ALIGNER_GGUF_DEFAULT = "q5"


def aligner_gguf_filename(quant: str = _ALIGNER_GGUF_DEFAULT) -> str:
    """量化代碼（q4/q5/q8）→ qwen3 ForcedAligner GGUF 檔名（未知時回傳 q5）。"""
    return _ALIGNER_GGUF_FILES.get(quant, _ALIGNER_GGUF_FILES[_ALIGNER_GGUF_DEFAULT])


def quick_check_aligner_gguf(crispasr_dir: Path,
                             quant: str = _ALIGNER_GGUF_DEFAULT) -> bool:
    """指定量化的 ForcedAligner GGUF 是否存在（排除 LFS pointer）。"""
    return _file_is_real(crispasr_dir / aligner_gguf_filename(quant))


def download_aligner_gguf(crispasr_dir: Path,
                          quant: str = _ALIGNER_GGUF_DEFAULT, progress_cb=None):
    """下載 qwen3 ForcedAligner GGUF 至 crispasr_dir（與 crispasr.exe 同層）。

    progress_cb(pct: float, msg: str)。支援斷點續傳與 HF 鏡像。
    """
    crispasr_dir.mkdir(parents=True, exist_ok=True)
    fname = aligner_gguf_filename(quant)
    dest  = crispasr_dir / fname
    if _file_is_real(dest):
        if progress_cb:
            progress_cb(1.0, f"{fname} 已存在")
        return

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                done / total_b,
                f"下載 {fname}…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    _download_file(f"{_ALIGNER_GGUF_BASE}/{fname}", dest, progress_cb=_cb)
    if progress_cb:
        progress_cb(1.0, f"✅ {fname}")


# ── Qwen3-ASR-1.7B GGUF（CrispASR -casr 後端，Qwen 核心跑在 crispasr.exe）──
# crispasr.exe --backend qwen3-1.7b：把 Qwen3-ASR 1.7B 當作 CrispASR 的後端，
# 走 Vulkan（Intel/AMD/NVIDIA 通吃），繁中斷句品質佳（見 crispasr-nemotron-eval）。
# 同作者(cstr)上傳的 GGUF；crisp_engine._infer_backend 依檔名含 "1.7b" 自動推斷。
_QWEN3_ASR_GGUF_REPO = "cstr/qwen3-asr-1.7b-GGUF"
_QWEN3_ASR_GGUF_BASE = f"https://huggingface.co/{_QWEN3_ASR_GGUF_REPO}/resolve/main"
_QWEN3_ASR_GGUF_FILES = {
    "q4": "qwen3-asr-1.7b-q4_k.gguf",   # 輕量
    "q8": "qwen3-asr-1.7b-q8_0.gguf",   # 精確
}
_QWEN3_ASR_GGUF_DEFAULT = "q8"


def qwen3_asr_gguf_filename(quant: str = _QWEN3_ASR_GGUF_DEFAULT) -> str:
    """量化代碼（q4/q8）→ Qwen3-ASR-1.7B GGUF 檔名（未知時回傳 q8 精確）。"""
    return _QWEN3_ASR_GGUF_FILES.get(quant, _QWEN3_ASR_GGUF_FILES[_QWEN3_ASR_GGUF_DEFAULT])


def quick_check_qwen3_asr_gguf(model_dir: Path,
                               quant: str = _QWEN3_ASR_GGUF_DEFAULT) -> bool:
    """指定量化的 Qwen3-ASR-1.7B GGUF 是否存在（排除 LFS pointer）。"""
    return _file_is_real(Path(model_dir) / qwen3_asr_gguf_filename(quant))


def download_qwen3_asr_gguf(model_dir: Path,
                            quant: str = _QWEN3_ASR_GGUF_DEFAULT, progress_cb=None):
    """下載指定量化的 Qwen3-ASR-1.7B GGUF 至 model_dir（通常為 ov_models/）。

    progress_cb(pct: float, msg: str)。支援斷點續傳與 HF 鏡像。
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    fname = qwen3_asr_gguf_filename(quant)
    dest  = model_dir / fname
    if _file_is_real(dest):
        if progress_cb:
            progress_cb(1.0, f"{fname} 已存在")
        return

    def _cb(done: int, total_b: int):
        if progress_cb and total_b > 0:
            progress_cb(
                done / total_b,
                f"下載 {fname}…  {done/1_048_576:.0f} / {total_b/1_048_576:.0f} MB",
            )

    _download_file(f"{_QWEN3_ASR_GGUF_BASE}/{fname}", dest, progress_cb=_cb)
    if progress_cb:
        progress_cb(1.0, f"✅ {fname}")


# ══════════════════════════════════════════════════════════════════════
# 完整性檢查
# ══════════════════════════════════════════════════════════════════════

def _sha256(path: Path, progress_cb=None) -> str:
    h = hashlib.sha256()
    total = path.stat().st_size
    done  = 0
    with open(path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
            done += len(buf)
            if progress_cb:
                progress_cb(done, total)
    return h.hexdigest()


def quick_check(model_dir: Path) -> bool:
    """快速存在性檢查（排除 Git LFS pointer，不計算雜湊）。"""
    ov_dir, vad_path = _get_paths(model_dir)
    if not _file_is_real(vad_path):
        return False
    for fname in list(REQUIRED_BIN) + REQUIRED_OTHER:
        if not _file_is_real(ov_dir / fname):
            return False
    return True


def full_verify(model_dir: Path, progress_cb=None) -> tuple[bool, str]:
    """存在 + SHA256 完整驗證。"""
    ov_dir, vad_path = _get_paths(model_dir)
    if not _file_is_real(vad_path):
        return False, f"遺失：{vad_path.name}"
    for fname in list(REQUIRED_BIN) + REQUIRED_OTHER:
        if not _file_is_real(ov_dir / fname):
            return False, f"遺失：{fname}"

    total_files = len(REQUIRED_BIN)
    for i, (fname, expected) in enumerate(REQUIRED_BIN.items()):
        fpath = ov_dir / fname
        if progress_cb:
            progress_cb(i / total_files * 0.9, f"驗證 {fname}…")

        def _inner(done, total, _i=i, _f=fname):
            if progress_cb:
                progress_cb((_i + done / total) / total_files * 0.9, f"驗證 {_f}…")

        actual = _sha256(fpath, _inner)
        if actual != expected:
            return False, f"{fname} 雜湊不符（檔案可能損壞）"

    if progress_cb:
        progress_cb(1.0, "✅ 所有模型完整")
    return True, "OK"


# ══════════════════════════════════════════════════════════════════════
# 直接 HTTP 下載（斷點續傳）
# ══════════════════════════════════════════════════════════════════════

def _download_file(url: str, dest: Path, progress_cb=None):
    """
    下載單一檔案至 dest，支援斷點續傳（Resume）。
    progress_cb(done_bytes: int, total_bytes: int)
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0

    url = _apply_mirror(url)   # 套用鏡像站改寫（若有設定）
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")

    try:
        resp = urllib.request.urlopen(req, timeout=30, context=_ssl_ctx())
    except urllib.error.HTTPError as e:
        if e.code == 416:
            # 416 Range Not Satisfiable = 檔案已完整，直接視為成功
            return
        raise

    content_length = int(resp.headers.get("Content-Length", 0))
    total = existing + content_length if content_length else 0

    # 追加寫入（resume）或全新寫入
    mode = "ab" if existing > 0 and resp.status == 206 else "wb"
    if mode == "wb":
        existing = 0

    done = existing
    with open(dest, mode) as f:
        while True:
            chunk = resp.read(1 << 16)   # 64 KB
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress_cb and total:
                progress_cb(done, total)

    resp.close()


def _download_file_with_fallback(
    fname: str,
    dest: Path,
    progress_cb=None,
):
    """
    先嘗試主要 HF 來源，若連線失敗則自動切換至備用來源。
    VAD 等非 HF 檔案（url 已為完整 URL）直接下載，不套用備援。
    """
    primary_url  = f"{_HF_BASE_PRIMARY}/{fname}"
    fallback_url = f"{_HF_BASE_FALLBACK}/{fname}"

    try:
        _download_file(primary_url, dest, progress_cb)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as primary_err:
        # 主要來源失敗，切換備用
        print(f"\n⚠ 主要來源失敗（{primary_err}），切換至備用來源…")
        _download_file(fallback_url, dest, progress_cb)


# ══════════════════════════════════════════════════════════════════════
# 批次下載所有模型
# ══════════════════════════════════════════════════════════════════════

def download_all(model_dir: Path, progress_cb=None):
    """
    下載所有缺少的模型至 model_dir。
    progress_cb(pct: float, msg: str)   pct ∈ [0, 1]
    下載失敗時拋出例外。
    """
    ov_dir, vad_path = _get_paths(model_dir)
    ov_dir.mkdir(parents=True, exist_ok=True)

    # 建立下載任務清單 (dest, hf_fname_or_direct_url, is_direct_url)
    # _file_is_real() 同時排除「不存在」與「Git LFS pointer」兩種情況
    tasks: list[tuple[Path, str, bool]] = []
    for fname in list(REQUIRED_BIN.keys()) + REQUIRED_OTHER:
        dest = ov_dir / fname
        # 小型設定檔若已存在則跳過；大型 .bin 若存在也先跳過（full_verify 再補）
        # 使用 _file_is_real() 避免將 Git LFS 指標檔誤判為有效檔案
        if not _file_is_real(dest):
            tasks.append((dest, fname, False))   # HF 相對路徑，使用備援機制
    if not _file_is_real(vad_path):
        tasks.append((vad_path, _VAD_URL, True))  # 直接 URL，不需備援

    if not tasks:
        if progress_cb:
            progress_cb(1.0, "所有檔案已存在")
        return

    total_tasks = len(tasks)
    for idx, (dest, fname_or_url, is_direct) in enumerate(tasks):
        fname = dest.name
        base_pct = idx / total_tasks
        span_pct = 1.0 / total_tasks

        if progress_cb:
            progress_cb(base_pct, f"下載 {fname}…")

        def _file_cb(done: int, total: int,
                     _b=base_pct, _s=span_pct, _f=fname):
            if progress_cb and total > 0:
                progress_cb(
                    _b + _s * done / total,
                    f"下載 {_f}…  {done/1_048_576:.1f} / {total/1_048_576:.1f} MB",
                )

        if is_direct:
            _download_file(fname_or_url, dest, progress_cb=_file_cb)
        else:
            _download_file_with_fallback(fname_or_url, dest, progress_cb=_file_cb)

        if progress_cb:
            progress_cb(base_pct + span_pct, f"✅ {fname}")

    if progress_cb:
        progress_cb(1.0, "下載完成！")


# ══════════════════════════════════════════════════════════════════════
# 命令列介面
# ══════════════════════════════════════════════════════════════════════

def _cli_bar(pct: float, msg: str):
    filled = int(pct * 40)
    bar    = "█" * filled + "░" * (40 - filled)
    print(f"\r[{bar}] {pct*100:5.1f}%  {msg:<45}", end="", flush=True)


if __name__ == "__main__":
    check_only = "--check" in sys.argv
    model_dir  = _DEFAULT_MODEL_DIR

    print("=== Qwen3-ASR 模型完整性檢查 ===\n")
    print(f"模型路徑：{model_dir}\n")

    if quick_check(model_dir):
        print("所有檔案存在，正在驗證雜湊…")
        ok, msg = full_verify(model_dir, progress_cb=_cli_bar)
        print()
        if ok:
            print("✅ 模型完整，無需下載")
        else:
            print(f"❌ {msg}")
            if not check_only:
                print("正在重新下載損壞的檔案…")
                download_all(model_dir, _cli_bar)
                print("\n✅ 完成")
    else:
        print("模型不完整或尚未下載")
        if check_only:
            sys.exit(1)
        print(f"從 HuggingFace 下載（約 1.2 GB）：{_HF_REPO_PRIMARY}（備用：{_HF_REPO_FALLBACK}）")
        print("首次下載視網路速度可能需要 5–30 分鐘\n")
        download_all(model_dir, _cli_bar)
        print("\n✅ 下載完成")
