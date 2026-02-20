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
import sys
import urllib.error
import urllib.request
from pathlib import Path

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
    """快速存在性檢查（不計算雜湊）。"""
    ov_dir, vad_path = _get_paths(model_dir)
    if not vad_path.exists():
        return False
    for fname in list(REQUIRED_BIN) + REQUIRED_OTHER:
        if not (ov_dir / fname).exists():
            return False
    return True


def full_verify(model_dir: Path, progress_cb=None) -> tuple[bool, str]:
    """存在 + SHA256 完整驗證。"""
    ov_dir, vad_path = _get_paths(model_dir)
    if not vad_path.exists():
        return False, f"遺失：{vad_path.name}"
    for fname in list(REQUIRED_BIN) + REQUIRED_OTHER:
        if not (ov_dir / fname).exists():
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

    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    if existing > 0:
        req.add_header("Range", f"bytes={existing}-")

    try:
        resp = urllib.request.urlopen(req, timeout=30)
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
    tasks: list[tuple[Path, str, bool]] = []
    for fname in list(REQUIRED_BIN.keys()) + REQUIRED_OTHER:
        dest = ov_dir / fname
        # 小型設定檔若已存在則跳過；大型 .bin 若存在也先跳過（full_verify 再補）
        if not dest.exists():
            tasks.append((dest, fname, False))   # HF 相對路徑，使用備援機制
    if not vad_path.exists():
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
