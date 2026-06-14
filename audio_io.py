"""audio_io.py — 不依賴 librosa / numba 的音訊載入

背景
----
`librosa` 會在 import 階段載入 `numba`。numba 0.60 不相容 numpy ≥ 2.1
（拋「numba needs numpy 2.0 or less」），導致 `import librosa` 直接失敗，
整條轉錄管線（檔案上傳、錄音上傳、批次）全中。降 numpy 會牽動 openvino，
因此改以零 numba 的組合取代 librosa.load：

  • soundfile（libsndfile）— 讀 wav / flac / ogg / mp3，回原生取樣率
  • soxr（C 綁定）        — 高品質重採樣；缺則退 scipy.resample_poly；
                            再缺則退 numpy 線性插值
  • ffmpeg 後援           — soundfile 讀不了的格式（m4a / aac / wma…）
                            先用 ffmpeg 轉 16k 單聲道 wav 再讀

對外：
  load_audio_16k_mono(path) -> (np.float32 mono @16k, 16000)
  audio_duration(path)      -> 秒數（float）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SR = 16000
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """重採樣（soxr → scipy → numpy 線性插值），皆不依賴 numba。"""
    if sr_in == sr_out:
        return data.astype(np.float32, copy=False)
    try:
        import soxr
        return soxr.resample(data, sr_in, sr_out).astype(np.float32)
    except Exception:
        pass
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(data, sr_out // g, sr_in // g).astype(np.float32)
    except Exception:
        n = max(1, int(round(len(data) * sr_out / sr_in)))
        xp = np.linspace(0, len(data) - 1, n)
        return np.interp(xp, np.arange(len(data)), data).astype(np.float32)


def _to_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.asarray(data, dtype=np.float32)


def _ffmpeg_to_wav(path: Path) -> Path | None:
    """soundfile 讀不了時，用 ffmpeg 轉 16k 單聲道 wav，回暫存路徑。"""
    try:
        import subprocess
        import tempfile
        from ffmpeg_utils import find_ffmpeg
        ff = find_ffmpeg()
        if not ff:
            return None
        out = Path(tempfile.mkdtemp(prefix="aio_")) / "audio.wav"
        subprocess.run(
            [str(ff), "-y", "-i", str(path), "-vn", "-ac", "1",
             "-ar", str(SR), "-f", "wav", str(out)],
            check=True, capture_output=True,
            creationflags=_CREATE_NO_WINDOW,
        )
        return out if out.exists() else None
    except Exception:
        return None


def load_audio_16k_mono(path, target_sr: int = SR):
    """載入音訊為 (np.float32 mono @ target_sr, target_sr)。不使用 librosa/numba。

    soundfile 直接可讀 → 讀後（必要時）重採樣；讀不了的格式退回 ffmpeg
    轉 16k wav 再讀。
    """
    import soundfile as sf
    p = Path(path)
    try:
        data, sr = sf.read(str(p), dtype="float32", always_2d=False)
        data = _to_mono(data)
    except Exception:
        wav = _ffmpeg_to_wav(p)
        if wav is None:
            raise
        data, sr = sf.read(str(wav), dtype="float32", always_2d=False)
        data = _to_mono(data)
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
    return data, target_sr


def audio_duration(path) -> float:
    """音訊長度（秒），不使用 librosa。soundfile 標頭優先，失敗才實際解碼。"""
    import soundfile as sf
    try:
        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass
    try:
        data, sr = load_audio_16k_mono(path)
        return len(data) / float(sr)
    except Exception:
        return 0.0
