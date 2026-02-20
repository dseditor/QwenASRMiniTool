"""
processor_numpy.py
─────────────────────────────────────────────────────────────────────
純 numpy 實作 Qwen3-ASR Processor，完整取代 torch / transformers / qwen_asr。

功能：
  • Mel 特徵提取  ─ 與 WhisperFeatureExtractor 完全對齊
  • BPE 解碼      ─ byte-level GPT-2 風格，從 vocab.json 讀取
  • Prompt 組裝   ─ 從 prompt_template.json 讀取預計算 IDs

依賴：
  numpy（已有）、pathlib（標準庫）
  不需要 torch、transformers、qwen_asr

使用：
  from processor_numpy import LightProcessor
  proc = LightProcessor(ov_dir)
  mel, ids = proc.prepare(audio_float32_16khz)
  text     = proc.decode(generated_token_ids)
"""
from __future__ import annotations

import json
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════
# Mel 特徵提取（對齊 WhisperFeatureExtractor）
# ══════════════════════════════════════════════════════════════════════

# 參數來源：preprocessor_config.json
_N_FFT         = 400
_HOP           = 160
_N_MELS        = 128
_N_SAMPLES     = 480_000              # 30s × 16000
_NB_FRAMES     = 3000                 # nb_max_frames
_PAD_LEN       = (_NB_FRAMES - 1) * _HOP + _N_FFT   # = 480240（使 center=False 剛好 3000 frames）
_SR            = 16_000
_FMIN          = 0.0
_FMAX          = 8_000.0


_MEL_FILTERS: np.ndarray | None = None
_MEL_FILTERS_PATH: Path | None  = None


def _load_mel_filters(model_dir: Path | None = None) -> np.ndarray:
    """
    載入從 WhisperFeatureExtractor 匯出的 mel filterbank。
    形狀為 (n_freqs, n_mels) = (201, 128)，由 generate_prompt_template.py 產生。

    不重新計算，避免 mel_scale/norm 參數不一致導致的精確度損失。
    """
    global _MEL_FILTERS, _MEL_FILTERS_PATH
    if _MEL_FILTERS is not None:
        return _MEL_FILTERS

    # 搜尋 mel_filters.npy
    candidates: list[Path] = []
    if model_dir is not None:
        candidates.append(model_dir.parent / "mel_filters.npy")  # ov_models/mel_filters.npy
        candidates.append(model_dir / "mel_filters.npy")
    candidates.append(Path(__file__).parent / "ov_models" / "mel_filters.npy")

    for p in candidates:
        if p.exists():
            raw = np.load(str(p))       # (201, 128) or (128, 201)
            # 確保形狀是 (n_mels, n_freqs) = (128, 201)
            if raw.shape == (_N_MELS, _N_FFT // 2 + 1):
                _MEL_FILTERS = raw.astype(np.float32)
            elif raw.shape == (_N_FFT // 2 + 1, _N_MELS):
                _MEL_FILTERS = raw.T.astype(np.float32)
            else:
                raise ValueError(f"mel_filters.npy shape {raw.shape} 不符預期")
            _MEL_FILTERS_PATH = p
            return _MEL_FILTERS

    raise FileNotFoundError(
        "找不到 ov_models/mel_filters.npy。\n"
        "請先執行：python generate_prompt_template.py"
    )


def _mel_filters(model_dir: Path | None = None) -> np.ndarray:
    return _load_mel_filters(model_dir)


# 週期性漢寧窗（與 transformers window_function(periodic=True) 一致）
_HANN_WINDOW: np.ndarray = np.hanning(_N_FFT + 1)[:-1].astype(np.float32)


def extract_mel(audio: np.ndarray) -> np.ndarray:
    """
    輸入：float32 音頻，16kHz，任意長度
    輸出：[1, 128, 3000] float32 mel 矩陣

    與 transformers WhisperFeatureExtractor 行為完全對齊：
      • 先截斷/補零至 n_samples = 480000（30 秒）
      • center=True：兩端各加 n_fft//2 = 200 個反射樣本
      • 滑窗 STFT（週期漢寧窗）→ 取前 3000 frames
    """
    # 1. 截斷至 n_samples；若更短則補零（center padding 前需補足）
    audio = audio.astype(np.float32)
    if len(audio) > _N_SAMPLES:
        audio = audio[:_N_SAMPLES]
    if len(audio) < _N_SAMPLES:
        audio = np.pad(audio, (0, _N_SAMPLES - len(audio)))  # 補零至 480000

    # 2. center=True：兩端各加 n_fft//2 個反射樣本 → 480400 個樣本
    half = _N_FFT // 2  # 200
    audio_c = np.pad(audio, half, mode="reflect")             # (480400,)

    # 3. sliding_window_view → 3001 frames（取前 3000）
    frames = np.lib.stride_tricks.sliding_window_view(audio_c, _N_FFT)[::_HOP]
    frames = frames[:_NB_FRAMES].astype(np.float32)           # (3000, 400)
    windowed = frames * _HANN_WINDOW                           # (3000, 400)

    # 4. FFT → power spectrum
    stft  = np.fft.rfft(windowed, axis=1)                     # (3000, 201)
    power = np.abs(stft).astype(np.float32) ** 2              # (3000, 201)

    # 5. Mel filterbank
    mel = (_load_mel_filters() @ power.T)                     # (128, 3000)

    # 6. Log scale + Whisper 正規化
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    return log_mel[np.newaxis, :, :].astype(np.float32)       # (1, 128, 3000)


# ══════════════════════════════════════════════════════════════════════
# BPE 解碼（byte-level GPT-2 風格）
# ══════════════════════════════════════════════════════════════════════

def _build_byte_decoder() -> dict[str, int]:
    """
    GPT-2 byte-to-unicode mapping 的反向版本（unicode char → byte value）。
    vocab.json 中的 token 字串使用此編碼。
    """
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    # 結果：unicode char → byte value
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER: dict[str, int] = _build_byte_decoder()


def _bpe_decode(token_strings: list[str]) -> str:
    """
    將 BPE token 字串列表解碼回 UTF-8 文字。
    先拼接 byte-level unicode 字串，再逐字元轉回 bytes，最後 UTF-8 decode。
    """
    merged = "".join(token_strings)
    byte_vals = []
    for ch in merged:
        bval = _BYTE_DECODER.get(ch)
        if bval is not None:
            byte_vals.append(bval)
        # 未知字元跳過（不應出現）
    try:
        return bytes(byte_vals).decode("utf-8", errors="replace")
    except Exception:
        return merged


# ══════════════════════════════════════════════════════════════════════
# LightProcessor：組合上面兩個元件
# ══════════════════════════════════════════════════════════════════════

class LightProcessor:
    """
    對應 ASREngine 中原本的 processor + pad_id。

    屬性（供 app.py 使用）：
        pad_id  : int   ← <|audio_pad|> 的 token id
        eos_id  : int   ← <|im_end|>
        eot_id  : int   ← <|endoftext|>
    """

    def __init__(self, model_dir: Path):
        """
        model_dir : OV_DIR，含 vocab.json、prompt_template.json
        prompt_template.json 從 generate_prompt_template.py 產生。
        """
        # ── 預先載入 mel filters（避免 extract_mel 每次重找路徑）─────
        _load_mel_filters(model_dir)

        # ── 讀取 prompt template ──────────────────────────────────────
        # 搜尋順序：BASE_DIR → OV_DIR → 本檔案同層（PyInstaller _internal/）
        tpl_path = model_dir.parent.parent / "prompt_template.json"
        if not tpl_path.exists():
            tpl_path = model_dir / "prompt_template.json"
        if not tpl_path.exists():
            tpl_path = Path(__file__).parent / "prompt_template.json"
        with open(tpl_path, "r", encoding="utf-8") as f:
            tpl = json.load(f)

        self._prefix_ids: list[int]  = tpl["prefix_ids"]
        self._suffix_ids: list[int]  = tpl["suffix_ids"]
        self._n_audio:    int        = tpl["n_audio_tokens"]
        self.pad_id:      int        = tpl["audio_pad_id"]
        self.eos_id:      int        = tpl["eos_id"]
        self.eot_id:      int        = tpl["eot_id"]
        self._special_ids: set[int]  = set(tpl["special_ids"])

        # ── 建立 id → token string 的對映（BPE decode 用）────────────
        vocab_path = model_dir / "vocab.json"
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab: dict[str, int] = json.load(f)   # str → id
        self._id2str: dict[int, str] = {v: k for k, v in vocab.items()}

        # 補上 added_tokens（tokenizer_config.json 裡的特殊 token）
        tc_path = model_dir / "tokenizer_config.json"
        with open(tc_path, "r", encoding="utf-8") as f:
            tc = json.load(f)
        for tok_id_str, info in tc.get("added_tokens_decoder", {}).items():
            self._id2str[int(tok_id_str)] = info["content"]

        # 預先計算固定的 input_ids（prompt 不含音頻 pad 部分）
        self._audio_pad_block = np.array(
            [self.pad_id] * self._n_audio, dtype=np.int64
        )

    # ── 外部 API ──────────────────────────────────────────────────────

    def prepare(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        輸入：16kHz float32 音頻
        輸出：(mel, input_ids)
            mel       : [1, 128, 3000] float32
            input_ids : [1, L]         int64
        """
        mel = extract_mel(audio)

        ids = np.array(
            self._prefix_ids + [self.pad_id] * self._n_audio + self._suffix_ids,
            dtype=np.int64,
        )[np.newaxis, :]     # (1, 9+390+6=405)

        return mel, ids

    def decode(self, token_ids: list[int], skip_special: bool = True) -> str:
        """
        將生成的 token id 列表解碼為 UTF-8 字串。
        skip_special=True 時跳過 special tokens（含 <asr_text>）。
        """
        parts: list[str] = []
        for tid in token_ids:
            if skip_special and tid in self._special_ids:
                continue
            s = self._id2str.get(tid, "")
            if s:
                parts.append(s)
        return _bpe_decode(parts)
