"""
一次性工具：生成 prompt_template.json
需要在 venv（含 torch + transformers + qwen_asr）中執行一次。
輸出的 JSON 讓 processor_numpy.py 不需任何 torch/transformers 即可運作。

用法：
    python generate_prompt_template.py
"""
import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent
OV_DIR   = BASE_DIR / "ov_models" / "qwen3_asr_int8"
OUT_PATH = BASE_DIR / "prompt_template.json"

print("載入 qwen_asr processor…")
import qwen_asr  # noqa
from qwen_asr.inference.qwen3_asr import (
    Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr",   Qwen3ASRConfig,                   exist_ok=True)
AutoModel.register( Qwen3ASRConfig, Qwen3ASRForConditionalGeneration, exist_ok=True)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor,            exist_ok=True)

processor = AutoProcessor.from_pretrained(str(OV_DIR), fix_mistral_regex=True)

# ── 建立 prompt（與 app.py transcribe() 完全一致）──────────────────
msgs = [
    {"role": "system", "content": ""},
    {"role": "user",   "content": [{"type": "audio", "audio": ""}]},
]
prompt_text = processor.apply_chat_template(
    msgs, add_generation_prompt=True, tokenize=False
)
print(f"Prompt text: {repr(prompt_text)}")

# ── 用靜音音頻跑一次，取得完整 input_ids ──────────────────────────
dummy_audio = np.zeros(480000, dtype=np.float32)
inp = processor(text=[prompt_text], audio=[dummy_audio], return_tensors="np", padding=True)
ids = inp["input_ids"][0].tolist()

AUDIO_PAD_ID = processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
print(f"audio_pad_id  = {AUDIO_PAD_ID}")
print(f"Total tokens  = {len(ids)}")

# 找出音頻 pad 區段的位置
pad_positions = [i for i, x in enumerate(ids) if x == AUDIO_PAD_ID]
assert pad_positions, "找不到 <|audio_pad|>，請確認模型正確"
print(f"Audio pad 位置：{pad_positions[0]}..{pad_positions[-1]}，共 {len(pad_positions)} 個")

prefix_ids = ids[: pad_positions[0]]
suffix_ids = ids[pad_positions[-1] + 1 :]
print(f"Prefix IDs ({len(prefix_ids)}): {prefix_ids}")
print(f"Suffix IDs ({len(suffix_ids)}): {suffix_ids}")

# ── 確認解碼 id 設定 ──────────────────────────────────────────────
eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
eot_id = processor.tokenizer.eos_token_id or 151643

# 所有 special token id（decode 時跳過）
special_ids = set()
for tok_id_str, info in processor.tokenizer.added_tokens_decoder.items():
    if info.special:
        special_ids.add(int(tok_id_str))

print(f"EOS id        = {eos_id}")
print(f"EOT id        = {eot_id}")
print(f"Special token count: {len(special_ids)}")

# ── 儲存 ──────────────────────────────────────────────────────────
template = {
    "prefix_ids":     prefix_ids,
    "suffix_ids":     suffix_ids,
    "n_audio_tokens": len(pad_positions),
    "audio_pad_id":   AUDIO_PAD_ID,
    "eos_id":         eos_id,
    "eot_id":         eot_id,
    "special_ids":    sorted(special_ids),
    "prompt_text":    prompt_text,
}
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(template, f, indent=2, ensure_ascii=False)

# ── 儲存 mel filters（直接從 WhisperFeatureExtractor 取出）─────────
import numpy as np
fe = processor.feature_extractor
mel_filters = fe.mel_filters   # shape: (n_freqs, n_mels) = (201, 128)
mel_filters_path = BASE_DIR / "ov_models" / "mel_filters.npy"
np.save(str(mel_filters_path), mel_filters)
print(f"mel_filters shape: {mel_filters.shape} → {mel_filters_path}")

print(f"\n✅  已儲存至 {OUT_PATH}")
print(f"✅  mel_filters 儲存至 {mel_filters_path}")
print(f"    prefix={len(prefix_ids)} tokens, audio_pad={len(pad_positions)}, suffix={len(suffix_ids)} tokens")
