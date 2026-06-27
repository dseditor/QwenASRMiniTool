"""
Qwen3 ASR 字幕生成器 - CustomTkinter 前端

功能：
  1. 音檔上傳 → SRT 字幕（支援 OpenVINO CPU / GPU）
  2. 錄製轉換：偵測音訊輸入裝置，錄音後於說話停頓時轉換成字幕
"""
from __future__ import annotations

# ── UTF-8 模式：在所有其他 import 之前設定 ────────────────────────────
# 解決 Traditional Chinese Windows（cp950）上第三方套件用系統預設編碼
# 讀取 UTF-8 檔案時出現 "utf-8 codec can't decode byte 0xa6" 的問題。
# PYTHONUTF8=1 等效於 `python -X utf8`，讓所有 open() 預設使用 UTF-8。
import os as _os, sys as _sys, io as _io
_os.environ.setdefault("PYTHONUTF8", "1")
# 同步修正 stdout/stderr（避免 print 中文在 cp950 console 出錯）
for _stream_name in ("stdout", "stderr"):
    _s = getattr(_sys, _stream_name)
    if hasattr(_s, "buffer") and _s.encoding.lower() not in ("utf-8", "utf8"):
        setattr(_sys, _stream_name,
                _io.TextIOWrapper(_s.buffer, encoding="utf-8", errors="replace"))
del _os, _sys, _io, _stream_name, _s

import json
import os
import re
import sys
import tempfile
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import customtkinter as ctk

# ── chatllm 後端（可選，import 延遲到 load 時進行）────────────────────
try:
    from chatllm_engine import (
        ChatLLMASREngine, detect_vulkan_devices, probe_vulkan_devices,
        format_vad_diag, detect_degenerate_asr,
    )
    _CHATLLM_AVAILABLE = True
except Exception:
    _CHATLLM_AVAILABLE = False
    ChatLLMASREngine   = None
    def detect_vulkan_devices(_): return []
    def probe_vulkan_devices(_):
        return {"devices": [], "all_devices": [], "raw": "",
                "error": "chatllm 未載入", "exit_code": None, "exe_found": False}
    def format_vad_diag(_stats): return "⚠ 未偵測到人聲，未產生字幕"
    def detect_degenerate_asr(_text): return None

# ── CrispASR 後端（可選，Whisper/Breeze 走 Vulkan）─────────────────────
try:
    from crisp_engine import CrispWhisperEngine
    _CRISPASR_AVAILABLE = True
except Exception:
    _CRISPASR_AVAILABLE = False
    CrispWhisperEngine  = None

# ── 字幕分行（全引擎共用，與 CrispASR/Whisper 統一）────────────────────
from subtitle_lines import (
    MAX_CHARS, _ZH_CLAUSE_END, _EN_SENT_END,
    _srt_ts, _merge_orphan_lines, _ts_chatllm_to_subtitle_lines,
    write_transcript,
)
import subtitle_lines as _subs

def _resolve_backend(core: str, model_label: str):
    """(核心, 模型標籤) → (backend, 量化/尺寸)。三層選擇的中央映射。

    回傳：
      ("crispasr", "q4"|"q5"|"q8")  — Whisper / Breeze
      ("chatllm",  None)            — Qwen 1.7B Q8 (Vulkan)
      ("openvino", "1.7B"|"0.6B")   — Qwen OpenVINO
    """
    if "Whisper" in core or "Breeze" in model_label:
        q = "q4" if "Q4" in model_label else "q8" if "Q8" in model_label else "q5"
        return "crispasr", q
    if "Q8 (Vulkan)" in model_label:
        return "chatllm", None
    if "1.7B INT8" in model_label:
        return "openvino", "1.7B"
    return "openvino", "0.6B"


def _core_for_backend(backend: str) -> str:
    """backend → 核心標籤（載入完成後同步 UI 用）。"""
    return "Whisper (Breeze)" if backend == "crispasr" else "Qwen"


def _ui_core_model(settings: dict):
    """settings → (核心標籤, 模型標籤)，供載入後同步 UI 下拉。"""
    backend = settings.get("backend", "openvino")
    if backend == "crispasr":
        q = settings.get("crisp_quant", "q5")
        label = {"q4": "Breeze Q4 (輕量)", "q5": "Breeze Q5 (標準)",
                 "q8": "Breeze Q8 (精確)"}.get(q, "Breeze Q5 (標準)")
        return "Whisper (Breeze)", label
    if backend == "chatllm":
        return "Qwen", "Qwen3-ASR-1.7B Q8 (Vulkan)"
    sz = settings.get("cpu_model_size", "0.6B")
    return "Qwen", ("Qwen3-ASR-1.7B INT8" if "1.7B" in sz else "Qwen3-ASR-0.6B")

# ── 路徑 ──────────────────────────────────────────────
# PyInstaller 凍結時，模型應放在 EXE 旁邊（非 _internal/）
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
# PyInstaller onefile 解壓的暫存資源根（_MEIPASS）：--add-data 進來的檔案在這裡，
# 不在 EXE 旁。隨包附帶的小檔（如 silero_vad_v4.onnx）要從這裡找得到，否則凍結後
# 在 BASE_DIR（EXE 旁）找不到 → VAD 載入失敗、模型永遠停在「載入 VAD…」。
_MEIPASS_DIR = Path(getattr(sys, "_MEIPASS", "")) if getattr(sys, "frozen", False) else None
_DEFAULT_MODEL_DIR = BASE_DIR / "ov_models"
SETTINGS_FILE      = BASE_DIR / "settings.json"
SRT_DIR            = BASE_DIR / "subtitles"
_CHATLLM_DIR       = BASE_DIR / "chatllm"
# .bin 優先找 ov_models/（開發期），再找 GPUModel/（打包後下載位置）
_BIN_PATH          = next(
    (p for p in [
        BASE_DIR / "ov_models"  / "qwen3-asr-1.7b.bin",
        BASE_DIR / "GPUModel"   / "qwen3-asr-1.7b.bin",
    ] if p.exists()),
    BASE_DIR / "GPUModel" / "qwen3-asr-1.7b.bin",  # 預設（未下載時）
)
SRT_DIR.mkdir(exist_ok=True)

# ── 常數 ────────# 常數
SAMPLE_RATE   = 16000
VAD_CHUNK     = 512
VAD_THRESHOLD = 0.5   # 可由設定頁調整（降低可減少掴字）
MAX_GROUP_SEC = 20
MIN_SUB_SEC   = 0.6
GAP_SEC       = 0.08

RT_SILENCE_CHUNKS    = 25   # ~0.8s 靜音後觸發轉錄
RT_MAX_BUFFER_CHUNKS = 600  # ~19s 上限強制轉錄

# ── ForcedAligner 相關常數 ─────────────────────────────────────────
GPU_MODEL_DIR      = BASE_DIR / "GPUModel"
ALIGNER_MODEL_NAME = "Qwen3-ForcedAligner-0.6B"

# 斷句標點集合 _ZH_CLAUSE_END / _EN_SENT_END 已移至 subtitle_lines（共用）


# ══════════════════════════════════════════════════════
# 共用工具函式
# ══════════════════════════════════════════════════════

def _detect_speech_groups(audio: np.ndarray, vad_sess, max_group_sec: int = MAX_GROUP_SEC,
                          stats: dict | None = None) -> list[tuple[float, float, np.ndarray]]:
    """Silero VAD 分段，回傳 [(start_s, end_s, chunk), ...]

    可傳入 stats dict，函式會填入 n_chunks / max_prob / mean_prob / threshold /
    n_segments，供上層在「未偵測到人聲」時產生明確診斷（見 format_vad_diag）。
    """
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
        # 取完整語音段並多含一點尾巴；mel 會在 processor 補零到固定長度，整秒對齊
        # 非必要。舊版 floor 會丟尾段近 1 秒；句尾輕聲字也常被 Silero 的 ge 排除。
        ch = audio[gs: min(len(audio), ge + _tail)].astype(np.float32)
        if len(ch) < SAMPLE_RATE // 2:      # 最小 0.5 秒
            continue
        result.append((gs / SAMPLE_RATE, ge / SAMPLE_RATE, ch))
    return result


def _split_to_lines(text: str) -> list[str]:
    """以標點符號切分短句，移除標點，每句獨立成行。

    斷句規則（英文/中文統一）：
    1. 所有標點（,.!?;: 及中文，。？！；：…—）→ 立即切行，標點不輸出
    2. 英文整字為最小單位，詞前補空格（詞界）
    3. MAX_CHARS 保護：超限才強制換行
    """
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    text = text.strip()
    if not text:
        return []

    # 中文、英文標點統一觸發切行（含英文逗號）
    PUNCT = frozenset('，。？！；：…—、.,!?;:')
    lines: list[str] = []
    buf   = ""

    i = 0
    while i < len(text):
        ch = text[i]

        # ── 標點符號：切行，標點不加入輸出（隱藏）────────────────────
        if ch in PUNCT:
            if buf.strip():
                lines.append(buf.strip())
            buf = ""
            i += 1
            continue

        # ── 英文單字：整字收集，詞前補空格（詞界）────────────────────
        if ch.isalpha() and ord(ch) < 128:
            j = i
            while j < len(text) and text[j].isalpha() and ord(text[j]) < 128:
                j += 1
            word = text[i:j]
            prefix = " " if buf and not buf.endswith(" ") else ""
            if len(buf) + len(prefix) + len(word) > MAX_CHARS and buf.strip():
                lines.append(buf.strip())
                buf = word
            else:
                buf += prefix + word
            i = j
            continue

        # ── 空格：保留分詞間距 ────────────────────────────────────────
        if ch == " ":
            if buf and not buf.endswith(" "):
                buf += " "
            i += 1
            if len(buf.rstrip()) >= MAX_CHARS:
                lines.append(buf.strip())
                buf = ""
            continue

        # ── 中文/日文/數字等：逐字累積 ────────────────────────────────
        buf += ch
        i += 1
        if len(buf) >= MAX_CHARS:
            lines.append(buf.strip())
            buf = ""

    if buf.strip():
        lines.append(buf.strip())
    return [l for l in lines if l.strip()]



# _srt_ts 已移至 subtitle_lines（共用）


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


def _find_vad_model(model_dir: Path | None = None) -> Path | None:
    """尋找 Silero VAD ONNX：指定 model_dir → ov_models/ → GPUModel/ → _MEIPASS（打包）。

    凍結後隨包附帶的 silero_vad_v4.onnx 位於 _MEIPASS/ov_models（--add-data），
    故必須把它納入搜尋，否則在 EXE 旁的 ov_models/ 找不到 → VAD 載入卡死。
    """
    candidates: list[Path] = []
    if model_dir is not None:
        candidates += [Path(model_dir) / "silero_vad_v4.onnx",
                       Path(model_dir) / "silero_vad.onnx"]
    candidates += [
        _DEFAULT_MODEL_DIR / "silero_vad_v4.onnx",
        GPU_MODEL_DIR / "silero_vad_v4.onnx",
        _DEFAULT_MODEL_DIR / "silero_vad.onnx",
        GPU_MODEL_DIR / "silero_vad.onnx",
    ]
    if _MEIPASS_DIR is not None:
        candidates += [_MEIPASS_DIR / "ov_models" / "silero_vad_v4.onnx",
                       _MEIPASS_DIR / "silero_vad_v4.onnx"]
    for p in candidates:
        if p.exists():
            return p
    return None


# _merge_orphan_lines 已移至 subtitle_lines（共用）


def _ts_to_subtitle_lines(
    ts_list,
    raw_text: str,
    chunk_offset: float,
    spk: str | None,
    cc,
    simplified: bool,
    aligner_processor=None,
    language: str | None = None,
) -> list[tuple[float, float, str, str | None]]:
    """ForcedAligner token（詞級別）+ ASR 原文（含標點）→ 字幕行。

    使用 FA 的 aligner_processor.tokenize_space_lang() 產出 word_list，
    保證與 ts_list 完全 1:1 對應。再將每個 word 映射回 raw_text 的
    原始位置，以標點觸發切行。
    """
    _all_punct = _ZH_CLAUSE_END | _EN_SENT_END
    MAX_WORDS    = 8
    MAX_ZH_CHARS = MAX_CHARS
    result: list[tuple[float, float, str, str | None]] = []

    if not ts_list or not raw_text.strip():
        return result

    # ── 1. 用 FA 的 tokenizer 產出 word_list（與 ts_list 1:1）────────
    lang_lower = (language or "chinese").lower()
    if aligner_processor is not None:
        if lang_lower == "japanese":
            word_list = aligner_processor.tokenize_japanese(raw_text)
        elif lang_lower == "korean":
            if aligner_processor.ko_tokenizer is None:
                try:
                    from soynlp.tokenizer import LTokenizer
                    aligner_processor.ko_tokenizer = LTokenizer(
                        scores=aligner_processor.ko_score)
                except ImportError:
                    pass
            if aligner_processor.ko_tokenizer is not None:
                word_list = aligner_processor.tokenize_korean(
                    aligner_processor.ko_tokenizer, raw_text)
            else:
                word_list = aligner_processor.tokenize_space_lang(raw_text)
        else:
            word_list = aligner_processor.tokenize_space_lang(raw_text)
    else:
        # Fallback: 模擬 tokenize_space_lang（相容舊路徑）
        word_list = []
        for seg in raw_text.split():
            cleaned = "".join(c for c in seg
                              if c.isalpha() or c.isdigit() or c == "'")
            if not cleaned:
                continue
            buf = ""
            for c in cleaned:
                if '\u4e00' <= c <= '\u9fff':
                    if buf:
                        word_list.append(buf); buf = ""
                    word_list.append(c)
                else:
                    buf += c
            if buf:
                word_list.append(buf)

    # 取 min 以防長度不一致（防禦性）
    n = min(len(word_list), len(ts_list))

    # ── 2. 為每個 word 在 raw_text 中找到對應位置 ────────────────────
    #    並記錄「在這個 word 之前有哪些標點」→ 用於切行
    seg_tokens: list = []      # 當前行的 FA token
    seg_words: list[str] = []  # 當前行的原始 word
    ri = 0                     # raw_text 掃描位置

    def _is_latin_word(w: str) -> bool:
        return any(c.isascii() and c.isalpha() for c in w)

    def _emit():
        nonlocal seg_tokens, seg_words
        if not seg_tokens:
            seg_tokens = []
            seg_words  = []
            return
        start = chunk_offset + seg_tokens[0].start_time
        end   = chunk_offset + seg_tokens[-1].end_time
        # 重建文字：有拉丁詞用空格 join，純中文直接 join
        if any(_is_latin_word(w) for w in seg_words):
            text = " ".join(seg_words)
        else:
            text = "".join(seg_words)
        if not simplified and cc is not None:
            text = cc.convert(text)
        if end > start and text.strip():
            result.append((start, end, text.strip(), spk))
        seg_tokens = []
        seg_words  = []

    def _over_limit() -> bool:
        if any(_is_latin_word(w) for w in seg_words):
            return len(seg_words) > MAX_WORDS
        return sum(len(w) for w in seg_words) > MAX_ZH_CHARS

    for wi in range(n):
        word = word_list[wi]
        tok  = ts_list[wi]     # ForcedAlignItem: .text, .start_time, .end_time

        # 在 raw_text 中前進到 word 的位置（跳過標點和空格）
        # 遇到標點 → 切行
        hit_punct = False
        while ri < len(raw_text):
            c = raw_text[ri]
            if c in _all_punct:
                hit_punct = True
                ri += 1
                continue
            if c == " ":
                ri += 1
                continue
            break  # 到達下一個有效字元

        if hit_punct:
            _emit()  # 標點前的內容先輸出

        seg_tokens.append(tok)
        seg_words.append(word)

        # 在 raw_text 中跳過 word 佔用的字元
        consumed = 0
        word_len = len(word)
        while ri < len(raw_text) and consumed < word_len:
            c = raw_text[ri]
            if c in _all_punct or c == " ":
                ri += 1
                continue
            ri += 1
            consumed += 1

        # MAX_CHARS / MAX_WORDS 保護
        if _over_limit():
            _emit()

    # ── 3. 清空剩餘 ──────────────────────────────────────────────────
    _emit()
    return _merge_orphan_lines(result)


# _ts_chatllm_to_subtitle_lines 已移至 subtitle_lines（共用，全引擎統一）


def _rebuild_text_with_spaces(raw_chars: list[str]) -> str:
    """以 raw_text 的字元序列（含空格）重建可讀字幕文字（輔助函式，保留相容）。"""
    result: list[str] = []
    for ch in raw_chars:
        if ch == " ":
            if result and result[-1] != " ":
                result.append(" ")
        else:
            result.append(ch)
    return "".join(result).strip()


# 全域：是否輸出簡體中文（True = 跳過 OpenCC 繁化）
_g_output_simplified: bool = False

# 全域：繁體輸出時是否啟用「簡繁詞彙轉換」
#   True  → OpenCC "s2twp"：除字形外，連詞彙也在地化（軟件→軟體、質量→品質）
#   False → OpenCC "s2t"  ：僅字形轉換，保留原始用詞（軟件、質量）
# 僅在繁體模式（_g_output_simplified=False）下有意義。
_g_vocab_convert: bool = True


def _opencc_config() -> str:
    """依目前詞彙轉換旗標回傳對應的 OpenCC 設定名稱。"""
    return "s2twp" if _g_vocab_convert else "s2t"

# ══════════════════════════════════════════════════════
# ASR 引擎
# ══════════════════════════════════════════════════════

class ASREngine:
    """封裝所有模型。transcribe() 加互斥鎖，多執行緒安全。"""

    max_chunk_secs: int = 30   # 每段最長音訊（秒），子類別可覆寫
    # 時間軸對齊改用 chatllm 原生 FA（無需 torch）；與 ChatLLMASREngine 共用
    # 同一檔名與下載流程，讓「時間軸對齊」UI 在 CPU/GPU 後端行為一致。
    FA_BIN_NAME = "qwen3-focedaligner-0.6b.bin"

    def __init__(self):
        self.ready       = False
        self._lock       = threading.Lock()
        self.vad_sess    = None
        self.audio_enc   = None
        self.embedder    = None
        self.dec_req     = None
        self.processor   = None   # LightProcessor（不含 torch）
        self.pad_id      = None
        self.cc          = None
        self.diar_engine = None   # DiarizationEngine（可選）
        self.aligner     = None   # 相容旗標（chatllm FA 就緒時為 True）
        self.use_aligner = False  # 是否啟用時間軸對齊
        self._fa         = None   # ChatLLMAligner（chatllm 原生 FA）
        self._fa_bin     = None   # FA .bin 路徑（就緒時非 None）
        self._model_dir  = None   # 模型資料夾（FA .bin 與下載位置）

    def load(self, device: str = "CPU", model_dir: Path = None, cb=None, cpu_threads: int = 0):
        """從背景執行緒呼叫。cb(msg) 用於更新 UI 狀態。
        cpu_threads: 0=OpenVINO 自動，>0=指定邏輯核心數（LATENCY hint）
        """
        import onnxruntime as ort
        import openvino as ov
        import opencc
        from processor_numpy import LightProcessor

        if model_dir is None:
            model_dir = _DEFAULT_MODEL_DIR
        self._model_dir = model_dir   # FA .bin 與按需下載位置
        ov_dir   = model_dir / "qwen3_asr_int8"

        # ── CPU 執行緒設定 ─────────────────────────────────────────────
        # LATENCY hint：單一請求最低延遲（不同於 THROUGHPUT 批次模式）
        # ENABLE_HYPER_THREADING YES：確保 P-core HT 與 E-core 均被使用
        cpu_cfg: dict = {}
        if device == "CPU":
            cpu_cfg["PERFORMANCE_HINT"] = "LATENCY"
            cpu_cfg["ENABLE_HYPER_THREADING"] = "YES"
            if cpu_threads > 0:
                cpu_cfg["INFERENCE_NUM_THREADS"] = str(cpu_threads)
        # VAD 路徑：含 _MEIPASS（打包）等多處搜尋，凍結後才找得到隨包附帶的 onnx。
        vad_path = _find_vad_model(model_dir)

        def _s(msg):
            if cb: cb(msg)

        if vad_path is None:
            raise FileNotFoundError(
                "找不到 silero_vad_v4.onnx（VAD 模型）。請確認已隨安裝包附帶或放入 ov_models/。")
        _s("載入 VAD 模型…")
        self.vad_sess = ort.InferenceSession(
            str(vad_path), providers=["CPUExecutionProvider"]
        )

        _s("載入說話者分離模型…")
        try:
            from diarize import DiarizationEngine
            diar_dir = model_dir / "diarization"
            eng = DiarizationEngine(diar_dir)
            self.diar_engine = eng if eng.ready else None
        except Exception:
            self.diar_engine = None

        _s(f"編譯 ASR 模型（{device}）…")
        core = ov.Core()
        self.audio_enc = core.compile_model(str(ov_dir / "audio_encoder_model.xml"),      device, cpu_cfg)
        self.embedder  = core.compile_model(str(ov_dir / "thinker_embeddings_model.xml"), device, cpu_cfg)
        dec_comp       = core.compile_model(str(ov_dir / "decoder_model.xml"),            device, cpu_cfg)
        self.dec_req   = dec_comp.create_infer_request()

        _s("載入 Processor（純 numpy）…")
        self.processor = LightProcessor(ov_dir)
        self.pad_id    = self.processor.pad_id
        self.cc        = opencc.OpenCC(_opencc_config())

        # ── ForcedAligner（chatllm 原生，CPU -ngl 0，無需 torch）──────────
        #    與 ChatLLMASREngine 共用 chatllm main.exe 子程序對齊邏輯。
        #    缺 .bin 時靜默退回比例估算；UI 會引導使用者按需下載。
        self._load_aligner(cb=_s)

        # 抑制 "Setting pad_token_id to eos_token_id" 重複警告
        try:
            import transformers.utils.logging as _tf_logging
            import logging as _logging
            _tf_logging.get_logger("transformers.generation.utils").setLevel(_logging.ERROR)
        except Exception:
            pass

        self.ready     = True
        aligner_info = "  + ForcedAligner" if self.use_aligner else ""
        _s(f"編譯完成（{device}{aligner_info}）")

    def rebuild_cc(self):
        """依目前的詞彙轉換旗標重建 OpenCC 轉換器（免重新載入模型）。"""
        try:
            import opencc
            self.cc = opencc.OpenCC(_opencc_config())
        except Exception:
            pass

    # ── 時間軸對齊（chatllm 原生 FA，CPU -ngl 0）──────────────────────────
    def _load_aligner(self, cb=None):
        """偵測 chatllm 版 ForcedAligner .bin（與 ASR .bin 同層 model_dir）。

        OpenVINO 後端沿用 chatllm main.exe 子程序做字級對齊（CPU -ngl 0），
        完全不需 PyTorch。缺 .bin 或驗證失敗時靜默退回比例估算，UI 會引導
        使用者按需下載。
        """
        from fa_aligner import ChatLLMAligner
        self.aligner     = None
        self.use_aligner = False
        self._fa_bin     = None
        # FA 一律以 CPU 子程序執行（n_gpu_layers=0），與 OpenVINO 裝置選擇無關，
        # 避免 Vulkan/Intel 裝置 id 不一致；CPU 對齊已足夠快。
        self._fa = ChatLLMAligner(fa_dir=self._model_dir, n_gpu_layers=0)
        if self._fa.load(cb=cb):
            self._fa_bin     = self._fa._fa_bin
            self.aligner     = True   # 相容旗標（沿用 use_aligner 判斷流程）
            self.use_aligner = True

    def _align_chunk(
        self,
        wav_path: str,
        ref_text: str,
        language: str = "Chinese",
    ) -> list[tuple[str, float, float]]:
        """委派 chatllm FA 對齊「參考文字 + 音訊」，回傳字級時間軸。"""
        if self._fa is None or not self._fa_bin:
            return []
        return self._fa.align_chunk(wav_path, ref_text, language)

    def transcribe(
        self,
        audio: np.ndarray,
        max_tokens: int = 300,
        language: str | None = None,
        context: str | None = None,
    ) -> str:
        """將 16kHz float32 音訊轉錄為繁體中文。
        language : 強制語系（如 "Chinese"），None 表示自動偵測
        context  : 辨識提示（歌詞/關鍵字），放入 system message
        """
        with self._lock:
            # ── 前處理（純 numpy，不需 torch）────────────────────────
            mel, ids = self.processor.prepare(audio, language=language, context=context)

            # ── 音頻編碼 + 文字 Embedding ────────────────────────────
            ae = list(self.audio_enc({"mel": mel}).values())[0]
            te = list(self.embedder({"input_ids": ids}).values())[0]

            # ── 音頻特徵填入音頻 pad 位置 ─────────────────────────────
            combined = te.copy()
            mask = ids[0] == self.pad_id
            np_ = int(mask.sum()); na = ae.shape[1]
            if np_ != na:
                mn = min(np_, na)
                combined[0, np.where(mask)[0][:mn]] = ae[0, :mn]
            else:
                combined[0, mask] = ae[0]

            # ── Decoder 自回歸生成 ────────────────────────────────────
            L   = combined.shape[1]
            pos = np.arange(L, dtype=np.int64)[np.newaxis, :]
            self.dec_req.reset_state()
            out    = self.dec_req.infer({0: combined, "position_ids": pos})
            logits = list(out.values())[0]

            eos = self.processor.eos_id
            eot = self.processor.eot_id
            gen: list[int] = []
            nxt = int(np.argmax(logits[0, -1, :])); cur = L
            while nxt not in (eos, eot) and len(gen) < max_tokens:
                gen.append(nxt)
                emb = list(self.embedder(
                    {"input_ids": np.array([[nxt]], dtype=np.int64)}
                ).values())[0]
                out    = self.dec_req.infer(
                    {0: emb, "position_ids": np.array([[cur]], dtype=np.int64)}
                )
                logits = list(out.values())[0]
                nxt = int(np.argmax(logits[0, -1, :])); cur += 1

            # ── 解碼（純 Python BPE decode）──────────────────────────
            raw = self.processor.decode(gen)
            if "<asr_text>" in raw:
                raw = raw.split("<asr_text>", 1)[1]
            text = raw.strip()
            return text if _g_output_simplified else self.cc.convert(text)

    def _enforce_chunk_limit(
        self,
        groups: list[tuple[float, float, np.ndarray, "str | None"]],
    ) -> list[tuple[float, float, np.ndarray, "str | None"]]:
        """將超過 max_chunk_secs 的音訊段落切分為等長子片段。

        不論是說話者分離路徑或 VAD 單段路徑，都可能產生比模型
        輸入長度（max_chunk_secs）更長的 chunk。若不切分，
        _extract_mel() 會靜默截斷尾段，造成掉字。
        """
        max_samples = self.max_chunk_secs * SAMPLE_RATE
        result = []
        for t0, t1, chunk, spk in groups:
            if len(chunk) <= max_samples:
                result.append((t0, t1, chunk, spk))
            else:
                pos = 0
                while pos < len(chunk):
                    piece = chunk[pos: pos + max_samples]
                    if len(piece) < SAMPLE_RATE:   # 不足 1 秒的殘餘片段跳過
                        break
                    piece_t0 = t0 + pos / SAMPLE_RATE
                    piece_t1 = min(t1, piece_t0 + len(piece) / SAMPLE_RATE)
                    result.append((piece_t0, piece_t1, piece, spk))
                    pos += max_samples
        return result

    def process_file(
        self,
        audio_path: Path,
        progress_cb=None,
        language: str | None = None,
        context: str | None = None,
        diarize: bool = False,
        n_speakers: int | None = None,
        original_path: Path | None = None,
        out_format: str | None = None,
    ) -> Path | None:
        """音檔 → 字幕檔，回傳輸出路徑（.srt 或 .txt）。
        language   : 強制語系（如 "Chinese"），None 表示自動偵測
        context    : 辨識提示（歌詞/關鍵字），放入 system message
        diarize    : True 時用說話者分離取代 VAD，SRT 加說話者前綴
        n_speakers : 指定說話者人數（None=自動偵測）
        out_format : "srt" | "txt"；None 採全域設定（端點固定傳 "srt"）
        """
        from audio_io import load_audio_16k_mono
        audio, _ = load_audio_16k_mono(audio_path, SAMPLE_RATE)
        self._deg_count = 0          # 本次退化（異常）輸出段數
        self._last_vad_diag = None   # 本次「未產生字幕」的明確原因

        # ── 分段策略：說話者分離 vs 傳統 VAD ─────────────────────────
        # groups_spk: [(g0_sec, g1_sec, audio_chunk, speaker_label | None), ...]
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

        # 強制切分超過 max_chunk_secs 的片段（兩條路徑都需要）
        groups_spk = self._enforce_chunk_limit(groups_spk)

        # ── ASR 逐段轉錄 ─────────────────────────────────────────────
        all_subs: list[tuple[float, float, str, str | None]] = []
        total = len(groups_spk)
        for i, (g0, g1, chunk, spk) in enumerate(groups_spk):
            if progress_cb:
                spk_info = f" [{spk}]" if spk else ""
                progress_cb(i, total,
                            f"[{i+1}/{total}] {g0:.1f}s~{g1:.1f}s{spk_info}")

            # ── ASR 轉錄（取簡體原始輸出，對齊後再繁化）─────────────────
            max_tok = 400 if language == "Japanese" else 300
            with self._lock:
                mel, ids = self.processor.prepare(
                    chunk, language=language, context=context)
                ae = list(self.audio_enc({"mel": mel}).values())[0]
                te = list(self.embedder({"input_ids": ids}).values())[0]
                combined = te.copy()
                mask = ids[0] == self.pad_id
                np_ = int(mask.sum()); na = ae.shape[1]
                if np_ != na:
                    mn = min(np_, na)
                    combined[0, np.where(mask)[0][:mn]] = ae[0, :mn]
                else:
                    combined[0, mask] = ae[0]
                L   = combined.shape[1]
                pos = np.arange(L, dtype=np.int64)[np.newaxis, :]
                self.dec_req.reset_state()
                out    = self.dec_req.infer({0: combined, "position_ids": pos})
                logits = list(out.values())[0]
                eos = self.processor.eos_id
                eot = self.processor.eot_id
                gen: list[int] = []
                nxt = int(np.argmax(logits[0, -1, :])); cur = L
                while nxt not in (eos, eot) and len(gen) < max_tok:
                    gen.append(nxt)
                    emb = list(self.embedder(
                        {"input_ids": np.array([[nxt]], dtype=np.int64)}
                    ).values())[0]
                    out    = self.dec_req.infer(
                        {0: emb, "position_ids": np.array([[cur]], dtype=np.int64)}
                    )
                    logits = list(out.values())[0]
                    nxt = int(np.argmax(logits[0, -1, :])); cur += 1
                raw_decoded = self.processor.decode(gen)
                if "<asr_text>" in raw_decoded:
                    raw_decoded = raw_decoded.split("<asr_text>", 1)[1]
                raw_text = raw_decoded.strip()

            if not raw_text:
                continue

            # 退化輸出（複誦提示範本 / 高度重複）→ 不寫入垃圾字幕，跳過並記錄
            deg = detect_degenerate_asr(raw_text)
            if deg:
                self._deg_count = getattr(self, "_deg_count", 0) + 1
                self._deg_reason = deg
                if progress_cb:
                    progress_cb(i, total, f"[{i+1}/{total}] 略過異常輸出：{deg}")
                continue

            # ── ForcedAligner 精確時間軸對齊（chatllm .bin，CPU -ngl 0）──
            #    把 chunk 寫成暫存 wav，交給 chatllm main.exe 做字級對齊，
            #    再用 _ts_chatllm_to_subtitle_lines 依字級時間軸＋標點切行。
            aligned = False
            if self.use_aligner and self._fa_bin is not None:
                import tempfile as _tempfile
                import soundfile as _sf
                with _tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tf:
                    _tmp_wav = _tf.name
                try:
                    _sf.write(_tmp_wav, chunk, SAMPLE_RATE, subtype="PCM_16")
                    align_lang = language if (language and language != "自動偵測") else "Chinese"
                    ts_items = self._align_chunk(_tmp_wav, raw_text, align_lang)
                    if ts_items:
                        subs = _ts_chatllm_to_subtitle_lines(
                            ts_items, raw_text, g0, spk,
                            self.cc, _g_output_simplified,
                        )
                        if subs:
                            all_subs.extend(subs)
                            aligned = True
                except Exception:
                    aligned = False  # 靜默 fallback 到比例估算
                finally:
                    try:
                        os.remove(_tmp_wav)
                    except OSError:
                        pass

            if not aligned:
                # ── 比例估算 Fallback ──────────────────────────────────────
                text = raw_text if _g_output_simplified else self.cc.convert(raw_text)
                lines = _split_to_lines(text)
                all_subs.extend(
                    (s, e, line, spk) for s, e, line in _assign_ts(lines, g0, g1)
                )

        if not all_subs:
            # 全部段落都退化（垃圾輸出）→ 給明確原因，而非「未偵測到人聲」
            if getattr(self, "_deg_count", 0) > 0:
                self._last_vad_diag = (
                    f"⚠ 模型輸出異常（{getattr(self, '_deg_reason', '推理退化')}），"
                    f"共 {self._deg_count} 段被略過，未產生有效字幕。"
                    f"此核心／裝置可能不相容，建議改用 CPU(OpenVINO) 核心。"
                )
            return None

        if progress_cb:
            progress_cb(total, total, "寫入字幕…")

        # 以原始檔案的目錄與檔名輸出（影片抽音軌時 audio_path 是暫存路徑）
        # 共用寫出層：依全域設定（或 out_format 覆寫）產出 .srt 或 .txt。
        ref = original_path if original_path is not None else audio_path
        return write_transcript(ref, all_subs, out_format)


# ══════════════════════════════════════════════════════
# ASR 引擎 — 1.7B INT8 KV-cache 版本
# ══════════════════════════════════════════════════════

class ASREngine1p7B(ASREngine):
    """
    Qwen3-ASR-1.7B OpenVINO KV-cache 引擎（INT8 版本）。

    模型目錄：ov_models/qwen3_asr_1p7b_kv_int8/
      audio_encoder_model.xml       — mel(128,1000)  → audio_embeds(1,130,2048)
      thinker_embeddings_model.xml  — input_ids      → token_embeds
      decoder_prefill_kv_model.xml  — prefill pass   → logit + past_keys + past_vals
      decoder_kv_model.xml          — decode step    → logit + new_keys  + new_vals
    """

    _OV_SUBDIR     = "qwen3_asr_1p7b_kv_int8"
    max_chunk_secs = 10   # audio_encoder 匯出固定 T=1000（10s）


# ══════════════════════════════════════════════════════
# 即時轉錄管理員
# ══════════════════════════════════════════════════════


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """即時重取樣（numpy 線性插值），供串流取樣率 ≠ 16kHz 時使用。"""
    if src_sr == dst_sr:
        return audio
    n_out = int(len(audio) * dst_sr / src_sr)
    indices = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


class RealtimeManager:
    """sounddevice 串流 + VAD + 緩衝轉錄。"""

    def __init__(
        self,
        asr: ASREngine,
        device_idx: int,
        on_text,
        on_status,
        language: str | None = None,
        context: str | None = None,
    ):
        self.asr       = asr
        self.dev_idx   = device_idx
        self.on_text   = on_text    # callback(text: str)
        self.on_status = on_status  # callback(msg: str)
        self.language  = language
        self.context   = context
        self._q        = queue.Queue()
        self._running  = False
        self._stream   = None

    def start(self):
        import sounddevice as sd
        self._running = True
        # 查詢裝置原生聲道數與取樣率
        dev_info      = sd.query_devices(self.dev_idx, "input")
        self._native_ch = max(1, int(dev_info["max_input_channels"]))
        native_sr       = int(dev_info["default_samplerate"])

        # 步驟 1：嘗試以 16kHz 開啟（麥克風等 MME/DirectSound 裝置通常支援）
        self._stream_sr = SAMPLE_RATE
        try:
            self._stream = sd.InputStream(
                device=self.dev_idx,
                samplerate=SAMPLE_RATE,
                channels=self._native_ch,
                blocksize=VAD_CHUNK,
                dtype="float32",
                callback=self._audio_cb,
            )
        except sd.PortAudioError:
            # 步驟 2：16kHz 不支援 → 用裝置原生取樣率開啟，回調中即時重取樣
            # 常見情境：WASAPI 裝置（48kHz only）、部分立體聲混音裝置
            try:
                self._stream_sr = native_sr
                # blocksize 等比例放大，維持 ~32ms 窗口
                scaled_block = int(VAD_CHUNK * native_sr / SAMPLE_RATE)
                self._stream = sd.InputStream(
                    device=self.dev_idx,
                    samplerate=native_sr,
                    channels=self._native_ch,
                    blocksize=scaled_block,
                    dtype="float32",
                    callback=self._audio_cb,
                )
            except sd.PortAudioError as e:
                # 步驟 3：任何取樣率都失敗（WDM-KS 立體聲混音等）→ 提供引導訊息
                raise RuntimeError(
                    f"無法開啟此音訊裝置（16kHz 與 {native_sr}Hz 均失敗）。\n"
                    f"此裝置可能為 WDM-KS 模式的立體聲混音，不支援直接錄音。\n\n"
                    f"擷取系統音訊的替代方案：\n"
                    f"  1. 安裝虛擬音訊裝置（如 VB-CABLE / CABLE Input）\n"
                    f"  2. 在 Windows 音效設定中將「立體聲混音」設為預設錄音裝置，\n"
                    f"     然後選擇 MME 版本的預設輸入裝置"
                ) from e

        threading.Thread(target=self._loop, daemon=True).start()
        self._stream.start()
        sr_note = f"（{self._stream_sr}→{SAMPLE_RATE}Hz 重取樣）" if self._stream_sr != SAMPLE_RATE else ""
        self.on_status(f"🔴 錄音中…{sr_note}")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.on_status("⏹ 已停止")

    def _audio_cb(self, indata, frames, time_info, status):
        # 多聲道混音取平均轉 mono（立體聲混音 / WASAPI loopback 2ch）
        mono = indata.mean(axis=1) if indata.shape[1] > 1 else indata[:, 0]
        # 串流取樣率 ≠ 16kHz 時，即時重取樣至 VAD/ASR 所需的 16kHz
        if self._stream_sr != SAMPLE_RATE:
            mono = _resample(mono, self._stream_sr, SAMPLE_RATE)
        self._q.put(mono.copy())

    def _loop(self):
        h   = np.zeros((2, 1, 64), dtype=np.float32)
        c   = np.zeros((2, 1, 64), dtype=np.float32)
        sr  = np.array(SAMPLE_RATE, dtype=np.int64)
        buf: list[np.ndarray] = []
        sil = 0

        while self._running:
            try:
                chunk = self._q.get(timeout=0.1)
            except queue.Empty:
                continue

            out, h, c = self.asr.vad_sess.run(
                None,
                {"input": chunk[np.newaxis, :].astype(np.float32), "h": h, "c": c, "sr": sr},
            )
            prob = float(out[0, 0])

            if prob >= VAD_THRESHOLD:
                buf.append(chunk); sil = 0
            elif buf:
                buf.append(chunk); sil += 1
                rt_max_buf = int(getattr(self.asr, "max_chunk_secs", 19) * SAMPLE_RATE / VAD_CHUNK)
                if sil >= RT_SILENCE_CHUNKS or len(buf) >= rt_max_buf:
                    audio = np.concatenate(buf)
                    # 不裁切到整秒（processor 會補零到固定長度）；舊版 floor
                    # 會丟掉尾段最多近 1 秒語音，造成句尾字消失。
                    _max_tok = 400 if self.language == "Japanese" else 300
                    try:
                        text = self.asr.transcribe(
                            audio,
                            max_tokens=_max_tok,
                            language=self.language,
                            context=self.context,
                        )
                        if text:
                            self.on_text(text)
                    except Exception as _e:
                        self.on_status(f"⚠ 轉錄錯誤：{_e}")
                    buf = []; sil = 0
                    h = np.zeros((2, 1, 64), dtype=np.float32)
                    c = np.zeros((2, 1, 64), dtype=np.float32)


# ══════════════════════════════════════════════════════
# GUI
# ══════════════════════════════════════════════════════

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

FONT_BODY  = ("Microsoft JhengHei", 13)
FONT_MONO  = ("Consolas", 12)
FONT_TITLE = ("Microsoft JhengHei", 22, "bold")


# ══════════════════════════════════════════════════════════════════════
# 字幕驗證 & 編輯視窗（共用模組 subtitle_editor.py）
# ══════════════════════════════════════════════════════════════════════
from subtitle_editor import SubtitleDetailEditor, SubtitleEditorWindow  # noqa: F401


class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Qwen3 ASR 字幕生成器")
        self.geometry("960x700")
        self.minsize(800, 580)

        self.engine       = ASREngine()
        # 本行程「已實際載入」的 backend（Vulkan 核心 chatllm/crispasr 的 GPU
        # context 不會在切換時釋放 → 兩者並存會觸發驅動 TDR 整機重啟，故跨 Vulkan
        # 核心切換一律擋下、要求重啟。None=尚未載入任何模型。
        self._loaded_backend: str | None = None
        self._rt_mgr: RealtimeManager | None = None
        self._rt_log: list[str]              = []
        self._rt_autosave_path: Path | None  = None   # 即時追加保存目標 .txt
        self._audio_file: Path | None        = None
        self._srt_output: Path | None        = None
        self._converting                     = False
        self._dev_idx_map: dict[str, int]    = {}
        self._model_dir: Path | None         = None   # 使用者選定的模型路徑
        self._lang_list: list[str]           = []     # 載入後填入
        self._selected_language: str | None  = None   # 目前選定的語系
        self._settings: dict                 = {}     # 目前生效的設定
        self._all_devices: dict              = {}     # 偵測到的所有裝置
        self._file_hint: str | None          = None   # 音檔轉字幕 hint
        self._file_diarize: bool             = False  # 說話者分離開關
        self._file_n_speakers: int | None    = None   # 指定說話者人數（None=自動）
        self._diar_downloading: bool         = False  # 說話者分離模型下載中旗標
        self._api_server                     = None   # TranscribeServer（OpenAI 相容端點）

        # ffmpeg：啟動時自動偵測（含編譯版 bundled <app>/ffmpeg/ffmpeg.exe）
        try:
            from ffmpeg_utils import find_ffmpeg
            self._ffmpeg_exe: Path | None = find_ffmpeg()
        except Exception:
            self._ffmpeg_exe = None

        # ── 早期套用：介面縮放與鏡像站（須在建構 UI 與引導畫面前生效）──
        try:
            _early = self._load_settings()
            ctk.set_widget_scaling(float(_early.get("ui_scale", 1.0)))
            import downloader as _dl
            _dl.set_mirror(_early.get("hf_mirror", ""))
        except Exception:
            pass

        self._build_ui()
        self._detect_all_devices()
        self._refresh_audio_devices()   # 音訊裝置獨立初始化，不依賴模型載入
        threading.Thread(target=self._startup_check, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 建構 ────────────────────────────────────────

    def _build_ui(self):
        # ── 標題列（含狀態摘要 + 下載進度條）─────────────────────────────
        # dev_bar 的模型/裝置/語系選擇已移至「模型」「設定」分頁，標題列僅
        # 保留豐富狀態摘要（就緒/核心/裝置/對齊），騰出下方版面空間。
        header = ctk.CTkFrame(self, corner_radius=0)
        header.pack(fill="x")

        title_row = ctk.CTkFrame(header, fg_color="transparent", height=54)
        title_row.pack(fill="x")
        title_row.pack_propagate(False)
        ctk.CTkLabel(
            title_row, text="  🎙 Qwen3 ASR 字幕生成器",
            font=FONT_TITLE, anchor="w"
        ).pack(side="left", padx=16, pady=8)

        self.status_dot = ctk.CTkLabel(
            title_row, text="⏳ 啟動中…",
            font=FONT_BODY, text_color="#AAAAAA", anchor="e", justify="right",
        )
        self.status_dot.pack(side="right", padx=16, pady=8)

        # 下載進度條（正常隱藏，由 _show_dl_bar / _hide_dl_bar 控制，置於標題列下緣）
        self.dl_bar = ctk.CTkProgressBar(header, height=6)
        self.dl_bar.set(0)

        # 分頁
        self.tabs = ctk.CTkTabview(self, anchor="nw")
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(8, 10))
        self.tabs.add("  音檔轉字幕  ")
        self.tabs.add("  批次辨識  ")
        self.tabs.add("  錄製轉換  ")
        self.tabs.add("  端點  ")
        self.tabs.add("  模型  ")
        self.tabs.add("  設定  ")

        self._build_file_tab(self.tabs.tab("  音檔轉字幕  "))
        self._build_batch_tab(self.tabs.tab("  批次辨識  "))
        self._build_rt_tab(self.tabs.tab("  錄製轉換  "))

        from endpoint_tab import EndpointTab
        self._endpoint_tab = EndpointTab(self.tabs.tab("  端點  "), self)
        self._endpoint_tab.pack(fill="both", expand=True)

        # 「模型」分頁：引擎/裝置/模型選擇 + 模型路徑/下載來源/CPU 效能
        #   widget 建立後回寫 self.device_var/device_combo/model_var/model_combo/
        #   reload_btn，既有 _on_models_ready / _detect_all_devices 邏輯零修改。
        from model_tab import ModelTab
        self._model_tab = ModelTab(
            self.tabs.tab("  模型  "), self,
            show_model_select=True, device_default="CPU",
        )
        self._model_tab.pack(fill="both", expand=True)

        from setting import SettingsTab
        self._settings_tab = SettingsTab(self.tabs.tab("  設定  "), self)
        self._settings_tab.pack(fill="both", expand=True)

    # ── 批次辨識 tab ───────────────────────────────────

    def _build_batch_tab(self, parent):
        from batch_tab import BatchTab
        tab_frame = ctk.CTkFrame(parent, fg_color="transparent")
        tab_frame.pack(fill="both", expand=True)
        tab_frame.columnconfigure(0, weight=1)
        tab_frame.rowconfigure(0, weight=1)
        self._batch_tab = BatchTab(
            tab_frame,
            engine=None,   # 引擎於模型載入完成後注入（_on_models_ready）
            open_subtitle_cb=lambda srt, audio, dz:
                SubtitleEditorWindow(self, srt, audio, dz),
        )
        self._batch_tab.grid(row=0, column=0, sticky="nsew")

    # ── 音檔轉字幕 tab ─────────────────────────────────

    def _build_file_tab(self, parent):
        # 選檔列
        row1 = ctk.CTkFrame(parent, fg_color="transparent")
        row1.pack(fill="x", padx=8, pady=(12, 4))

        self.file_entry = ctk.CTkEntry(
            row1, placeholder_text="選擇或拖曳音訊檔案…",
            font=FONT_BODY, height=34,
        )
        self.file_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(
            row1, text="瀏覽…", width=80, height=34, font=FONT_BODY,
            command=self._on_browse,
        ).pack(side="left")

        # 操作按鈕列
        row2 = ctk.CTkFrame(parent, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=4)

        self.convert_btn = ctk.CTkButton(
            row2, text="▶  開始轉換", width=130, height=36,
            font=FONT_BODY, state="disabled",
            command=self._on_convert,
        )
        self.convert_btn.pack(side="left", padx=(0, 10))

        self.open_dir_btn = ctk.CTkButton(
            row2, text="📁  開啟輸出資料夾", width=150, height=36,
            font=FONT_BODY, state="disabled",
            fg_color="gray35", hover_color="gray25",
            command=lambda: os.startfile(str(SRT_DIR)),
        )
        self.open_dir_btn.pack(side="left")

        self.verify_btn = ctk.CTkButton(
            row2, text="🔍  字幕驗證", width=120, height=36,
            font=FONT_BODY, state="disabled",
            fg_color="#1A3050", hover_color="#265080",
            command=self._on_verify,
        )
        self.verify_btn.pack(side="left", padx=(8, 0))

        # 說話者分離：預設開啟（模型未就緒時於載入後自動下載）
        self._diarize_var = ctk.BooleanVar(value=True)
        self.diarize_chk = ctk.CTkCheckBox(
            row2, text="說話者分離", variable=self._diarize_var,
            font=FONT_BODY, state="disabled",
            command=self._on_diarize_toggle,
        )
        self.diarize_chk.pack(side="left", padx=(20, 0))

        ctk.CTkLabel(row2, text="人數：", font=FONT_BODY,
                     text_color="#AAAAAA").pack(side="left", padx=(8, 2))
        self.n_spk_combo = ctk.CTkComboBox(
            row2,
            values=["自動", "2", "3", "4", "5", "6", "7", "8"],
            width=76, state="disabled", font=FONT_BODY,
        )
        self.n_spk_combo.set("自動")
        self.n_spk_combo.pack(side="left")

        # ── 時間軸對齊 checkbox（ForcedAligner 載入後才啟用）────────────
        self._align_var = ctk.BooleanVar(value=True)
        self.align_chk = ctk.CTkCheckBox(
            row2, text="時間軸對齊",
            variable=self._align_var,
            font=FONT_BODY, state="disabled",
            command=self._on_align_toggle,
        )
        self.align_chk.pack(side="left", padx=(18, 0))

        # 辨識提示（Hint / Context）
        hint_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hint_hdr.pack(fill="x", padx=8, pady=(6, 0))
        # 右側按鈕要在左側標籤之前 pack，才能正確定位
        ctk.CTkButton(
            hint_hdr, text="讀入 TXT…", width=100, height=26,
            font=("Microsoft JhengHei", 11),
            fg_color="gray35", hover_color="gray25",
            command=lambda: self._load_hint_txt(self.hint_box),
        ).pack(side="right")
        ctk.CTkLabel(
            hint_hdr, text="辨識提示（可選）：", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            hint_hdr,
            text="貼入歌詞、關鍵字或背景說明，可提升辨識準確度",
            font=("Microsoft JhengHei", 11),
            text_color="#555555",
        ).pack(side="left", padx=(6, 0))

        self.hint_box = ctk.CTkTextbox(
            parent, font=FONT_MONO, height=72,
        )
        self.hint_box.pack(fill="x", padx=8, pady=(2, 4))
        self._bind_ctx_menu(self.hint_box._textbox, is_text=True)

        # 進度
        prog_frame = ctk.CTkFrame(parent, fg_color="transparent")
        prog_frame.pack(fill="x", padx=8, pady=(4, 2))

        self.prog_label = ctk.CTkLabel(
            prog_frame, text="", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        )
        self.prog_label.pack(fill="x")

        self.prog_bar = ctk.CTkProgressBar(prog_frame, height=10)
        self.prog_bar.pack(fill="x", pady=(2, 0))
        self.prog_bar.set(0)

        # 記錄
        ctk.CTkLabel(
            parent, text="轉換記錄", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 2))

        self.file_log = ctk.CTkTextbox(
            parent, font=FONT_MONO, state="disabled",
        )
        self.file_log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── 錄製轉換 tab ───────────────────────────────────

    def _build_rt_tab(self, parent):
        # 功能說明：此功能為「錄製轉換」而非真正的即時轉換，
        # 辨識會在偵測到說話停頓時才進行。
        ctk.CTkLabel(
            parent,
            text="錄製轉換：邊錄音邊辨識，於說話停頓時將該段語音轉成文字（非逐字即時）",
            font=("Microsoft JhengHei", 12),
            text_color="#8899AA", anchor="w", justify="left",
        ).pack(fill="x", padx=8, pady=(12, 0))

        # 裝置選擇列
        dev_row = ctk.CTkFrame(parent, fg_color="transparent")
        dev_row.pack(fill="x", padx=8, pady=(6, 4))

        ctk.CTkLabel(dev_row, text="音訊輸入裝置：", font=FONT_BODY).pack(
            side="left", padx=(0, 8)
        )
        self.rt_dev_combo = ctk.CTkComboBox(
            dev_row, values=["偵測中…"], width=380, font=FONT_BODY,
        )
        self.rt_dev_combo.pack(side="left")

        ctk.CTkButton(
            dev_row, text="重新整理", width=80, height=30,
            font=FONT_BODY, fg_color="gray35", hover_color="gray25",
            command=self._refresh_audio_devices,
        ).pack(side="left", padx=8)

        # Hint 輸入列（即時模式）
        hint_row = ctk.CTkFrame(parent, fg_color="transparent")
        hint_row.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(hint_row, text="辨識提示：", font=FONT_BODY,
                     text_color="#AAAAAA").pack(side="left", padx=(0, 6))
        # 右側按鈕先 pack
        ctk.CTkButton(
            hint_row, text="讀入 TXT…", width=90, height=26,
            font=("Microsoft JhengHei", 11),
            fg_color="gray35", hover_color="gray25",
            command=lambda: self._load_hint_txt(self.rt_hint_entry, is_textbox=False),
        ).pack(side="right")
        self.rt_hint_entry = ctk.CTkEntry(
            hint_row,
            placeholder_text="（可選）貼入歌詞、關鍵字或說明文字…",
            font=FONT_BODY, height=30,
        )
        self.rt_hint_entry.pack(side="left", fill="x", expand=True)
        self._bind_ctx_menu(self.rt_hint_entry._entry, is_text=False)

        # 控制按鈕列
        btn_row = ctk.CTkFrame(parent, fg_color="transparent")
        btn_row.pack(fill="x", padx=8, pady=4)

        self.rt_start_btn = ctk.CTkButton(
            btn_row, text="▶  開始錄音", width=130, height=36,
            font=FONT_BODY, state="disabled",
            fg_color="#2E7D32", hover_color="#1B5E20",
            command=self._on_rt_start,
        )
        self.rt_start_btn.pack(side="left", padx=(0, 10))

        self.rt_stop_btn = ctk.CTkButton(
            btn_row, text="■  停止錄音", width=130, height=36,
            font=FONT_BODY, state="disabled",
            fg_color="#C62828", hover_color="#B71C1C",
            command=self._on_rt_stop,
        )
        self.rt_stop_btn.pack(side="left", padx=(0, 14))

        self.rt_status_lbl = ctk.CTkLabel(
            btn_row, text="", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        )
        self.rt_status_lbl.pack(side="left")

        ctk.CTkLabel(
            btn_row,
            text="（會在說話停頓中處理辨識）",
            font=("Microsoft JhengHei", 11),
            text_color="#666666",
        ).pack(side="left", padx=(12, 0))

        # 字幕顯示
        ctk.CTkLabel(
            parent, text="錄製字幕", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 2))

        self.rt_textbox = ctk.CTkTextbox(
            parent, font=("Microsoft JhengHei", 15), state="disabled",
        )
        self.rt_textbox.pack(fill="both", expand=True, padx=8, pady=(0, 6))

        # 即時追加保存列
        save_row = ctk.CTkFrame(parent, fg_color="transparent")
        save_row.pack(fill="x", padx=8, pady=(0, 2))

        self._rt_autosave_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            save_row, text="即時追加保存識別結果",
            variable=self._rt_autosave_var, font=FONT_BODY,
            command=self._on_rt_autosave_toggle,
        ).pack(side="left")

        self._rt_autosave_lbl = ctk.CTkLabel(
            save_row, text="（每段辨識完成即追加寫入 .txt，可隨時中斷不遺失）",
            font=("Microsoft JhengHei", 11), text_color="#888888", anchor="w",
        )
        self._rt_autosave_lbl.pack(side="left", padx=(8, 0))

        # 操作列
        act_row = ctk.CTkFrame(parent, fg_color="transparent")
        act_row.pack(fill="x", padx=8, pady=(0, 10))

        ctk.CTkButton(
            act_row, text="清除", width=80, height=32,
            font=FONT_BODY, fg_color="gray35", hover_color="gray25",
            command=self._on_rt_clear,
        ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            act_row, text="💾  儲存 SRT", width=120, height=32,
            font=FONT_BODY, command=self._on_rt_save,
        ).pack(side="left")

    # ── 說話者分離 UI 輔助 ───────────────────────────────────────────

    def _on_diarize_toggle(self):
        """說話者分離 checkbox 切換時，同步啟用／停用人數選擇器。
        若開啟但模型尚未就緒，於背景自動下載。"""
        on = self._diarize_var.get()
        self.n_spk_combo.configure(state="readonly" if on else "disabled")
        if on:
            eng = getattr(self, "engine", None)
            ready = bool(eng and getattr(eng, "diar_engine", None)
                         and eng.diar_engine.ready)
            if not ready and self._model_dir is not None:
                threading.Thread(
                    target=self._check_diarization_models, daemon=True
                ).start()

    # ── 時間軸對齊 UI ──────────────────────────────────

    def _on_align_toggle(self):
        """切換時間軸對齊。FA 未就緒時引導下載 chatllm ForcedAligner 模型。

        CPU(OpenVINO) 與 GPU(chatllm) 後端皆用 chatllm 原生 FA，流程一致。
        """
        # 關閉：直接停用
        if not self._align_var.get():
            if hasattr(self.engine, 'use_aligner'):
                self.engine.use_aligner = False
            return

        # 開啟：模型已就緒（_fa_bin 真值）→ 直接啟用（含重新勾選的情況）
        if getattr(self.engine, '_fa_bin', None):
            self.engine.use_aligner = True
            return

        # FA 尚未就緒：先取消勾選，背景檢查模型 → 缺少則引導下載
        self._align_var.set(False)
        threading.Thread(target=self._check_aligner_model, daemon=True).start()

    # ── 時間軸對齊模型：檢查 + 按需下載（chatllm 後端）────────────────

    def _aligner_dir(self) -> Path | None:
        """FA .bin 應放置的資料夾（與 ASR .bin 同層）。"""
        mp = getattr(self.engine, "_model_path", None)
        if mp:
            return Path(mp).parent
        return self._model_dir

    def _check_aligner_model(self):
        """背景執行緒：FA 模型在 → 重載 aligner；不在 → 詢問下載。

        crispasr（Whisper 核心）走 qwen3 ForcedAligner GGUF；其餘後端走 chatllm .bin。
        """
        backend = (self._settings or {}).get("backend", "openvino")
        if backend == "crispasr":
            self._check_aligner_gguf()
            return
        from downloader import quick_check_aligner
        fa_dir = self._aligner_dir()
        if fa_dir is None:
            return
        if quick_check_aligner(fa_dir):
            self.after(0, self._reload_aligner)
        else:
            self.after(0, lambda: self._ask_download_aligner(fa_dir))

    def _check_aligner_gguf(self):
        """背景執行緒（crispasr）：FA gguf 在 → 啟用；不在 → 詢問下載。"""
        from downloader import (quick_check_aligner_gguf, download_aligner_gguf,
                                aligner_gguf_filename)
        crispasr_dir = Path((self._settings or {}).get(
            "crispasr_dir", str(BASE_DIR / "crispasr")))
        fa_quant = (self._settings or {}).get("crisp_fa_quant", "q5")

        def _enable():
            path = crispasr_dir / aligner_gguf_filename(fa_quant)
            self.engine._aligner_path = path
            self.engine._fa_bin       = path
            self.engine.use_aligner   = True
            self.after(0, self._sync_align_checkbox)

        if quick_check_aligner_gguf(crispasr_dir, fa_quant):
            _enable()
            return

        def _ask():
            if not messagebox.askyesno(
                "時間軸對齊器",
                f"「時間軸對齊」需要 qwen3 ForcedAligner 模型（{fa_quant.upper()}，"
                "約 643 MB）。\n\n是否立即下載？",
            ):
                return

            def _dl():
                self.after(0, self._show_dl_bar)
                try:
                    download_aligner_gguf(crispasr_dir, fa_quant,
                                          progress_cb=self._on_dl_progress)
                except Exception as e:
                    msg = str(e)
                    self.after(0, self._hide_dl_bar)
                    self.after(0, lambda: messagebox.showerror(
                        "下載失敗", f"對齊器下載失敗：\n{msg}\n\n請確認網路後重試。"))
                    return
                self.after(0, self._hide_dl_bar)
                _enable()

            threading.Thread(target=_dl, daemon=True).start()

        self.after(0, _ask)

    def _ask_download_aligner(self, fa_dir: Path):
        """主執行緒：詢問是否下載 chatllm ForcedAligner 模型（約 939 MB）。"""
        fname = getattr(self.engine, "FA_BIN_NAME", "qwen3-focedaligner-0.6b.bin")
        answer = messagebox.askyesno(
            "時間軸對齊模型",
            "「時間軸對齊」需要額外下載 ForcedAligner 模型（約 939 MB）：\n"
            f"  • {fname}\n\n"
            "下載後即可產生精確的字級時間軸（純 chatllm，無需 PyTorch）。\n"
            "是否立即下載？",
        )
        if answer:
            threading.Thread(
                target=lambda: self._download_aligner_model(fa_dir), daemon=True
            ).start()

    def _download_aligner_model(self, fa_dir: Path):
        """背景執行緒：下載 FA 模型，完成後重新載入 aligner。"""
        from downloader import download_aligner
        self.after(0, self._show_dl_bar)
        self._set_status("⬇ 下載時間軸對齊模型…")
        try:
            download_aligner(fa_dir, progress_cb=self._on_dl_progress)
        except Exception as e:
            msg = str(e)
            self.after(0, self._hide_dl_bar)
            self.after(0, lambda: messagebox.showerror(
                "下載失敗",
                f"時間軸對齊模型下載失敗：\n{msg}\n\n請確認網路連線後重試。",
            ))
            self.after(0, lambda: self._set_status("❌ FA 模型下載失敗"))
            return
        self.after(0, self._hide_dl_bar)
        self.after(0, self._reload_aligner)

    def _reload_aligner(self):
        """重新載入 aligner（chatllm 引擎）並同步勾選狀態。"""
        def _w():
            try:
                self.engine._load_aligner(cb=self._set_status)
            except Exception:
                pass
            self.after(0, self._sync_align_checkbox)
        threading.Thread(target=_w, daemon=True).start()

    def _sync_align_checkbox(self):
        """依 aligner 載入結果同步「時間軸對齊」勾選與狀態。"""
        if not hasattr(self, 'align_chk'):
            return
        if getattr(self.engine, 'use_aligner', False):
            self.align_chk.configure(state="normal")
            self._align_var.set(True)
            self.engine.use_aligner = True
            self._set_status("✅ 時間軸對齊已啟用")
        else:
            self._align_var.set(False)
            self._set_status("⚠ 時間軸對齊模型未就緒")

    # ── Hint 輸入輔助 ─────────────────────────────────────────────────

    def _bind_ctx_menu(self, native_widget, is_text: bool = False):
        """為原生 tkinter widget 綁定右鍵貼上選單（支援 Text 與 Entry）。"""
        def show(event):
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(
                label="貼上",
                command=lambda: native_widget.event_generate("<<Paste>>"),
            )
            if is_text:
                menu.add_command(
                    label="全選",
                    command=lambda: native_widget.tag_add("sel", "1.0", "end"),
                )
                menu.add_separator()
                menu.add_command(
                    label="清除全部",
                    command=lambda: native_widget.delete("1.0", "end"),
                )
            else:
                menu.add_command(
                    label="全選",
                    command=lambda: native_widget.select_range(0, "end"),
                )
                menu.add_separator()
                menu.add_command(
                    label="清除全部",
                    command=lambda: native_widget.delete(0, "end"),
                )
            menu.tk_popup(event.x_root, event.y_root)
        native_widget.bind("<Button-3>", show)

    def _load_hint_txt(self, target, is_textbox: bool = True):
        """開啟 TXT 檔案，將內容填入 hint 輸入框。
        target     : CTkTextbox（is_textbox=True）或 CTkEntry（is_textbox=False）
        """
        path = filedialog.askopenfilename(
            title="選擇提示文字檔",
            filetypes=[("文字檔", "*.txt"), ("所有檔案", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="cp950", errors="replace") as f:
                    text = f.read()
            except Exception as e:
                messagebox.showerror("讀取失敗", str(e))
                return
        if is_textbox:
            target.delete("1.0", "end")
            target.insert("1.0", text)
        else:
            target.delete(0, "end")
            target.insert(0, text)

    def _sync_core_model_ui(self, settings: dict):
        """主執行緒：依 settings 同步「核心 + 模型」下拉（預設開放，恆常可選）。"""
        core, label = _ui_core_model(settings)
        if hasattr(self, "_model_tab"):
            self._model_tab.set_core(core)   # 刷新該核心的模型清單
        try:
            self.model_combo.configure(state="readonly")
            self.model_var.set(label)
        except Exception:
            pass

    def _refresh_model_combo(self, model_dir: Path):
        """主執行緒：動態顯示模型選項（沿用目前核心）。"""
        core = self.core_var.get() if hasattr(self, "core_var") else "Qwen"
        if hasattr(self, "_model_tab"):
            self._model_tab.set_core(core)

    def _refresh_model_combo_from_settings(self, settings: dict):
        """主執行緒：依 settings 同步核心+模型下拉。"""
        self._sync_core_model_ui(settings)

    def _detect_all_devices(self):
        """同時偵測 OpenVINO（CPU / Intel iGPU）與 Vulkan（NVIDIA / AMD）裝置。
        結果儲存在 self._all_devices，並更新 device_combo 選單。
        """
        # ── OpenVINO 裝置 ───────────────────────────────────────────────
        ov_labels = ["CPU"]
        igpu_list: list[dict] = []
        try:
            import openvino as ov
            core = ov.Core()
            for d in core.available_devices:
                if not d.startswith("GPU"):
                    continue
                try:
                    name = core.get_property(d, "FULL_DEVICE_NAME")
                except Exception:
                    name = d
                if "Intel" in name:
                    label = f"{d} ({name})"
                    ov_labels.append(label)
                    igpu_list.append({"device": d, "name": name, "label": label})
        except Exception:
            pass

        # ── Vulkan 裝置（NVIDIA / AMD）──────────────────────────────────
        nvidia_amd: list[dict] = []
        vk_diag: dict = {"devices": [], "all_devices": [], "raw": "",
                         "error": None, "exit_code": None, "exe_found": False}
        if _CHATLLM_AVAILABLE:
            chatllm_dir = str(_CHATLLM_DIR)
            if not _CHATLLM_DIR.exists():
                # 嘗試 chatllmtest 目錄（開發模式）
                chatllm_dir = str(BASE_DIR / "chatllmtest" / "chatllm_win_x64" / "bin")
            vk_diag = probe_vulkan_devices(chatllm_dir)
            nvidia_amd = [
                {"id": d["id"], "name": d["name"], "vram_free": d["vram_free"]}
                for d in vk_diag["devices"]
            ]

        self._all_devices = {
            "cpu":       True,
            "igpu":      igpu_list,
            "nvidia_amd": nvidia_amd,
        }
        self._vk_diag = vk_diag

        # ── 更新 device_combo ────────────────────────────────────────────
        all_labels = list(ov_labels)
        for dev in nvidia_amd:
            all_labels.append(f"GPU:{dev['id']} ({dev['name']}) [Vulkan]")

        self.device_combo.configure(values=all_labels)
        self.device_var.set(all_labels[0])

    def _gpu_diag_lines(self) -> list[tuple[str, str]]:
        """偵測不到獨立 GPU 時，依 self._vk_diag 產生要顯示給使用者的診斷行。

        目的：把「真的沒有 GPU」與「chatllm 列舉失敗（逾時 / 崩潰 / 卡在內顯
        跳不到第二張卡）」這兩種情況分開講清楚，而不是一律顯示「僅 CPU 可用」。

        回傳 [(顯示文字, 顏色), ...]。
        """
        diag = getattr(self, "_vk_diag", None) or {}
        GREY, WARN = "#888888", "#E0A030"

        # chatllm 不存在（純 OpenVINO 發行包）→ 沒得用 Vulkan，屬正常
        if not diag.get("exe_found", False):
            return [("ℹ 未偵測到獨立 GPU，僅 CPU 推理可用", GREY)]

        # 列舉到的全部裝置（含 CPU / 內顯），這份清單本身就是最有用的線索
        enumerated = [d.get("name", "?") for d in diag.get("all_devices", [])]

        # 探測過程出錯（逾時 / 崩潰）→ 不是「沒有 GPU」，是「沒問成」
        err = diag.get("error")
        if err:
            lines = [(f"⚠ GPU 偵測未完成：{err}", WARN)]
            if enumerated:
                lines.append((f"　chatllm 列舉到：{', '.join(enumerated)}", GREY))
            lines.append(("　已自動改用 CPU 推理；如需 GPU 請更新顯卡 / Vulkan 驅動", GREY))
            return lines

        # 有正常結束，但清單裡只有 CPU（典型：卡在內顯、跳不到獨立 GPU）
        if enumerated:
            return [
                ("ℹ 未偵測到可用的獨立 GPU，僅 CPU 推理可用", GREY),
                (f"　chatllm 僅列舉到：{', '.join(enumerated)}", GREY),
            ]

        # 完全沒有輸出
        return [("ℹ 未偵測到獨立 GPU，僅 CPU 推理可用", GREY)]

    # ── 設定檔讀寫（記住模型路徑）──────────────────────────────────────

    def _load_settings(self) -> dict:
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_settings(self, settings: dict):
        """儲存完整設定 dict 到 settings.json。
        schema:
          backend       : "openvino" | "chatllm"
          device        : "CPU" | "GPU.0 (Intel UHD...)" | "GPU:0 (NVIDIA...) [Vulkan]"
          cpu_model_size: "0.6B" | "1.7B"
          model_dir     : OpenVINO 模型資料夾
          model_path    : chatllm .bin 模型路徑（chatllm 後端用）
          chatllm_dir   : chatllm DLL 目錄
        """
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _patch_setting(self, key: str, value):
        """讀取現有設定、更新單一 key，再寫回 settings.json。"""
        s = self._load_settings()
        s[key] = value
        self._save_settings(s)

    def _apply_ui_prefs(self, settings: dict):
        """主執行緒：根據儲存的偏好設定同步 UI 控件與外觀。"""
        global VAD_THRESHOLD
        mode = settings.get("appearance_mode", "dark")
        ctk.set_appearance_mode(mode)
        # 介面縮放
        try:
            ctk.set_widget_scaling(float(settings.get("ui_scale", 1.0)))
        except Exception:
            pass
        # 鏡像站
        try:
            import downloader as _dl
            _dl.set_mirror(settings.get("hf_mirror", ""))
        except Exception:
            pass
        # VAD 閾值：從設定還原
        vad = settings.get("vad_threshold")
        if vad is not None:
            VAD_THRESHOLD = float(vad)
        # 全域輸出格式（SRT / 純文字）→ 同步至共用寫出層
        _subs.OUTPUT_FORMAT = (settings.get("output_format", "srt") or "srt").lower()
        if hasattr(self, "_settings_tab"):
            self._settings_tab.sync_prefs(settings)
        if hasattr(self, "_model_tab"):
            self._model_tab.sync_prefs(settings)

    def _on_chinese_mode_change(self, value: str):
        """輸出模式切換：繁體（OpenCC）or 簡體（直接輸出）。"""
        global _g_output_simplified
        _g_output_simplified = (value == "簡體")
        self._patch_setting("output_simplified", _g_output_simplified)
        # 同步更新 chatllm_engine 模組旗標（ChatLLM 後端使用）
        if _CHATLLM_AVAILABLE:
            import chatllm_engine as _ce
            _ce._output_simplified = _g_output_simplified

    def _on_appearance_change(self, value: str):
        """主題切換：深色 🌑 or 淺色 ☀。"""
        mode = "light" if value == "☀" else "dark"
        ctk.set_appearance_mode(mode)
        self._patch_setting("appearance_mode", mode)

    def _on_vocab_convert_change(self, on: bool):
        """簡繁詞彙轉換開關：繁體模式下 s2twp（開）/ s2t（關）。"""
        global _g_vocab_convert
        _g_vocab_convert = bool(on)
        self._patch_setting("vocab_convert", _g_vocab_convert)
        # 即時重建目前引擎的 OpenCC 轉換器（免重新載入模型）
        eng = getattr(self, "engine", None)
        if eng and hasattr(eng, "rebuild_cc"):
            eng.rebuild_cc()
        # 同步 chatllm_engine 模組旗標（ChatLLM 後端使用）
        if _CHATLLM_AVAILABLE:
            import chatllm_engine as _ce
            _ce._vocab_convert = _g_vocab_convert

    def _on_output_format_change(self, fmt: str):
        """全域輸出格式切換（"srt" | "txt"）。

        影響批次轉換、單檔轉換、錄製轉換與 API 端點：寫出層 subtitle_lines.
        write_transcript 讀此全域旗標決定產 .srt 或 .txt。端點另以此設定其預設
        response_format 與上傳網頁預設選項。
        """
        fmt = "txt" if str(fmt).lower() == "txt" else "srt"
        _subs.OUTPUT_FORMAT = fmt
        self._patch_setting("output_format", fmt)

    def _on_ui_scale_change(self, scale: float):
        """介面縮放：等比放大／縮小所有元件與字體（高 DPI 螢幕適用）。"""
        try:
            ctk.set_widget_scaling(float(scale))
        except Exception:
            pass
        self._patch_setting("ui_scale", round(float(scale), 2))

    def _on_mirror_change(self, base: str):
        """HuggingFace 鏡像站切換：空字串＝官方，否則改寫下載網域。"""
        base = (base or "").strip()
        try:
            import downloader as _dl
            _dl.set_mirror(base)
        except Exception:
            pass
        self._patch_setting("hf_mirror", base)

    def _settings_valid(self, s: dict) -> bool:
        """檢查設定是否足夠完整（不需要重新引導）。"""
        if not s:
            return False
        backend = s.get("backend", "")
        if backend == "chatllm":
            mdl  = s.get("model_path", "") or s.get("gguf_path", "")
            cdir = s.get("chatllm_dir", "")
            return bool(mdl and cdir and Path(mdl).exists() and Path(cdir).exists())
        elif backend == "crispasr":
            # crispasr.exe（核心）+ Breeze 模型皆就緒才算有效；否則進 onboarding。
            # 缺此分支會讓 backend=crispasr 永遠判為無效 → 切到 Whisper 後重啟卻又
            # 跳回引導畫面（與舊版守衛不持久化合併造成「切不過去」的無限循環）。
            try:
                from downloader import quick_check_crispasr, quick_check_breeze
                cdir  = Path(s.get("crispasr_dir", str(BASE_DIR / "crispasr")))
                quant = s.get("crisp_quant", "q5")
                return quick_check_crispasr(cdir) and quick_check_breeze(cdir, quant)
            except Exception:
                return False
        elif backend == "openvino":
            model_dir = s.get("model_dir", "")
            if not model_dir:
                return False
            # 至少 0.6B 必須存在
            from downloader import quick_check
            return quick_check(Path(model_dir))
        return False

    def _resolve_model_dir(self) -> Path | None:
        """
        依序檢查：
          1. 預設 portable 路徑（EXE 旁邊的 ov_models/）
          2. settings.json 記住的路徑
        回傳第一個模型完整的路徑，或 None（需要詢問使用者）。
        """
        from downloader import quick_check
        if quick_check(_DEFAULT_MODEL_DIR):
            return _DEFAULT_MODEL_DIR
        saved = self._load_settings().get("model_dir")
        if saved:
            p = Path(saved)
            if quick_check(p):
                return p
        return None

    # ── 啟動檢查：設定有效 → 直接載入；否則 → 引導畫面 ────────────────

    def _startup_check(self):
        """背景執行緒：確認設定有效性 → 必要時顯示引導畫面 → 載入模型。"""
        settings = self._load_settings()

        if not self._settings_valid(settings):
            # 顯示引導畫面（主執行緒）
            chosen = [None]
            evt = threading.Event()
            self.after(0, lambda: self._run_onboarding(chosen, evt))
            evt.wait()

            if chosen[0] is None:
                # 使用者取消 → 嘗試 CPU + 0.6B 預設值
                default_dir = _DEFAULT_MODEL_DIR
                from downloader import quick_check
                if quick_check(default_dir):
                    settings = {
                        "backend":        "openvino",
                        "device":         "CPU",
                        "cpu_model_size": "0.6B",
                        "model_dir":      str(default_dir),
                    }
                else:
                    self.after(0, lambda: self._set_status("⚠ 已取消，模型未載入"))
                    return
            else:
                settings = chosen[0]

            self._save_settings(settings)

        self._settings = settings

        # 套用 UI 偏好（簡繁模式 + 詞彙轉換 + 外觀主題）
        global _g_output_simplified, _g_vocab_convert
        _g_output_simplified = settings.get("output_simplified", False)
        _g_vocab_convert     = settings.get("vocab_convert", True)
        # 同步 chatllm_engine 模組旗標
        if _CHATLLM_AVAILABLE:
            import chatllm_engine as _ce
            _ce._output_simplified = _g_output_simplified
            _ce._vocab_convert     = _g_vocab_convert
        self.after(0, lambda s=settings: self._apply_ui_prefs(s))

        # 同步 device_combo 到已儲存的裝置
        saved_dev = settings.get("device", "CPU")
        def _sync_device():
            vals = self.device_combo.cget("values")
            if saved_dev in vals:
                self.device_var.set(saved_dev)
        self.after(0, _sync_device)

        # 更新模型選單
        self.after(0, lambda: self._refresh_model_combo_from_settings(settings))

        self._set_status("⏳ 模型載入中…")
        self._load_models()

    # ── 引導畫面：硬體偵測 + 後端選擇 + 下載 ────────────────────────────

    def _run_onboarding(self, chosen: list, evt: threading.Event):
        """主執行緒：顯示初始設定引導畫面（modal）。
        chosen[0] = 選定設定 dict（或 None 表示取消）。
        """
        dlg = ctk.CTkToplevel(self)
        dlg.title("QwenASR 初始設定")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.focus_set()

        self.update_idletasks()
        scr_h  = dlg.winfo_screenheight()
        dlg_w  = 640
        dlg_h  = min(scr_h - 120, 660)   # 最多 660，低解析度自動縮短
        x = self.winfo_x() + (self.winfo_width()  - dlg_w) // 2
        y = max(40, self.winfo_y() + (self.winfo_height() - dlg_h) // 2)
        dlg.geometry(f"{dlg_w}x{dlg_h}+{x}+{y}")

        # ══ 底部按鈕列（先 pack → 永遠可見，不被內容擠走）══════════════
        bottom_bar = ctk.CTkFrame(dlg, fg_color="#252525", height=72)
        bottom_bar.pack(side="bottom", fill="x")
        bottom_bar.pack_propagate(False)

        # 分隔線
        ctk.CTkFrame(dlg, fg_color="#3A3A3A", height=1).pack(
            side="bottom", fill="x"
        )

        confirm_btn = ctk.CTkButton(
            bottom_bar,
            text="✔  確認並開始下載",
            width=200, height=44,
            font=("Microsoft JhengHei", 14, "bold"),
            corner_radius=8,
        )
        confirm_btn.pack(side="left", padx=(24, 10), pady=14)

        ctk.CTkButton(
            bottom_bar,
            text="取消",
            width=110, height=44,
            font=("Microsoft JhengHei", 14),
            fg_color="gray35", hover_color="gray25",
            corner_radius=8,
            command=lambda: _cancel_onboarding(),
        ).pack(side="left", padx=0, pady=14)

        # ══ 可捲動內容區（低解析度也能捲動到底）═════════════════════════
        scroll = ctk.CTkScrollableFrame(dlg, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # ── 標題 ──────────────────────────────────────────────────────
        ctk.CTkLabel(
            scroll, text="🎙  QwenASR 初始設定",
            font=("Microsoft JhengHei", 18, "bold"), anchor="w",
        ).pack(fill="x", padx=24, pady=(20, 4))

        ctk.CTkLabel(
            scroll, text="首次啟動需要選擇推理方式並下載對應模型。",
            font=FONT_BODY, text_color="#AAAAAA", anchor="w",
        ).pack(fill="x", padx=24, pady=(0, 12))

        # ── 偵測到的裝置 ──────────────────────────────────────────────
        dev_frame = ctk.CTkFrame(scroll, fg_color="#1E1E1E", corner_radius=8)
        dev_frame.pack(fill="x", padx=24, pady=(0, 14))

        ctk.CTkLabel(
            dev_frame, text="偵測到的裝置", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(anchor="w", padx=12, pady=(8, 2))

        ctk.CTkLabel(dev_frame, text="✅ CPU（可用）", font=FONT_BODY, anchor="w").pack(
            anchor="w", padx=20, pady=2
        )
        igpu_list   = self._all_devices.get("igpu", [])
        nvidia_list = self._all_devices.get("nvidia_amd", [])
        for g in igpu_list:
            ctk.CTkLabel(
                dev_frame, text=f"✅ Intel GPU：{g['name']}", font=FONT_BODY, anchor="w",
            ).pack(anchor="w", padx=20, pady=2)
        for g in nvidia_list:
            vram_gb = g['vram_free'] / 1_073_741_824
            ctk.CTkLabel(
                dev_frame,
                text=f"✅ GPU：{g['name']}（可用 VRAM {vram_gb:.1f} GB，Vulkan）",
                font=FONT_BODY, anchor="w",
            ).pack(anchor="w", padx=20, pady=2)
        if not igpu_list and not nvidia_list:
            for text, color in self._gpu_diag_lines():
                ctk.CTkLabel(
                    dev_frame, text=text, font=FONT_BODY, text_color=color,
                    anchor="w", justify="left", wraplength=560,
                ).pack(anchor="w", padx=20, pady=2)
        ctk.CTkLabel(dev_frame, text="").pack(pady=2)

        # ── 後端選擇 ──────────────────────────────────────────────────
        ctk.CTkLabel(
            scroll, text="選擇推理方式：", font=FONT_BODY, anchor="w",
        ).pack(fill="x", padx=24, pady=(0, 6))

        backend_var = ctk.StringVar(value="openvino_cpu")
        opt_frame   = ctk.CTkFrame(scroll, fg_color="transparent")
        opt_frame.pack(fill="x", padx=24, pady=(0, 10))

        # CPU 選項框
        cpu_box = ctk.CTkFrame(opt_frame, fg_color="#1E1E1E", corner_radius=8)
        cpu_box.pack(fill="x", pady=(0, 6))

        ctk.CTkRadioButton(
            cpu_box, text="CPU 推理（OpenVINO）",
            variable=backend_var, value="openvino_cpu",
            font=FONT_BODY,
        ).pack(anchor="w", padx=12, pady=(10, 4))

        size_frame = ctk.CTkFrame(cpu_box, fg_color="transparent")
        size_frame.pack(fill="x", padx=32, pady=(0, 10))
        size_var = ctk.StringVar(value="0.6B")
        ctk.CTkRadioButton(
            size_frame, text="0.6B 輕量（~1.2 GB，速度快）",
            variable=size_var, value="0.6B", font=FONT_BODY,
            command=lambda: backend_var.set("openvino_cpu"),
        ).pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(
            size_frame, text="1.7B 高精度（~4.3 GB）",
            variable=size_var, value="1.7B", font=FONT_BODY,
            command=lambda: backend_var.set("openvino_cpu"),
        ).pack(side="left")

        # GPU 選項框（有 NVIDIA/AMD 才顯示）
        if nvidia_list:
            gpu_options = [f"GPU:{g['id']} ({g['name']}) [Vulkan]" for g in nvidia_list]
            gpu_box = ctk.CTkFrame(opt_frame, fg_color="#1E1E1E", corner_radius=8)
            gpu_box.pack(fill="x", pady=(0, 6))
            gpu_var = ctk.StringVar(value=gpu_options[0] if gpu_options else "")
            ctk.CTkRadioButton(
                gpu_box, text="GPU 推理（Vulkan，速度最快）",
                variable=backend_var, value="chatllm",
                font=FONT_BODY,
            ).pack(anchor="w", padx=12, pady=(10, 4))
            for opt in gpu_options:
                ctk.CTkRadioButton(
                    gpu_box, text=f"  {opt}",
                    variable=gpu_var, value=opt, font=FONT_BODY,
                    command=lambda: backend_var.set("chatllm"),
                ).pack(anchor="w", padx=32, pady=2)
            ctk.CTkLabel(
                gpu_box,
                text="  1.7B .bin 格式（~2.3 GB），需先下載",
                font=("Microsoft JhengHei", 11), text_color="#888888",
            ).pack(anchor="w", padx=32, pady=(0, 10))
        else:
            gpu_var = ctk.StringVar(value="")

        # ── 路徑設定（模型存放位置）────────────────────────────────────
        path_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        path_frame.pack(fill="x", padx=24, pady=(0, 8))
        ctk.CTkLabel(path_frame, text="模型存放位置：", font=FONT_BODY).pack(
            side="left", padx=(0, 6)
        )
        saved_dir = self._load_settings().get("model_dir", str(_DEFAULT_MODEL_DIR))
        path_var = ctk.StringVar(value=saved_dir)
        ctk.CTkEntry(path_frame, textvariable=path_var, width=280, font=FONT_BODY).pack(
            side="left"
        )
        def _browse_dir():
            d = filedialog.askdirectory(title="選擇模型存放資料夾", parent=dlg)
            if d:
                path_var.set(d)
        ctk.CTkButton(
            path_frame, text="瀏覽…", width=70, font=FONT_BODY,
            command=_browse_dir,
        ).pack(side="left", padx=(6, 0))

        # ── 下載來源（HuggingFace 鏡像站）──────────────────────────────
        saved_mirror = (self._load_settings().get("hf_mirror", "") or "").strip()
        mirror_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        mirror_frame.pack(fill="x", padx=24, pady=(0, 8))
        ctk.CTkLabel(mirror_frame, text="下載來源：", font=FONT_BODY).pack(
            side="left", padx=(0, 6)
        )
        mirror_seg = ctk.CTkSegmentedButton(
            mirror_frame, values=["官方 HF", "鏡像站"],
            width=150, height=30, font=FONT_BODY,
        )
        mirror_entry_var = ctk.StringVar(
            value=saved_mirror or "https://hf-mirror.com"
        )
        mirror_entry = ctk.CTkEntry(
            mirror_frame, textvariable=mirror_entry_var, width=210, font=FONT_BODY,
        )

        def _on_onb_mirror(_v=None):
            use = "鏡" in mirror_seg.get()
            mirror_entry.configure(state="normal" if use else "disabled")

        mirror_seg.configure(command=_on_onb_mirror)
        mirror_seg.set("鏡像站" if saved_mirror else "官方 HF")
        mirror_seg.pack(side="left")
        mirror_entry.pack(side="left", padx=(8, 0))
        _on_onb_mirror()

        ctk.CTkLabel(
            scroll,
            text="直連 huggingface.co 緩慢或逾時時，可改用鏡像站（預設 hf-mirror.com）。",
            font=("Microsoft JhengHei", 11), text_color="#888888", anchor="w",
            wraplength=560, justify="left",
        ).pack(fill="x", padx=24, pady=(0, 8))

        # ── 下載進度條（平時隱藏）──────────────────────────────────────
        prog_frame = ctk.CTkFrame(scroll, fg_color="transparent")
        prog_frame.pack(fill="x", padx=24, pady=(0, 8))
        onb_prog_lbl = ctk.CTkLabel(
            prog_frame, text="", font=("Microsoft JhengHei", 11),
            text_color="#AAAAAA", anchor="w",
        )
        onb_prog_lbl.pack(fill="x")
        onb_bar = ctk.CTkProgressBar(prog_frame, height=10)
        onb_bar.set(0)
        onb_bar.pack(fill="x")
        onb_bar.pack_forget()
        onb_prog_lbl.pack_forget()

        def _onb_progress(pct: float, msg: str):
            def _do():
                onb_bar.set(pct)
                onb_prog_lbl.configure(text=msg)
            dlg.after(0, _do)
            self._set_status(f"⬇ {msg}")

        def _show_onb_prog():
            onb_prog_lbl.pack(fill="x")
            onb_bar.pack(fill="x")

        def _hide_onb_prog():
            onb_bar.pack_forget()
            onb_prog_lbl.pack_forget()

        def _cancel_onboarding():
            chosen[0] = None
            dlg.destroy()
            evt.set()

        def _do_download():
            """背景執行緒：執行下載動作，完成後關閉引導畫面。"""
            import downloader as _dl
            from downloader import (quick_check, download_all,
                                    quick_check_1p7b, download_1p7b)

            backend    = backend_var.get()
            model_path = Path(path_var.get().strip())
            model_path.mkdir(parents=True, exist_ok=True)

            # 套用下載來源（鏡像站）並記住，供後續更新沿用
            mirror_base = (mirror_entry_var.get().strip()
                           if "鏡" in mirror_seg.get() else "")
            _dl.set_mirror(mirror_base)
            self._patch_setting("hf_mirror", mirror_base)

            # 禁用按鈕
            dlg.after(0, lambda: confirm_btn.configure(state="disabled", text="⏳  下載中…"))
            dlg.after(0, _show_onb_prog)

            try:
                if backend == "chatllm":
                    # 確保 VAD 存在（OpenVINO onboarding 才呼叫 download_all；
                    # chatllm 路徑需要另外確認）
                    vad_dest = _DEFAULT_MODEL_DIR / "silero_vad_v4.onnx"
                    if not vad_dest.exists():
                        self._set_status("⬇ 下載 VAD 模型…")
                        from downloader import _download_file, _VAD_URL
                        _DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
                        _download_file(_VAD_URL, vad_dest)

                    # 下載 chatllm .bin 模型至「使用者指定的模型資料夾」。
                    # 注意：model_path 變數此處實為資料夾（path_var）。
                    # 舊版寫死 _BIN_PATH（EXE 旁），導致使用者指定路徑無效、
                    # 即使該資料夾已有 .bin 仍強制下載。改為以 model_dir 為準。
                    bin_dest = model_path / "qwen3-asr-1.7b.bin"
                    bin_dest.parent.mkdir(parents=True, exist_ok=True)
                    if not bin_dest.exists():
                        self._set_status("⬇ 下載 chatllm 模型（~2.3 GB）…")
                        url = ("https://huggingface.co/dseditor/Collection"
                               "/resolve/main/qwen3-asr-1.7b.bin")

                        def _dl_bin():
                            import ssl, urllib.request
                            from downloader import _ssl_ctx, _apply_mirror
                            req = urllib.request.Request(
                                _apply_mirror(url),
                                headers={"User-Agent": "Mozilla/5.0 (compatible; QwenASR)"}
                            )
                            with urllib.request.urlopen(req, context=_ssl_ctx()) as resp, \
                                 open(str(bin_dest) + ".tmp", "wb") as out:
                                total = int(resp.headers.get("Content-Length", 0))
                                done  = 0
                                while True:
                                    block = resp.read(65536)
                                    if not block:
                                        break
                                    out.write(block)
                                    done += len(block)
                                    if total > 0:
                                        pct = done / total
                                        mb  = done / 1_048_576
                                        tmb = total / 1_048_576
                                        dlg.after(0, lambda p=pct, m=mb, t=tmb:
                                            _onb_progress(p, f"下載模型 {m:.0f} / {t:.0f} MB"))
                            import os
                            os.replace(str(bin_dest) + ".tmp", str(bin_dest))
                        _dl_bin()

                    # chatllm_dir：優先 chatllm/，fallback chatllmtest
                    cl_dir = _CHATLLM_DIR if _CHATLLM_DIR.exists() else \
                             BASE_DIR / "chatllmtest" / "chatllm_win_x64" / "bin"

                    # 選取的 GPU device
                    gpu_label = gpu_var.get()   # e.g. "GPU:0 (NVIDIA...) [Vulkan]"

                    final_settings = {
                        "backend":      "chatllm",
                        "device":       gpu_label,
                        "model_dir":    str(model_path),
                        "model_path":   str(bin_dest),
                        "chatllm_dir":  str(cl_dir),
                    }

                else:  # openvino_cpu
                    sz = size_var.get()   # "0.6B" | "1.7B"
                    # 下載 0.6B（必要）
                    if not quick_check(model_path):
                        self._set_status("⬇ 下載 0.6B 模型…")
                        download_all(model_path, progress_cb=_onb_progress)

                    # 下載 1.7B（若選擇）
                    if sz == "1.7B" and not quick_check_1p7b(model_path):
                        self._set_status("⬇ 下載 1.7B 模型（~4.3 GB）…")
                        download_1p7b(model_path, progress_cb=_onb_progress)

                    final_settings = {
                        "backend":        "openvino",
                        "device":         "CPU",
                        "cpu_model_size": sz,
                        "model_dir":      str(model_path),
                    }

                dlg.after(0, lambda: _onb_progress(1.0, "下載完成！"))
                dlg.after(0, _hide_onb_prog)
                chosen[0] = final_settings
                dlg.after(0, dlg.destroy)
                evt.set()

            except Exception as e:
                err = str(e)
                dlg.after(0, _hide_onb_prog)
                dlg.after(0, lambda: confirm_btn.configure(
                    state="normal", text="✔  確認並開始下載"
                ))
                dlg.after(0, lambda: messagebox.showerror(
                    "下載失敗", f"下載失敗：\n{err}\n\n請確認網路連線後重試。", parent=dlg
                ))

        confirm_btn.configure(command=lambda: threading.Thread(
            target=_do_download, daemon=True,
        ).start())

        dlg.protocol("WM_DELETE_WINDOW", _cancel_onboarding)

    def _on_dl_progress(self, pct: float, msg: str):
        self.after(0, lambda: self.dl_bar.set(pct))
        self.after(0, lambda: self._set_status(f"⬇ {msg} ({pct*100:.0f}%)"))

    def _show_dl_bar(self):
        # 置於標題列下緣，橫跨整個寬度
        self.dl_bar.pack(fill="x", side="bottom")

    def _hide_dl_bar(self):
        self.dl_bar.pack_forget()

    def _ask_yesno_sync(self, title: str, msg: str) -> bool:
        """從背景執行緒安全地彈出 askyesno（在主執行緒顯示並等待回應）。"""
        ev = threading.Event()
        box = {"ans": False}

        def _ask():
            box["ans"] = messagebox.askyesno(title, msg)
            ev.set()

        self.after(0, _ask)
        ev.wait()
        return box["ans"]

    def _load_models(self):
        import gc

        # ── 釋放舊引擎記憶體 ───────────────────────────────────────────
        for attr in ("audio_enc", "embedder", "dec_req", "vad_sess",
                     "pf_model", "dc_model", "_llm"):
            if hasattr(self.engine, attr):
                setattr(self.engine, attr, None)
        gc.collect()

        # ── 讀取設定：先用儲存的，再 fallback 至 UI 選擇 ───────────────
        settings       = self._settings or self._load_settings()
        backend        = settings.get("backend", "openvino")
        device_label   = settings.get("device", self.device_var.get())
        # 解析 OV 裝置名（如 "GPU.0 (Intel...)" → "GPU.0"）
        ov_device      = device_label.split(" (")[0].split(" [")[0]
        # 防呆：模型驅動 backend 後，若選到 Vulkan 裝置(GPU:N)卻走 OpenVINO，
        # 該名稱對 OV 無效 → 退回 CPU（Vulkan 裝置僅供 chatllm / crispasr 使用）。
        if "Vulkan" in device_label or ov_device.startswith("GPU:"):
            if backend == "openvino":
                ov_device = "CPU"

        if backend == "chatllm":
            # ── chatllm / Vulkan 路線 ──────────────────────────────────
            if not _CHATLLM_AVAILABLE:
                self.after(0, lambda: self._on_models_failed(
                    "chatllm", "chatllm_engine 無法載入，請確認 chatllm/ 目錄"
                ))
                return

            # 向下相容：新 key=model_path，舊 key=gguf_path
            _saved_mdl  = settings.get("model_path") or settings.get("gguf_path") or str(_BIN_PATH)
            model_path  = Path(_saved_mdl)
            chatllm_dir = Path(settings.get("chatllm_dir", str(_CHATLLM_DIR)))

            # 若記住的 .bin 不存在，依序在「使用者指定的模型資料夾」與預設
            # _BIN_PATH 內尋找同名 .bin，避免明明已有檔案卻強制重新下載。
            if not model_path.exists():
                for _cand in (
                    Path(settings.get("model_dir", "")) / "qwen3-asr-1.7b.bin"
                        if settings.get("model_dir") else None,
                    _BIN_PATH,
                ):
                    if _cand is not None and _cand.exists():
                        model_path = _cand
                        break

            # chatllm .bin 是否存在
            if not model_path.exists():
                self.after(0, self._show_dl_bar)
                self._set_status("⬇ 下載 chatllm 模型（~2.3 GB）…")
                try:
                    import urllib.request
                    from downloader import _ssl_ctx
                    url = ("https://huggingface.co/dseditor/Collection"
                           "/resolve/main/qwen3-asr-1.7b.bin")
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    req = urllib.request.Request(
                        url, headers={"User-Agent": "Mozilla/5.0 (compatible; QwenASR)"}
                    )
                    with urllib.request.urlopen(req, context=_ssl_ctx()) as resp, \
                         open(str(model_path) + ".tmp", "wb") as out:
                        total = int(resp.headers.get("Content-Length", 0))
                        done  = 0
                        while True:
                            block = resp.read(65536)
                            if not block:
                                break
                            out.write(block)
                            done += len(block)
                            if total > 0:
                                self._on_dl_progress(done / total,
                                    f"模型 {done/1_048_576:.0f}/{total/1_048_576:.0f} MB")
                    import os as _os
                    _os.replace(str(model_path) + ".tmp", str(model_path))
                    self.after(0, self._hide_dl_bar)
                except Exception as e:
                    msg = str(e)
                    self.after(0, self._hide_dl_bar)
                    self.after(0, lambda: messagebox.showerror(
                        "下載失敗",
                        f"chatllm 模型下載失敗：\n{msg}\n\n請確認網路連線後點「重新載入」重試。",
                    ))
                    self.after(0, lambda: self._set_status("❌ 下載失敗"))
                    self.after(0, lambda: self.reload_btn.configure(state="normal"))
                    return

            # 持久化完整設定（確保下次啟動不會重觸 onboarding）
            settings["model_path"]  = str(model_path)
            settings["chatllm_dir"] = str(chatllm_dir)
            self._settings = settings
            self._save_settings(settings)

            # 設定 _model_dir 供 diarization 下載確認流程使用
            self._model_dir = Path(settings.get("model_dir", str(BASE_DIR / "ov_models")))

            # 從 device_label 解析 Vulkan device ID
            # 格式：「GPU:0 (AMD Radeon(TM) Graphics) [Vulkan]」
            _vk_dev_id = 0
            _m = re.search(r"GPU:(\d+)", device_label)
            if _m:
                _vk_dev_id = int(_m.group(1))

            self.engine = ChatLLMASREngine()
            try:
                self.engine.load(
                    model_path  = model_path,
                    chatllm_dir = chatllm_dir,
                    n_gpu_layers= 99,
                    device_id   = _vk_dev_id,
                    cb          = self._set_status,
                )
                self.after(0, self._on_models_ready)
            except Exception as e:
                first_line = str(e).splitlines()[0][:120]
                self.after(0, lambda r=first_line: self._on_models_failed("chatllm", r))

        elif backend == "crispasr":
            # ── CrispASR(Whisper/Breeze) / Vulkan 路線 ──────────────────
            if not _CRISPASR_AVAILABLE:
                self.after(0, lambda: self._on_models_failed(
                    "crispasr", "crisp_engine 無法載入，請確認 crisp_engine.py"
                ))
                return
            from downloader import (quick_check_crispasr, download_crispasr_core,
                                    quick_check_breeze, download_breeze, breeze_filename,
                                    quick_check_aligner_gguf, download_aligner_gguf,
                                    aligner_gguf_filename)

            crispasr_dir = Path(settings.get("crispasr_dir", str(BASE_DIR / "crispasr")))
            quant        = settings.get("crisp_quant", "q5")
            model_path   = crispasr_dir / breeze_filename(quant)
            # FA 對齊器（預設開）：缺檔才引導下載；使用者婉拒則退回 whisper native
            fa_enabled   = bool(settings.get("crisp_fa", True))
            fa_quant     = settings.get("crisp_fa_quant", "q5")

            def _abort(reason: str):
                self.after(0, self._hide_dl_bar)
                self.after(0, lambda: self.reload_btn.configure(state="normal"))
                self.after(0, lambda r=reason: self._set_status(r))

            # ① 檢查並取得核心（無 → 確認後下載解壓）
            if not quick_check_crispasr(crispasr_dir):
                if not self._ask_yesno_sync(
                    "下載 CrispASR 核心",
                    "Whisper 核心（CrispASR Vulkan，約 27 MB）尚未安裝。\n"
                    "首次使用需下載並解壓，是否立即下載？",
                ):
                    _abort("已取消（缺少 CrispASR 核心）")
                    return
                self.after(0, self._show_dl_bar)
                try:
                    download_crispasr_core(crispasr_dir, progress_cb=self._on_dl_progress)
                except Exception as e:
                    msg = str(e)
                    self.after(0, self._hide_dl_bar)
                    self.after(0, lambda: self.reload_btn.configure(state="normal"))
                    self.after(0, lambda m=msg: messagebox.showerror(
                        "下載失敗", f"CrispASR 核心下載失敗：\n{m}\n\n請確認網路後重試。"))
                    self.after(0, lambda: self._set_status("❌ 核心下載失敗"))
                    return
                self.after(0, self._hide_dl_bar)

            # ② 檢查並取得模型（無 → 確認後下載）
            if not quick_check_breeze(crispasr_dir, quant):
                size_hint = {"q4": "約 889 MB", "q5": "約 1.08 GB",
                             "q8": "約 1.66 GB"}.get(quant, "")
                if not self._ask_yesno_sync(
                    "下載 Whisper 模型",
                    f"Breeze-ASR-26 {quant.upper()} 模型（{size_hint}）尚未下載。\n"
                    "是否立即下載？",
                ):
                    _abort("已取消（缺少 Whisper 模型）")
                    return
                self.after(0, self._show_dl_bar)
                try:
                    download_breeze(crispasr_dir, quant, progress_cb=self._on_dl_progress)
                except Exception as e:
                    msg = str(e)
                    self.after(0, self._hide_dl_bar)
                    self.after(0, lambda: self.reload_btn.configure(state="normal"))
                    self.after(0, lambda m=msg: messagebox.showerror(
                        "下載失敗", f"Whisper 模型下載失敗：\n{m}\n\n請確認網路後重試。"))
                    self.after(0, lambda: self._set_status("❌ 模型下載失敗"))
                    return
                self.after(0, self._hide_dl_bar)

            # ②.5 時間軸對齊器（預設開）：缺 gguf → 確認後下載（婉拒則退回 native）
            aligner_path = None
            if fa_enabled:
                if quick_check_aligner_gguf(crispasr_dir, fa_quant):
                    aligner_path = crispasr_dir / aligner_gguf_filename(fa_quant)
                elif self._ask_yesno_sync(
                    "下載時間軸對齊器",
                    f"Whisper 自帶時間軸較粗。時間軸對齊器（qwen3 ForcedAligner "
                    f"{fa_quant.upper()}，約 643 MB）可產生精確字級時間軸。\n\n"
                    "是否立即下載？（婉拒則使用 Whisper 自帶時間軸）",
                ):
                    self.after(0, self._show_dl_bar)
                    try:
                        download_aligner_gguf(crispasr_dir, fa_quant,
                                              progress_cb=self._on_dl_progress)
                        aligner_path = crispasr_dir / aligner_gguf_filename(fa_quant)
                    except Exception as e:
                        msg = str(e)
                        self.after(0, self._hide_dl_bar)
                        self.after(0, lambda m=msg: messagebox.showwarning(
                            "下載失敗",
                            f"對齊器下載失敗：\n{m}\n\n將先使用 Whisper 自帶時間軸。"))
                        aligner_path = None
                    self.after(0, self._hide_dl_bar)

            # ③ 持久化設定 + 載入引擎（核心與模型皆就緒，不重複下載）
            settings["crispasr_dir"] = str(crispasr_dir)
            settings["crisp_quant"]  = quant
            settings["crisp_fa"]     = fa_enabled
            settings["crisp_fa_quant"] = fa_quant
            self._settings = settings
            self._save_settings(settings)

            _vk_dev_id = 0
            _m = re.search(r"GPU:(\d+)", device_label)
            if _m:
                _vk_dev_id = int(_m.group(1))
            self._model_dir = Path(settings.get("model_dir", str(BASE_DIR / "ov_models")))
            self.engine = CrispWhisperEngine()
            try:
                self.engine.load(
                    model_path   = model_path,
                    crispasr_dir = crispasr_dir,
                    device_id    = _vk_dev_id,
                    cb           = self._set_status,
                    aligner_path = aligner_path,
                )
                self.after(0, self._on_models_ready)
            except Exception as e:
                first_line = str(e).splitlines()[0][:120]
                self.after(0, lambda r=first_line: self._on_models_failed("crispasr", r))

        else:
            # ── OpenVINO 路線 ──────────────────────────────────────────
            model_dir  = Path(settings.get("model_dir", str(_DEFAULT_MODEL_DIR)))
            model_size = settings.get("cpu_model_size", self.model_var.get())
            self._model_dir = model_dir

            # 1.7B 按需下載
            use_17b = "1.7B" in model_size
            if use_17b:
                from downloader import quick_check_1p7b, download_1p7b
                if not quick_check_1p7b(model_dir):
                    self.after(0, self._show_dl_bar)
                    self._set_status("⬇ 下載 1.7B 模型（約 4.3 GB）…")
                    try:
                        download_1p7b(model_dir, progress_cb=self._on_dl_progress)
                    except Exception as e:
                        msg = str(e)
                        self.after(0, self._hide_dl_bar)
                        self.after(0, lambda: self.reload_btn.configure(state="normal"))
                        self.after(0, lambda: messagebox.showerror(
                            "下載失敗",
                            f"1.7B 模型下載失敗：\n{msg}\n\n"
                            "請確認網路連線後點「重新載入」重試。",
                        ))
                        self.after(0, lambda: self._set_status("❌ 下載失敗"))
                        return
                    self.after(0, self._hide_dl_bar)

            cpu_threads = int(settings.get("cpu_threads", 0))
            self.engine = ASREngine1p7B() if use_17b else ASREngine()
            try:
                self.engine.load(device=ov_device, model_dir=model_dir, cb=self._set_status,
                                 cpu_threads=cpu_threads)
                self.after(0, self._on_models_ready)
            except Exception as e:
                first_line = str(e).splitlines()[0][:120]
                self.after(0, lambda d=ov_device, r=first_line: self._on_models_failed(d, r))

    def _on_models_ready(self):
        self.device_combo.configure(state="readonly")
        self.reload_btn.configure(state="normal")
        self.convert_btn.configure(state="normal")
        self.rt_start_btn.configure(state="normal")
        # 注入引擎到批次辨識頁籤
        if hasattr(self, "_batch_tab"):
            self._batch_tab.set_engine(self.engine)

        # API 服務：若使用者先前啟用 → 模型就緒後自動開服（端點分頁）
        if hasattr(self, "_endpoint_tab"):
            self._endpoint_tab.start_api_if_enabled()

        settings = self._settings or {}
        backend  = settings.get("backend", "openvino")
        device   = self.device_var.get()
        # 記錄本行程實際載入的 backend（供跨 Vulkan 核心切換守衛判斷）
        self._loaded_backend = backend

        # ── 核心 + 模型下拉同步（預設開放，恆常可選）─────────────────────
        self._sync_core_model_ui(settings)
        self._set_status(self._ready_summary(device, backend))

        # 填入語系清單（模型載入後才知道 supported_languages）
        if self.engine.processor and self.engine.processor.supported_languages:
            langs = ["自動偵測"] + self.engine.processor.supported_languages
            self._lang_list = self.engine.processor.supported_languages
            self.lang_combo.configure(values=langs, state="readonly")
            self.lang_var.set("自動偵測")
        elif backend in ("chatllm", "crispasr"):
            # chatllm / CrispASR 模型支援多語系，提供常用語系清單
            common_langs = [
                "Chinese", "English", "Japanese", "Korean",
                "Cantonese", "French", "German", "Spanish",
                "Portuguese", "Russian", "Arabic", "Thai",
                "Vietnamese", "Indonesian", "Malay",
            ]
            self.lang_combo.configure(
                values=["自動偵測"] + common_langs, state="readonly"
            )
            self.lang_var.set("自動偵測")
        # 說話者分離 checkbox：啟用控件並依預設值同步人數選擇器
        self.diarize_chk.configure(state="normal")
        self.n_spk_combo.configure(
            state="readonly" if self._diarize_var.get() else "disabled"
        )
        if not (self.engine.diar_engine and self.engine.diar_engine.ready):
            # 模型未就緒：若說話者分離預設／使用者已開啟 → 自動下載
            if self._diarize_var.get():
                threading.Thread(
                    target=self._check_diarization_models, daemon=True
                ).start()

        # ForcedAligner checkbox：永不灰掉（CPU/GPU 後端皆用 chatllm 原生 FA）
        #   已就緒 → 勾選；未就緒 → 不勾選但仍可點（點擊時引導下載 .bin）。
        if hasattr(self, 'align_chk'):
            self.align_chk.configure(state="normal")
            self._align_var.set(bool(getattr(self.engine, 'use_aligner', False)))

    # ── 說話者分離模型：啟動時檢查 + 按需下載 ─────────────────────────

    def _check_diarization_models(self):
        """背景執行緒：說話者分離模型不存在時自動下載（不再詢問）。

        說話者分離預設開啟，缺模型時直接於背景下載並重新載入。
        以 _diar_downloading 旗標避免（models_ready 與 toggle）重複觸發。
        """
        from downloader import quick_check_diarization
        if not self._model_dir:
            return
        if quick_check_diarization(self._model_dir):
            return  # 模型已存在
        if getattr(self, "_diar_downloading", False):
            return  # 已有下載進行中
        self._diar_downloading = True
        self.after(0, lambda: threading.Thread(
            target=self._download_diarization_models, daemon=True
        ).start())

    def _download_diarization_models(self):
        """背景執行緒：下載說話者分離模型，完成後重新載入 DiarizationEngine。"""
        from downloader import download_diarization
        from diarize import DiarizationEngine

        diar_dir = self._model_dir / "diarization"
        self.after(0, self._show_dl_bar)
        self._set_status("⬇ 下載說話者分離模型…")
        try:
            download_diarization(diar_dir, progress_cb=self._on_dl_progress)
        except Exception as e:
            msg = str(e)
            self.after(0, self._hide_dl_bar)
            self.after(0, lambda: messagebox.showerror(
                "下載失敗",
                f"說話者分離模型下載失敗：\n{msg}\n\n請確認網路連線後重試。",
            ))
            self.after(0, lambda: self._set_status("❌ 下載失敗"))
            self._diar_downloading = False
            return

        self.after(0, self._hide_dl_bar)

        # 重新載入 DiarizationEngine
        try:
            eng = DiarizationEngine(diar_dir)
            if eng.ready:
                self.engine.diar_engine = eng
                self.after(0, lambda: self.diarize_chk.configure(state="normal"))
                device = self.device_var.get()
                self.after(0, lambda: self._set_status(f"✅ 就緒（{device}）"))
            else:
                self.after(0, lambda: messagebox.showerror(
                    "載入失敗", "說話者分離模型下載完成，但無法正常載入，請重新啟動程式。"
                ))
        except Exception as e:
            err = str(e)
            self.after(0, lambda: messagebox.showerror(
                "載入失敗", f"說話者分離模型載入失敗：{err}"
            ))
        finally:
            self._diar_downloading = False

    def _on_models_failed(self, device: str, reason: str):
        """模型載入失敗：若為 Vulkan（chatllm）後端，自動退回 CPU 重試；
        若本身已是 OpenVINO 路線，還原 UI 讓使用者手動選擇。
        """
        # ── 判斷是否為 Vulkan 後端失敗 ──────────────────────────────────
        failed_backend = (self._settings or {}).get("backend", "openvino")

        if failed_backend == "chatllm":
            # Vulkan 引擎（AMD / NVIDIA）失敗 → 自動 fallback 到 CPU
            # 1. 通知使用者（非阻塞式，因為要繼續觸發 fallback 載入）
            self.after(0, lambda: messagebox.showwarning(
                "GPU 引擎失敗，自動退回 CPU",
                f"Vulkan GPU（{device}）載入失敗：\n{reason}\n\n"
                "已自動切換為 CPU 模式重新載入，請稍候…",
            ))
            # 2. 更新設定與 UI 選單至 CPU
            fallback: dict = dict(self._settings) if self._settings else {}
            fallback["backend"] = "openvino"
            fallback["device"]  = "CPU"
            self._settings = fallback
            self._save_settings(fallback)
            self.device_var.set("CPU")
            # 3. 在背景執行緒重新以 CPU 載入（不阻塞 UI thread）
            self.engine.ready = False
            threading.Thread(target=self._load_models, daemon=True).start()
        else:
            # OpenVINO 路線失敗（GPU.0 Intel iGPU 等）→ 還原 UI 讓使用者重試
            self.device_combo.configure(state="readonly")
            self.reload_btn.configure(state="normal")
            self.status_dot.configure(
                text=f"❌ {device} 載入失敗，請切換裝置後點「重新載入」",
                text_color="#EF5350",
            )
            messagebox.showerror(
                "模型載入失敗",
                f"裝置「{device}」載入失敗：\n{reason}\n\n"
                "建議：將裝置切換為 CPU 後點「重新載入」。",
            )

    def _on_reload_models(self):
        if self._converting:
            messagebox.showwarning("提示", "轉換進行中，請等候完成後再重新載入")
            return
        if self._rt_mgr:
            self._on_rt_stop()

        # 從 UI 狀態同步設定（允許使用者在 dev_bar 手動切換裝置後重新載入）
        dev_label  = self.device_var.get()
        model_sel  = self.model_var.get()
        cur        = dict(self._settings) if self._settings else self._load_settings()

        # 三層選擇：核心(core) + 模型(model) → backend（裝置維持現狀，僅選 CPU/GPU）
        core_sel       = self.core_var.get() if hasattr(self, "core_var") else "Qwen"
        backend, extra = _resolve_backend(core_sel, model_sel)

        # ── 跨 Vulkan 核心切換守衛 ────────────────────────────────────────
        # chatllm 與 crispasr 各自持有 Vulkan GPU context（chatllm 的 DLL context
        # 行程結束前不會釋放）。在同一次執行中由其一切到另一，兩個 Vulkan context
        # 並存 → 觸發顯卡驅動 TDR / 整機重啟。故偵測到跨 Vulkan 核心切換即擋下，
        # 還原下拉並請使用者重啟程式後再切。
        _VK = ("chatllm", "crispasr")
        if (self._loaded_backend in _VK and backend in _VK
                and backend != self._loaded_backend):
            # 不能在同一次執行中熱切換（兩個 Vulkan context 並存 → TDR），但「必須把
            # 新選擇寫入 settings.json」。舊版在此 return 前未持久化、且把下拉還原回舊
            # 核心，導致重啟後讀回舊 backend → 永遠切不過去（無限循環）。
            # 修法：先持久化新核心，UI 維持在新選擇，重啟後 _load_settings 自動以新核心載入。
            cur["backend"] = backend
            cur["device"]  = dev_label
            if backend == "crispasr":
                cur["crisp_quant"] = extra            # q4 / q5 / q8
            self._settings = cur
            self._save_settings(cur)
            messagebox.showwarning(
                "需要重新啟動程式",
                "已記住您選擇的核心。\n\nQwen（Vulkan）與 Whisper（Vulkan）兩種 GPU 核心"
                "無法在同一次執行中切換——兩者若同時載入，顯示卡驅動會重置並導致整台電腦"
                "重新開機。\n\n請關閉本程式並重新啟動，啟動後將自動以新核心載入。",
            )
            return

        cur["backend"] = backend
        cur["device"]  = dev_label
        if backend == "crispasr":
            cur["crisp_quant"] = extra            # q4 / q5 / q8
        elif backend == "openvino":
            cur["cpu_model_size"] = extra         # 0.6B / 1.7B

        self._settings = cur

        self.engine.ready = False
        self.convert_btn.configure(state="disabled")
        self.rt_start_btn.configure(state="disabled")
        self.reload_btn.configure(state="disabled")
        threading.Thread(target=self._load_models, daemon=True).start()

    def _ready_summary(self, device: str, backend: str) -> str:
        """組合頂部標題列的豐富就緒摘要：模型 · 推理核心 · 時間軸對齊狀態。"""
        model_label = self.model_var.get()
        if backend == "chatllm":
            core = "GPU 推理（Vulkan）"
        elif backend == "crispasr":
            core = "GPU 推理（Vulkan / CrispASR）"
        elif device == "CPU":
            core = "CPU 推理"
        else:
            core = f"GPU 推理（{device}）"
        align = ("時間軸對齊已啟用"
                 if getattr(self.engine, "use_aligner", False)
                 else "時間軸對齊未啟用")
        return f"✅ 就緒 · {model_label} · {core} · {align}"

    def _set_status(self, msg: str):
        self.after(0, lambda: self.status_dot.configure(text=msg))

    def _refresh_audio_devices(self):
        try:
            import sounddevice as sd
            devs    = sd.query_devices()
            choices = []
            self._dev_idx_map = {}
            for i, d in enumerate(devs):
                if d["max_input_channels"] > 0:
                    name = d["name"][:50]
                    choices.append(name)
                    self._dev_idx_map[name] = i
            if choices:
                self.rt_dev_combo.configure(values=choices)
                default = sd.default.device[0]
                default_name = next(
                    (k for k, v in self._dev_idx_map.items() if v == default), choices[0]
                )
                self.rt_dev_combo.set(default_name)
        except ImportError:
            self.rt_dev_combo.configure(values=["（需安裝 sounddevice）"])

    # ── 音檔轉字幕操作 ─────────────────────────────────

    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="選擇音訊 / 影片檔案",
            filetypes=[
                ("音訊 / 影片檔案",
                 "*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.opus *.wma "
                 "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.ts"),
                ("音訊檔案", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac *.opus *.wma"),
                ("影片檔案", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv *.webm *.ts *.m2ts"),
                ("所有檔案", "*.*"),
            ],
        )
        if path:
            self._audio_file = Path(path)
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, str(self._audio_file))
            if self.engine.ready:
                self.convert_btn.configure(state="normal")

    def _on_verify(self):
        """開啟字幕驗證編輯視窗。"""
        if not self._srt_output or not self._srt_output.exists():
            messagebox.showwarning("提示", "尚無可驗證的字幕，請先執行轉換。")
            return
        # 純文字輸出無時間軸，字幕編輯器僅支援 SRT
        if self._srt_output.suffix.lower() != ".srt":
            messagebox.showinfo(
                "提示",
                "目前輸出為純文字（.txt），無時間軸可編輯。\n"
                "若需使用字幕編輯器，請至「設定 → 輸出格式」改回「SRT 字幕」後重新轉換。")
            return
        SubtitleEditorWindow(
            self,
            srt_path     = self._srt_output,
            audio_path   = self._audio_file,
            diarize_mode = getattr(self, "_file_diarize", False),
        )

    def _on_convert(self):
        if self._converting:
            return
        path = Path(self.file_entry.get().strip())
        if not path.exists():
            messagebox.showwarning("提示", "找不到檔案，請重新選擇")
            return
        if not self.engine.ready:
            messagebox.showwarning("提示", "模型尚未載入完成")
            return

        self._audio_file = path
        # 讀取語系、hint 與說話者分離選項（在主執行緒讀取 UI 值，再傳給 worker）
        lang_sel = self.lang_var.get()
        self._selected_language = lang_sel if lang_sel != "自動偵測" else None
        hint_text = self.hint_box.get("1.0", "end").strip()
        self._file_hint = hint_text if hint_text else None
        self._file_diarize = self._diarize_var.get()
        n_spk_sel = self.n_spk_combo.get()
        self._file_n_speakers = (int(n_spk_sel)
                                  if n_spk_sel.isdigit() else None)

        # 影片檔案需要 ffmpeg → 先確保可用
        from ffmpeg_utils import is_video, ensure_ffmpeg
        if is_video(path):
            def _on_ffmpeg_ready(ffmpeg_path):
                self._ffmpeg_exe = ffmpeg_path
                self._do_start_convert()
            ensure_ffmpeg(self, on_ready=_on_ffmpeg_ready)
            return   # 等 ensure_ffmpeg 回呼（同步有 ffmpeg 時也會回呼）

        # 非影片檔案：保留啟動時自動偵測到的 ffmpeg（供設定頁顯示），轉換不會用到
        self._do_start_convert()

    def _do_start_convert(self):
        """ffmpeg 確認後（或非影片檔案時）實際啟動轉換執行緒。"""
        self._converting = True
        self.convert_btn.configure(state="disabled", text="轉換中…")
        self.prog_bar.set(0)
        self._file_log_clear()
        threading.Thread(target=self._convert_worker, daemon=True).start()

    def _convert_worker(self):
        path = self._audio_file

        # 擷取語系、hint 與說話者分離（在主執行緒已取好，直接帶入 worker）
        language   = self._selected_language
        context    = self._file_hint
        diarize    = getattr(self, "_file_diarize", False)
        n_speakers = getattr(self, "_file_n_speakers", None)
        ffmpeg_exe = getattr(self, "_ffmpeg_exe", None)

        def prog_cb(done, total, msg):
            pct = done / total if total > 0 else 0
            self.after(0, lambda: self.prog_bar.set(pct))
            self.after(0, lambda: self.prog_label.configure(text=msg))
            self._file_log(msg)

        tmp_wav: Path | None = None
        try:
            t0 = time.perf_counter()
            # 影片音軌提取
            from ffmpeg_utils import is_video, extract_audio_to_wav
            if is_video(path):
                if not ffmpeg_exe:
                    raise RuntimeError("找不到 ffmpeg，無法提取影片音軌。")
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(tmp_fd)
                tmp_wav = Path(tmp_path)
                self._file_log(f"🎬 提取音軌中：{path.name}")
                extract_audio_to_wav(path, tmp_wav, ffmpeg_exe)
                self._file_log(f"   音軌提取完成，開始辨識…")
                proc_path = tmp_wav
            else:
                proc_path = path

            lang_info  = f"  語系：{language or '自動'}"
            hint_info  = f"  提示：{context[:30]}…" if context and len(context) > 30 else (f"  提示：{context}" if context else "")
            if diarize:
                n_str = str(n_speakers) if n_speakers else "自動"
                diar_info = f"  [說話者分離，人數：{n_str}]"
            else:
                diar_info = ""
            self._file_log(f"開始處理：{path.name}{lang_info}{hint_info}{diar_info}")
            srt = self.engine.process_file(
                proc_path, progress_cb=prog_cb, language=language,
                context=context, diarize=diarize, n_speakers=n_speakers,
                original_path=path,
            )
            elapsed = time.perf_counter() - t0

            if srt:
                self._srt_output = srt
                self._file_log(f"\n✅ 完成！耗時 {elapsed:.1f}s")
                self._file_log(f"SRT 儲存至：{srt}")
                self.after(0, lambda: [
                    self.prog_bar.set(1.0),
                    self.open_dir_btn.configure(state="normal"),
                    self.verify_btn.configure(state="normal"),
                    self.prog_label.configure(text="完成"),
                ])
            else:
                diag = getattr(self.engine, "_last_vad_diag", None)
                self._file_log(diag or "⚠ 未偵測到人聲，未產生字幕")
                self.after(0, lambda: self.prog_bar.set(0))
        except Exception as e:
            self._file_log(f"❌ 錯誤：{e}")
            self.after(0, lambda: self.prog_bar.set(0))
        finally:
            # 清理臨時 WAV（影片音軌提取）
            if tmp_wav and tmp_wav.exists():
                try:
                    tmp_wav.unlink()
                except Exception:
                    pass
            self._converting = False
            self.after(0, lambda: self.convert_btn.configure(
                state="normal", text="▶  開始轉換"
            ))

    def _file_log(self, msg: str):
        def _do():
            self.file_log.configure(state="normal")
            self.file_log.insert("end", msg + "\n")
            self.file_log.see("end")
            self.file_log.configure(state="disabled")
        self.after(0, _do)

    def _file_log_clear(self):
        self.file_log.configure(state="normal")
        self.file_log.delete("1.0", "end")
        self.file_log.configure(state="disabled")

    # ── 即時轉換操作 ───────────────────────────────────

    def _on_rt_start(self):
        name = self.rt_dev_combo.get()
        idx  = self._dev_idx_map.get(name)
        if idx is None:
            messagebox.showwarning("提示", "請選擇有效的音訊輸入裝置")
            return

        lang_sel = self.lang_var.get()
        rt_lang  = lang_sel if lang_sel != "自動偵測" else None
        rt_hint  = self.rt_hint_entry.get().strip() or None

        self._rt_mgr = RealtimeManager(
            asr=self.engine,
            device_idx=idx,
            on_text=self._on_rt_text,
            on_status=self._on_rt_status,
            language=rt_lang,
            context=rt_hint,
        )
        try:
            self._rt_mgr.start()
        except Exception as e:
            messagebox.showerror("錯誤", f"無法開啟音訊裝置：{e}")
            self._rt_mgr = None
            return

        self.rt_start_btn.configure(state="disabled")
        self.rt_stop_btn.configure(state="normal")

    def _on_rt_stop(self):
        if self._rt_mgr:
            self._rt_mgr.stop()
            self._rt_mgr = None
        self.rt_start_btn.configure(state="normal")
        self.rt_stop_btn.configure(state="disabled")

    def _on_rt_autosave_toggle(self):
        """切換即時追加保存。開啟時建立 .txt 並寫入既有內容。"""
        if self._rt_autosave_var.get():
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = SRT_DIR / f"realtime_{ts}.txt"
            try:
                SRT_DIR.mkdir(parents=True, exist_ok=True)
                # 先寫入目前已累積的內容，之後逐段追加
                with open(path, "w", encoding="utf-8") as f:
                    for line in self._rt_log:
                        f.write(line + "\n")
                self._rt_autosave_path = path
                self._rt_autosave_lbl.configure(
                    text=f"✅ 追加保存中：{path.name}",
                    text_color=("green", "#88CC88"),
                )
            except Exception as e:
                self._rt_autosave_var.set(False)
                self._rt_autosave_path = None
                messagebox.showerror("錯誤", f"無法建立保存檔案：{e}")
        else:
            self._rt_autosave_path = None
            self._rt_autosave_lbl.configure(
                text="（每段辨識完成即追加寫入 .txt，可隨時中斷不遺失）",
                text_color="#888888",
            )

    def _rt_autosave_append(self, line: str):
        """即時追加單行到保存檔（背景安全；失敗則靜默停用）。"""
        path = self._rt_autosave_path
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            self._rt_autosave_path = None

    def _on_rt_text(self, text: str):
        self._rt_log.append(text)
        # 即時追加保存（在背景執行緒即寫入，確保中斷不遺失）
        self._rt_autosave_append(text)
        def _do():
            ts = datetime.now().strftime("%H:%M:%S")
            self.rt_textbox.configure(state="normal")
            self.rt_textbox.insert("end", f"[{ts}]  {text}\n")
            self.rt_textbox.see("end")
            self.rt_textbox.configure(state="disabled")
        self.after(0, _do)

    def _on_rt_status(self, msg: str):
        self.after(0, lambda: self.rt_status_lbl.configure(text=msg))

    def _on_rt_clear(self):
        self._rt_log.clear()
        self.rt_textbox.configure(state="normal")
        self.rt_textbox.delete("1.0", "end")
        self.rt_textbox.configure(state="disabled")

    def _on_rt_save(self):
        if not self._rt_log:
            messagebox.showinfo("提示", "目前沒有字幕內容可儲存")
            return
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 合成時間軸（錄製轉換無精確時間戳，每行固定 5s）→ 共用寫出層依全域格式產檔
        lines: list[tuple[float, float, str, str | None]] = []
        t = 0.0
        for line in self._rt_log:
            end = t + 5.0
            lines.append((t, end, line, None))
            t = end + 0.1
        out = write_transcript(SRT_DIR / f"realtime_{ts}", lines)
        messagebox.showinfo("儲存完成", f"已儲存至：\n{out}")
        os.startfile(str(SRT_DIR))

    # ── 關閉處理 ───────────────────────────────────────

    def _on_close(self):
        # 轉換進行中：請使用者確認
        if self._converting:
            if not messagebox.askyesno(
                "確認關閉",
                "音訊轉換正在進行中。\n確定要強制關閉嗎？（目前進度將遺失）",
                icon="warning",
                default="no",
            ):
                return

        # 停止 API 服務與對外通道
        if hasattr(self, "_endpoint_tab"):
            self._endpoint_tab.stop_all()

        # 停止即時錄音（安靜地停，不需要確認）
        if self._rt_mgr:
            try:
                self._rt_mgr.stop()
            except Exception:
                pass

        # 銷毀視窗，再強制終止 process。
        # os._exit(0) 確保 OpenVINO / onnxruntime 的 C++ 背景執行緒
        # 不會讓程式殘留在工作管理員中。
        self.destroy()
        os._exit(0)


# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    # 綁進 Job Object：辨識衍生的 crispasr.exe / chatllm main.exe(FA) 子程序
    # 會自動繼承同一 Job，主行程結束（含崩潰/被強制結束）時連帶被終止，
    # 杜絕孤兒程序殘留。與 app_webview.py 共用 proc_guard 邏輯。
    from proc_guard import setup_kill_on_close_job
    setup_kill_on_close_job()

    app = App()
    app.mainloop()
