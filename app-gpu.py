"""
Qwen3 ASR 字幕生成器 - GPU 版本（PyTorch 版本，暫停更新）

推理後端：PyTorch (CUDA / CPU)，使用 Qwen3-ASR-1.7B
模型路徑：GPUModel/Qwen3-ASR-1.7B
          GPUModel/Qwen3-ForcedAligner-0.6B（可選）

此檔案不納入 EXE 構建，供有 NVIDIA GPU 的使用者以
系統 Python 或獨立虛擬環境執行。
啟動方式：start-gpu.bat

NOTE: 此 PyTorch GPU 版本目前暫停更新，僅供保留舊版參考。
      主要 GPU 支援已改為 chatllm_engine.py（Vulkan 後端），
      可於 app.py 的 EXE 版本中使用，無須 PyTorch 環境。
      若有需要再行維護此 PyTorch 版本。
"""
from __future__ import annotations

# ── UTF-8 模式：在所有其他 import 之前設定 ────────────────────────────
import os as _os, sys as _sys, io as _io
_os.environ.setdefault("PYTHONUTF8", "1")
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
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import customtkinter as ctk

# ── 路徑 ──────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
GPU_MODEL_DIR   = BASE_DIR / "GPUModel"
OV_MODEL_DIR    = BASE_DIR / "ov_models"      # 借用 CPU 版的 VAD 模型
SETTINGS_FILE   = BASE_DIR / "settings-gpu.json"
SRT_DIR         = BASE_DIR / "subtitles"
SRT_DIR.mkdir(exist_ok=True)

ASR_MODEL_NAME      = "Qwen3-ASR-1.7B"
ALIGNER_MODEL_NAME  = "Qwen3-ForcedAligner-0.6B"

# ── 語系清單（與 CPU 版相同，來自 Qwen3-ASR 規格）────────────────────
SUPPORTED_LANGUAGES = [
    "Chinese", "English", "Cantonese", "Arabic", "German", "French",
    "Spanish", "Portuguese", "Indonesian", "Italian", "Korean", "Russian",
    "Thai", "Vietnamese", "Japanese", "Turkish", "Hindi", "Malay",
    "Dutch", "Swedish", "Danish", "Finnish", "Polish", "Czech",
    "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian",
]

# ── 常數 ──────────────────────────────────────────────
SAMPLE_RATE          = 16000
VAD_CHUNK            = 512
VAD_THRESHOLD        = 0.5
MAX_GROUP_SEC        = 20
MAX_CHARS            = 20
MIN_SUB_SEC          = 0.6
GAP_SEC              = 0.08
RT_SILENCE_CHUNKS    = 25
RT_MAX_BUFFER_CHUNKS = 600


# ══════════════════════════════════════════════════════
# 共用工具函式（與 app.py 相同）
# ══════════════════════════════════════════════════════

def _detect_speech_groups(audio: np.ndarray, vad_sess) -> list[tuple[float, float, np.ndarray]]:
    """Silero VAD 分段，回傳 [(start_s, end_s, chunk), ...]"""
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

    mx_samp = MAX_GROUP_SEC * SAMPLE_RATE
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


def _find_vad_model() -> Path | None:
    """依序在 GPUModel/ 和 ov_models/ 尋找 Silero VAD ONNX。"""
    candidates = [
        GPU_MODEL_DIR / "silero_vad_v4.onnx",
        OV_MODEL_DIR  / "silero_vad_v4.onnx",
        GPU_MODEL_DIR / "silero_vad.onnx",
        OV_MODEL_DIR  / "silero_vad.onnx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ══════════════════════════════════════════════════════
# GPU ASR 引擎
# ══════════════════════════════════════════════════════

class GPUASREngine:
    """PyTorch 推理引擎。使用 qwen_asr 官方 API，支援 CUDA / CPU。"""

    def __init__(self):
        self.ready      = False
        self._lock      = threading.Lock()
        self.vad_sess   = None
        self.model      = None   # Qwen3ASRModel
        self.device     = "cpu"
        self.cc         = None
        self.diar_engine = None

    def load(self, device: str = "cuda", model_dir: Path = None, cb=None):
        """從背景執行緒呼叫。device: 'cuda' 或 'cpu'。"""
        import torch
        import onnxruntime as ort
        import opencc
        from qwen_asr import Qwen3ASRModel

        if model_dir is None:
            model_dir = GPU_MODEL_DIR

        asr_path = model_dir / ASR_MODEL_NAME

        def _s(msg):
            if cb: cb(msg)

        # ── VAD（ONNX CPU，輕量）──────────────────────────────────────
        _s("載入 VAD 模型…")
        vad_path = _find_vad_model()
        if vad_path is None:
            raise FileNotFoundError(
                "找不到 Silero VAD 模型 (silero_vad_v4.onnx)。\n"
                f"請將模型放入 {GPU_MODEL_DIR} 或先執行 CPU 版本下載。"
            )
        self.vad_sess = ort.InferenceSession(
            str(vad_path), providers=["CPUExecutionProvider"]
        )

        # ── 說話者分離（可選，沿用 ov_models/diarization）─────────────
        _s("載入說話者分離模型…")
        try:
            from diarize import DiarizationEngine
            diar_dir = OV_MODEL_DIR / "diarization"
            eng = DiarizationEngine(diar_dir)
            self.diar_engine = eng if eng.ready else None
        except Exception:
            self.diar_engine = None

        # ── PyTorch ASR 模型 ──────────────────────────────────────────
        _s(f"載入 ASR 模型（{asr_path.name}）…")
        if not asr_path.exists():
            raise FileNotFoundError(
                f"找不到 ASR 模型：{asr_path}\n"
                f"請將 {ASR_MODEL_NAME} 放入 {model_dir}"
            )

        import torch
        self.device = device.lower()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        _s(f"編譯模型（{device.upper()}，{str(dtype).split('.')[-1]}）…")
        self.model = Qwen3ASRModel.from_pretrained(
            str(asr_path),
            device_map=self.device,
            dtype=dtype,
        )

        self.cc    = opencc.OpenCC("s2twp")
        self.ready = True
        _s(f"就緒（{device.upper()}  {ASR_MODEL_NAME}）")

    def transcribe(
        self,
        audio: np.ndarray,
        max_tokens: int = 300,          # 保留參數以維持介面相容性
        language: str | None = None,
        context: str | None = None,
    ) -> str:
        """將 16kHz float32 音訊轉錄為繁體中文。"""
        with self._lock:
            results = self.model.transcribe(
                [(audio, SAMPLE_RATE)],
                language=language,
                context=context or "",
            )
            text = results[0].text if results else ""
            return self.cc.convert(text.strip())

    def process_file(
        self,
        audio_path: Path,
        progress_cb=None,
        language: str | None = None,
        context: str | None = None,
        diarize: bool = False,
        n_speakers: int | None = None,
    ) -> Path | None:
        """音檔 → SRT，回傳 SRT 路徑。"""
        import librosa
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

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
            vad_groups = _detect_speech_groups(audio, self.vad_sess)
            if not vad_groups:
                return None
            groups_spk = [(g0, g1, chunk, None) for g0, g1, chunk in vad_groups]

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

        out = SRT_DIR / (audio_path.stem + ".srt")
        with open(out, "w", encoding="utf-8") as f:
            for idx, (s, e, line, spk) in enumerate(all_subs, 1):
                prefix = f"{spk}：" if spk else ""
                f.write(f"{idx}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{prefix}{line}\n\n")
        return out


# ══════════════════════════════════════════════════════
# 即時轉錄管理員（與 app.py 相同）
# ══════════════════════════════════════════════════════

class RealtimeManager:
    def __init__(self, asr, device_idx, on_text, on_status,
                 language=None, context=None):
        self.asr       = asr
        self.dev_idx   = device_idx
        self.on_text   = on_text
        self.on_status = on_status
        self.language  = language
        self.context   = context
        self._q        = queue.Queue()
        self._running  = False
        self._stream   = None

    def start(self):
        import sounddevice as sd
        self._running = True
        self._stream  = sd.InputStream(
            device=self.dev_idx, samplerate=SAMPLE_RATE,
            channels=1, blocksize=VAD_CHUNK, dtype="float32",
            callback=self._audio_cb,
        )
        threading.Thread(target=self._loop, daemon=True).start()
        self._stream.start()
        self.on_status("🔴 錄音中…")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop(); self._stream.close(); self._stream = None
        self.on_status("⏹ 已停止")

    def _audio_cb(self, indata, frames, time_info, status):
        self._q.put(indata[:, 0].copy())

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
                if sil >= RT_SILENCE_CHUNKS or len(buf) >= RT_MAX_BUFFER_CHUNKS:
                    audio = np.concatenate(buf)
                    n = max(1, len(audio) // SAMPLE_RATE) * SAMPLE_RATE
                    try:
                        text = self.asr.transcribe(
                            audio[:n], language=self.language, context=self.context
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


class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Qwen3 ASR 字幕生成器 [GPU]")
        self.geometry("960x700")
        self.minsize(800, 580)

        self.engine       = GPUASREngine()
        self._rt_mgr: RealtimeManager | None = None
        self._rt_log: list[str]              = []
        self._audio_file: Path | None        = None
        self._srt_output: Path | None        = None
        self._converting                     = False
        self._dev_idx_map: dict[str, int]    = {}
        self._selected_language: str | None  = None
        self._file_hint: str | None          = None
        self._file_diarize: bool             = False
        self._file_n_speakers: int | None    = None

        self._build_ui()
        self._detect_devices()
        self._refresh_audio_devices()
        threading.Thread(target=self._startup_check, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 建構 ────────────────────────────────────────

    def _build_ui(self):
        title_bar = ctk.CTkFrame(self, height=54, corner_radius=0)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar, text="  🎙 Qwen3 ASR 字幕生成器  ⚡ GPU",
            font=FONT_TITLE, anchor="w"
        ).pack(side="left", padx=16, pady=8)

        dev_bar = ctk.CTkFrame(self, height=46)
        dev_bar.pack(fill="x", padx=10, pady=(6, 0))
        dev_bar.pack_propagate(False)

        ctk.CTkLabel(dev_bar, text="推理裝置：", font=FONT_BODY).pack(
            side="left", padx=(14, 4), pady=12
        )
        self.device_var   = ctk.StringVar(value="CUDA")
        self.device_combo = ctk.CTkComboBox(
            dev_bar, values=["CUDA"], variable=self.device_var,
            width=160, state="disabled", font=FONT_BODY,
        )
        self.device_combo.pack(side="left", pady=12)

        self.reload_btn = ctk.CTkButton(
            dev_bar, text="重新載入", width=90, state="disabled",
            font=FONT_BODY, fg_color="gray35", hover_color="gray25",
            command=self._on_reload_models,
        )
        self.reload_btn.pack(side="left", padx=8, pady=12)

        ctk.CTkLabel(dev_bar, text="語系：", font=FONT_BODY).pack(
            side="left", padx=(12, 2), pady=12
        )
        self.lang_var   = ctk.StringVar(value="自動偵測")
        self.lang_combo = ctk.CTkComboBox(
            dev_bar, values=["自動偵測"] + SUPPORTED_LANGUAGES,
            variable=self.lang_var,
            width=130, state="disabled", font=FONT_BODY,
        )
        self.lang_combo.pack(side="left", pady=12)

        self.status_dot = ctk.CTkLabel(
            dev_bar, text="⏳ 啟動中…",
            font=FONT_BODY, text_color="#AAAAAA", anchor="w"
        )
        self.status_dot.pack(side="left", padx=12, pady=12)

        self.tabs = ctk.CTkTabview(self, anchor="nw")
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(8, 10))
        self.tabs.add("  音檔轉字幕  ")
        self.tabs.add("  即時轉換  ")

        self._build_file_tab(self.tabs.tab("  音檔轉字幕  "))
        self._build_rt_tab(self.tabs.tab("  即時轉換  "))

    # ── 音檔轉字幕 tab ─────────────────────────────────

    def _build_file_tab(self, parent):
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

        self._diarize_var = ctk.BooleanVar(value=False)
        self.diarize_chk = ctk.CTkCheckBox(
            row2, text="說話者分離", variable=self._diarize_var,
            font=FONT_BODY, state="disabled",
            command=self._on_diarize_toggle,
        )
        self.diarize_chk.pack(side="left", padx=(20, 0))

        ctk.CTkLabel(row2, text="人數：", font=FONT_BODY,
                     text_color="#AAAAAA").pack(side="left", padx=(8, 2))
        self.n_spk_combo = ctk.CTkComboBox(
            row2, values=["自動", "2", "3", "4", "5", "6", "7", "8"],
            width=76, state="disabled", font=FONT_BODY,
        )
        self.n_spk_combo.set("自動")
        self.n_spk_combo.pack(side="left")

        hint_hdr = ctk.CTkFrame(parent, fg_color="transparent")
        hint_hdr.pack(fill="x", padx=8, pady=(6, 0))
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
            font=("Microsoft JhengHei", 11), text_color="#555555",
        ).pack(side="left", padx=(6, 0))

        self.hint_box = ctk.CTkTextbox(parent, font=FONT_MONO, height=72)
        self.hint_box.pack(fill="x", padx=8, pady=(2, 4))
        self._bind_ctx_menu(self.hint_box._textbox, is_text=True)

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

        ctk.CTkLabel(
            parent, text="轉換記錄", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 2))

        self.file_log = ctk.CTkTextbox(parent, font=FONT_MONO, state="disabled")
        self.file_log.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ── 即時轉換 tab ───────────────────────────────────

    def _build_rt_tab(self, parent):
        dev_row = ctk.CTkFrame(parent, fg_color="transparent")
        dev_row.pack(fill="x", padx=8, pady=(12, 4))

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

        hint_row = ctk.CTkFrame(parent, fg_color="transparent")
        hint_row.pack(fill="x", padx=8, pady=(0, 4))
        ctk.CTkLabel(hint_row, text="辨識提示：", font=FONT_BODY,
                     text_color="#AAAAAA").pack(side="left", padx=(0, 6))
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
            btn_row, text="", font=FONT_BODY, text_color="#AAAAAA", anchor="w",
        )
        self.rt_status_lbl.pack(side="left")

        ctk.CTkLabel(
            btn_row, text="（會在說話停頓中處理辨識）",
            font=("Microsoft JhengHei", 11), text_color="#666666",
        ).pack(side="left", padx=(12, 0))

        ctk.CTkLabel(
            parent, text="即時字幕", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 2))

        self.rt_textbox = ctk.CTkTextbox(
            parent, font=("Microsoft JhengHei", 15), state="disabled",
        )
        self.rt_textbox.pack(fill="both", expand=True, padx=8, pady=(0, 6))

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

    # ── 裝置偵測 ───────────────────────────────────────

    def _detect_devices(self):
        """偵測 CUDA 可用性，建立裝置選項清單。"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name  = torch.cuda.get_device_name(0)
                vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1024**3
                cuda_label = f"CUDA  ({gpu_name[:24]}, {vram_gb:.0f}GB)"
                options = [cuda_label, "CPU"]
                self.device_combo.configure(values=options, state="readonly")
                self.device_var.set(cuda_label)
                self._cuda_label = cuda_label   # 記住完整標籤
            else:
                self.device_combo.configure(values=["CPU"], state="readonly")
                self.device_var.set("CPU")
                self._cuda_label = None
        except ImportError:
            self.device_combo.configure(values=["CPU"], state="readonly")
            self.device_var.set("CPU")
            self._cuda_label = None

    def _get_torch_device(self) -> str:
        """將 UI 選項轉換成 torch device 字串。"""
        if hasattr(self, "_cuda_label") and self.device_var.get() == self._cuda_label:
            return "cuda"
        return "cpu"

    # ── 啟動檢查 ───────────────────────────────────────

    def _startup_check(self):
        """背景執行緒：檢查模型存在 → 載入。"""
        asr_path = GPU_MODEL_DIR / ASR_MODEL_NAME
        if not asr_path.exists():
            self.after(0, lambda: self._show_missing_model_error(asr_path))
            return
        self._set_status("⏳ 模型載入中…")
        self._load_models()

    def _show_missing_model_error(self, missing: Path):
        self._set_status("❌ 找不到模型")
        messagebox.showerror(
            "找不到 GPU 模型",
            f"找不到 ASR 模型：\n{missing}\n\n"
            f"請將 {ASR_MODEL_NAME} 下載並放入：\n{GPU_MODEL_DIR}\n\n"
            "可執行 start-gpu.bat 並選擇自動下載。",
        )

    def _load_models(self):
        device = self._get_torch_device()
        try:
            self.engine.load(device=device, model_dir=GPU_MODEL_DIR, cb=self._set_status)
            self.after(0, self._on_models_ready)
        except Exception as e:
            first_line = str(e).splitlines()[0][:140]
            self.after(0, lambda d=device, r=first_line: self._on_models_failed(d, r))

    def _on_models_ready(self):
        self.device_combo.configure(state="readonly")
        self.reload_btn.configure(state="normal")
        self.convert_btn.configure(state="normal")
        self.rt_start_btn.configure(state="normal")
        self.lang_combo.configure(state="readonly")
        device_label = self.device_var.get()
        self._set_status(f"✅ 就緒（{device_label}）")
        if self.engine.diar_engine and self.engine.diar_engine.ready:
            self.diarize_chk.configure(state="normal")

    def _on_models_failed(self, device: str, reason: str):
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
        self.engine.ready = False
        self.convert_btn.configure(state="disabled")
        self.rt_start_btn.configure(state="disabled")
        self.reload_btn.configure(state="disabled")
        threading.Thread(target=self._load_models, daemon=True).start()

    def _set_status(self, msg: str):
        self.after(0, lambda: self.status_dot.configure(text=msg))

    # ── 說話者分離 UI ──────────────────────────────────

    def _on_diarize_toggle(self):
        state = "readonly" if self._diarize_var.get() else "disabled"
        self.n_spk_combo.configure(state=state)

    # ── Hint 輸入輔助 ──────────────────────────────────

    def _bind_ctx_menu(self, native_widget, is_text: bool = False):
        def show(event):
            menu = tk.Menu(self, tearoff=0)
            menu.add_command(label="貼上",
                             command=lambda: native_widget.event_generate("<<Paste>>"))
            if is_text:
                menu.add_command(label="全選",
                                 command=lambda: native_widget.tag_add("sel", "1.0", "end"))
                menu.add_separator()
                menu.add_command(label="清除全部",
                                 command=lambda: native_widget.delete("1.0", "end"))
            else:
                menu.add_command(label="全選",
                                 command=lambda: native_widget.select_range(0, "end"))
                menu.add_separator()
                menu.add_command(label="清除全部",
                                 command=lambda: native_widget.delete(0, "end"))
            menu.tk_popup(event.x_root, event.y_root)
        native_widget.bind("<Button-3>", show)

    def _load_hint_txt(self, target, is_textbox: bool = True):
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
                messagebox.showerror("讀取失敗", str(e)); return
        if is_textbox:
            target.delete("1.0", "end"); target.insert("1.0", text)
        else:
            target.delete(0, "end"); target.insert(0, text)

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
                default      = sd.default.device[0]
                default_name = next(
                    (k for k, v in self._dev_idx_map.items() if v == default), choices[0]
                )
                self.rt_dev_combo.set(default_name)
        except ImportError:
            self.rt_dev_combo.configure(values=["（需安裝 sounddevice）"])

    # ── 音檔轉換 ───────────────────────────────────────

    def _on_browse(self):
        path = filedialog.askopenfilename(
            title="選擇音訊檔案",
            filetypes=[("音訊檔案", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac"),
                       ("所有檔案", "*.*")],
        )
        if path:
            self._audio_file = Path(path)
            self.file_entry.delete(0, "end")
            self.file_entry.insert(0, str(self._audio_file))
            if self.engine.ready:
                self.convert_btn.configure(state="normal")

    def _on_convert(self):
        if self._converting:
            return
        path = Path(self.file_entry.get().strip())
        if not path.exists():
            messagebox.showwarning("提示", "找不到檔案，請重新選擇"); return
        if not self.engine.ready:
            messagebox.showwarning("提示", "模型尚未載入完成"); return

        self._audio_file = path
        lang_sel = self.lang_var.get()
        self._selected_language = lang_sel if lang_sel != "自動偵測" else None
        hint_text = self.hint_box.get("1.0", "end").strip()
        self._file_hint       = hint_text if hint_text else None
        self._file_diarize    = self._diarize_var.get()
        n_spk_sel             = self.n_spk_combo.get()
        self._file_n_speakers = int(n_spk_sel) if n_spk_sel.isdigit() else None

        self._converting = True
        self.convert_btn.configure(state="disabled", text="轉換中…")
        self.prog_bar.set(0)
        self._file_log_clear()
        threading.Thread(target=self._convert_worker, daemon=True).start()

    def _convert_worker(self):
        path       = self._audio_file
        language   = self._selected_language
        context    = self._file_hint
        diarize    = getattr(self, "_file_diarize", False)
        n_speakers = getattr(self, "_file_n_speakers", None)

        def prog_cb(done, total, msg):
            pct = done / total if total > 0 else 0
            self.after(0, lambda: self.prog_bar.set(pct))
            self.after(0, lambda: self.prog_label.configure(text=msg))
            self._file_log(msg)

        try:
            t0        = time.perf_counter()
            lang_info = f"  語系：{language or '自動'}"
            hint_info = (f"  提示：{context[:30]}…" if context and len(context) > 30
                         else (f"  提示：{context}" if context else ""))
            diar_info = (f"  [說話者分離，人數：{n_speakers or '自動'}]"
                         if diarize else "")
            self._file_log(f"開始處理：{path.name}{lang_info}{hint_info}{diar_info}")
            srt = self.engine.process_file(
                path, progress_cb=prog_cb, language=language,
                context=context, diarize=diarize, n_speakers=n_speakers,
            )
            elapsed = time.perf_counter() - t0

            if srt:
                self._srt_output = srt
                self._file_log(f"\n✅ 完成！耗時 {elapsed:.1f}s")
                self._file_log(f"SRT 儲存至：{srt}")
                self.after(0, lambda: [
                    self.prog_bar.set(1.0),
                    self.open_dir_btn.configure(state="normal"),
                    self.prog_label.configure(text="完成"),
                ])
            else:
                self._file_log("⚠ 未偵測到人聲，未產生字幕")
                self.after(0, lambda: self.prog_bar.set(0))
        except Exception as e:
            self._file_log(f"❌ 錯誤：{e}")
            self.after(0, lambda: self.prog_bar.set(0))
        finally:
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

    # ── 即時轉換 ───────────────────────────────────────

    def _on_rt_start(self):
        name = self.rt_dev_combo.get()
        idx  = self._dev_idx_map.get(name)
        if idx is None:
            messagebox.showwarning("提示", "請選擇有效的音訊輸入裝置"); return

        lang_sel = self.lang_var.get()
        rt_lang  = lang_sel if lang_sel != "自動偵測" else None
        rt_hint  = self.rt_hint_entry.get().strip() or None

        self._rt_mgr = RealtimeManager(
            asr=self.engine, device_idx=idx,
            on_text=self._on_rt_text, on_status=self._on_rt_status,
            language=rt_lang, context=rt_hint,
        )
        try:
            self._rt_mgr.start()
        except Exception as e:
            messagebox.showerror("錯誤", f"無法開啟音訊裝置：{e}")
            self._rt_mgr = None; return

        self.rt_start_btn.configure(state="disabled")
        self.rt_stop_btn.configure(state="normal")

    def _on_rt_stop(self):
        if self._rt_mgr:
            self._rt_mgr.stop(); self._rt_mgr = None
        self.rt_start_btn.configure(state="normal")
        self.rt_stop_btn.configure(state="disabled")

    def _on_rt_text(self, text: str):
        self._rt_log.append(text)
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
            messagebox.showinfo("提示", "目前沒有字幕內容可儲存"); return
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = SRT_DIR / f"realtime_{ts}.srt"
        t   = 0.0
        with open(out, "w", encoding="utf-8") as f:
            for idx, line in enumerate(self._rt_log, 1):
                end = t + 5.0
                f.write(f"{idx}\n{_srt_ts(t)} --> {_srt_ts(end)}\n{line}\n\n")
                t = end + 0.1
        messagebox.showinfo("儲存完成", f"已儲存至：\n{out}")
        os.startfile(str(SRT_DIR))

    # ── 關閉處理 ───────────────────────────────────────

    def _on_close(self):
        if self._converting:
            if not messagebox.askyesno(
                "確認關閉", "音訊轉換正在進行中。\n確定要強制關閉嗎？",
                icon="warning", default="no",
            ):
                return
        if self._rt_mgr:
            try: self._rt_mgr.stop()
            except Exception: pass
        self.destroy()
        os._exit(0)


# ══════════════════════════════════════════════════════

if __name__ == "__main__":
    app = App()
    app.mainloop()
