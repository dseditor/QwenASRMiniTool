"""
Qwen3 ASR 字幕生成器 - CustomTkinter 前端

功能：
  1. 音檔上傳 → SRT 字幕（支援 OpenVINO CPU / GPU）
  2. 即時轉換：偵測音訊輸入裝置，邊說邊顯示字幕
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
import queue
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import numpy as np
import customtkinter as ctk

# ── 路徑 ──────────────────────────────────────────────
# PyInstaller 凍結時，模型應放在 EXE 旁邊（非 _internal/）
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent
else:
    BASE_DIR = Path(__file__).parent
_DEFAULT_MODEL_DIR = BASE_DIR / "ov_models"
SETTINGS_FILE      = BASE_DIR / "settings.json"
SRT_DIR            = BASE_DIR / "subtitles"
SRT_DIR.mkdir(exist_ok=True)

# ── 常數 ──────────────────────────────────────────────
SAMPLE_RATE   = 16000
VAD_CHUNK     = 512
VAD_THRESHOLD = 0.5
MAX_GROUP_SEC = 20
MAX_CHARS     = 20
MIN_SUB_SEC   = 0.6
GAP_SEC       = 0.08

RT_SILENCE_CHUNKS    = 25   # ~0.8s 靜音後觸發轉錄
RT_MAX_BUFFER_CHUNKS = 600  # ~19s 上限強制轉錄


# ══════════════════════════════════════════════════════
# 共用工具函式
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
    """以標點符號切分短句，移除標點，每句獨立成行。"""
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
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
# ASR 引擎
# ══════════════════════════════════════════════════════

class ASREngine:
    """封裝所有模型。transcribe() 加互斥鎖，多執行緒安全。"""

    def __init__(self):
        self.ready     = False
        self._lock     = threading.Lock()
        self.vad_sess  = None
        self.audio_enc = None
        self.embedder  = None
        self.dec_req   = None
        self.processor = None   # LightProcessor（不含 torch）
        self.pad_id    = None
        self.cc        = None

    def load(self, device: str = "CPU", model_dir: Path = None, cb=None):
        """從背景執行緒呼叫。cb(msg) 用於更新 UI 狀態。"""
        import onnxruntime as ort
        import openvino as ov
        import opencc
        from processor_numpy import LightProcessor

        if model_dir is None:
            model_dir = _DEFAULT_MODEL_DIR
        ov_dir   = model_dir / "qwen3_asr_int8"
        vad_path = model_dir / "silero_vad_v4.onnx"

        def _s(msg):
            if cb: cb(msg)

        _s("載入 VAD 模型…")
        self.vad_sess = ort.InferenceSession(
            str(vad_path), providers=["CPUExecutionProvider"]
        )

        _s(f"編譯 ASR 模型（{device}）…")
        core = ov.Core()
        self.audio_enc = core.compile_model(str(ov_dir / "audio_encoder_model.xml"),      device)
        self.embedder  = core.compile_model(str(ov_dir / "thinker_embeddings_model.xml"), device)
        dec_comp       = core.compile_model(str(ov_dir / "decoder_model.xml"),            device)
        self.dec_req   = dec_comp.create_infer_request()

        _s("載入 Processor（純 numpy）…")
        self.processor = LightProcessor(ov_dir)
        self.pad_id    = self.processor.pad_id
        self.cc        = opencc.OpenCC("s2twp")
        self.ready     = True
        _s(f"編譯完成（{device}）")

    def transcribe(self, audio: np.ndarray, max_tokens: int = 300) -> str:
        """將 16kHz float32 音訊轉錄為繁體中文。"""
        with self._lock:
            # ── 前處理（純 numpy，不需 torch）────────────────────────
            mel, ids = self.processor.prepare(audio)

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
            return self.cc.convert(raw.strip())

    def process_file(
        self, audio_path: Path, progress_cb=None
    ) -> Path | None:
        """音檔 → SRT，回傳 SRT 路徑。"""
        import librosa
        audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        groups = _detect_speech_groups(audio, self.vad_sess)
        if not groups:
            return None

        all_subs: list[tuple[float, float, str]] = []
        for i, (g0, g1, chunk) in enumerate(groups):
            if progress_cb:
                progress_cb(i, len(groups), f"[{i+1}/{len(groups)}] {g0:.1f}s ~ {g1:.1f}s")
            text = self.transcribe(chunk)
            if not text:
                continue
            lines = _split_to_lines(text)
            all_subs.extend(_assign_ts(lines, g0, g1))

        if progress_cb:
            progress_cb(len(groups), len(groups), "寫入 SRT…")

        out = SRT_DIR / (audio_path.stem + ".srt")
        with open(out, "w", encoding="utf-8") as f:
            for idx, (s, e, line) in enumerate(all_subs, 1):
                f.write(f"{idx}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{line}\n\n")
        return out


# ══════════════════════════════════════════════════════
# 即時轉錄管理員
# ══════════════════════════════════════════════════════

class RealtimeManager:
    """sounddevice 串流 + VAD + 緩衝轉錄。"""

    def __init__(self, asr: ASREngine, device_idx: int, on_text, on_status):
        self.asr       = asr
        self.dev_idx   = device_idx
        self.on_text   = on_text    # callback(text: str)
        self.on_status = on_status  # callback(msg: str)
        self._q        = queue.Queue()
        self._running  = False
        self._stream   = None

    def start(self):
        import sounddevice as sd
        self._running = True
        self._stream  = sd.InputStream(
            device=self.dev_idx,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=VAD_CHUNK,
            dtype="float32",
            callback=self._audio_cb,
        )
        threading.Thread(target=self._loop, daemon=True).start()
        self._stream.start()
        self.on_status("🔴 錄音中…")

    def stop(self):
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
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
                        text = self.asr.transcribe(audio[:n])
                        if text:
                            self.on_text(text)
                    except Exception:
                        pass
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
        self.title("Qwen3 ASR 字幕生成器")
        self.geometry("960x700")
        self.minsize(800, 580)

        self.engine       = ASREngine()
        self._rt_mgr: RealtimeManager | None = None
        self._rt_log: list[str]              = []
        self._audio_file: Path | None        = None
        self._srt_output: Path | None        = None
        self._converting                     = False
        self._dev_idx_map: dict[str, int]    = {}
        self._model_dir: Path | None         = None   # 使用者選定的模型路徑

        self._build_ui()
        self._detect_ov_devices()
        self._refresh_audio_devices()   # 音訊裝置獨立初始化，不依賴模型載入
        threading.Thread(target=self._startup_check, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI 建構 ────────────────────────────────────────

    def _build_ui(self):
        # 標題列
        title_bar = ctk.CTkFrame(self, height=54, corner_radius=0)
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)
        ctk.CTkLabel(
            title_bar, text="  🎙 Qwen3 ASR 字幕生成器",
            font=FONT_TITLE, anchor="w"
        ).pack(side="left", padx=16, pady=8)

        # 裝置選擇列
        dev_bar = ctk.CTkFrame(self, height=46)
        dev_bar.pack(fill="x", padx=10, pady=(6, 0))
        dev_bar.pack_propagate(False)

        ctk.CTkLabel(dev_bar, text="推理裝置：", font=FONT_BODY).pack(
            side="left", padx=(14, 4), pady=12
        )
        self.device_var   = ctk.StringVar(value="CPU")
        self.device_combo = ctk.CTkComboBox(
            dev_bar, values=["CPU"], variable=self.device_var,
            width=110, state="disabled", font=FONT_BODY,
        )
        self.device_combo.pack(side="left", pady=12)

        self.reload_btn = ctk.CTkButton(
            dev_bar, text="重新載入", width=90, state="disabled",
            font=FONT_BODY, fg_color="gray35", hover_color="gray25",
            command=self._on_reload_models,
        )
        self.reload_btn.pack(side="left", padx=8, pady=12)

        self.status_dot = ctk.CTkLabel(
            dev_bar, text="⏳ 啟動中…",
            font=FONT_BODY, text_color="#AAAAAA", anchor="w"
        )
        self.status_dot.pack(side="left", padx=12, pady=12)

        # 下載進度條（正常情況下隱藏）
        self.dl_bar = ctk.CTkProgressBar(dev_bar, width=200, height=12)
        self.dl_bar.set(0)
        # 啟動時不 pack，由 _show_dl_bar / _hide_dl_bar 控制

        # 分頁
        self.tabs = ctk.CTkTabview(self, anchor="nw")
        self.tabs.pack(fill="both", expand=True, padx=10, pady=(8, 10))
        self.tabs.add("  音檔轉字幕  ")
        self.tabs.add("  即時轉換  ")

        self._build_file_tab(self.tabs.tab("  音檔轉字幕  "))
        self._build_rt_tab(self.tabs.tab("  即時轉換  "))

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

        # 進度
        prog_frame = ctk.CTkFrame(parent, fg_color="transparent")
        prog_frame.pack(fill="x", padx=8, pady=(6, 2))

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

    # ── 即時轉換 tab ───────────────────────────────────

    def _build_rt_tab(self, parent):
        # 裝置選擇列
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
            parent, text="即時字幕", font=FONT_BODY,
            text_color="#AAAAAA", anchor="w",
        ).pack(fill="x", padx=8, pady=(8, 2))

        self.rt_textbox = ctk.CTkTextbox(
            parent, font=("Microsoft JhengHei", 15), state="disabled",
        )
        self.rt_textbox.pack(fill="both", expand=True, padx=8, pady=(0, 6))

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

    # ── 模型載入 ───────────────────────────────────────

    def _detect_ov_devices(self):
        """
        固定使用 CPU。
        OpenVINO GPU 外掛針對 Intel GPU 優化，NVIDIA OpenCL 不相容。
        如需 Intel iGPU 支援，請安裝 Intel GPU 驅動後修改此處。
        """
        self.device_combo.configure(values=["CPU"], state="readonly")
        self.device_var.set("CPU")

    # ── 設定檔讀寫（記住模型路徑）──────────────────────────────────────

    def _load_settings(self) -> dict:
        try:
            if SETTINGS_FILE.exists():
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_settings(self, model_dir: Path):
        try:
            data = {"model_dir": str(model_dir)}
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

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

    # ── 啟動檢查：模型完整性 → 必要時下載 → 載入模型 ─────────────────

    def _startup_check(self):
        """背景執行緒：確認模型路徑 → 下載（若需要）→ 載入。"""
        from downloader import quick_check, download_all

        # 1. 解析模型路徑
        model_dir = self._resolve_model_dir()
        if model_dir is None:
            chosen = [None]
            evt = threading.Event()
            self.after(0, lambda: self._show_model_path_dialog(chosen, evt))
            evt.wait()
            if chosen[0] is None:
                self.after(0, lambda: self._set_status("⚠ 已取消，模型未載入"))
                return
            model_dir = chosen[0]
            self._save_settings(model_dir)

        self._model_dir = model_dir

        # 2. 下載缺少的模型
        if not quick_check(model_dir):
            self.after(0, self._show_dl_bar)
            self._set_status("⬇ 下載模型中…")
            try:
                download_all(model_dir, progress_cb=self._on_dl_progress)
            except Exception as e:
                msg = str(e)
                self.after(0, self._hide_dl_bar)
                self.after(0, lambda: messagebox.showerror(
                    "下載失敗",
                    f"模型下載失敗：\n{msg}\n\n"
                    "請確認網路連線後重新啟動程式。"
                ))
                self.after(0, lambda: self._set_status("❌ 下載失敗"))
                return
            self.after(0, self._hide_dl_bar)

        # 3. 載入模型
        self._set_status("⏳ 模型載入中…")
        self._load_models()

    def _show_model_path_dialog(self, chosen: list, evt: threading.Event):
        """主執行緒：顯示模型路徑選擇對話框。"""
        dlg = ctk.CTkToplevel(self)
        dlg.title("選擇模型存放路徑")
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.focus_set()

        self.update_idletasks()
        x = self.winfo_x() + (self.winfo_width()  - 480) // 2
        y = self.winfo_y() + (self.winfo_height() - 230) // 2
        dlg.geometry(f"480x230+{x}+{y}")

        ctk.CTkLabel(
            dlg,
            text="找不到 Qwen3 ASR 模型\n請選擇模型的存放資料夾（首次將自動下載，約 1.2 GB）",
            justify="left",
        ).pack(anchor="w", padx=20, pady=(18, 8))

        # 優先顯示上次記住的路徑，否則預設
        _saved = self._load_settings().get("model_dir")
        path_var = ctk.StringVar(value=_saved if _saved else str(_DEFAULT_MODEL_DIR))

        row = ctk.CTkFrame(dlg, fg_color="transparent")
        row.pack(fill="x", padx=20)

        entry = ctk.CTkEntry(row, textvariable=path_var, width=340)
        entry.pack(side="left", fill="x", expand=True)

        def _browse():
            d = filedialog.askdirectory(title="選擇模型存放資料夾", parent=dlg)
            if d:
                path_var.set(d)

        ctk.CTkButton(row, text="瀏覽…", width=72, command=_browse).pack(side="left", padx=(6, 0))

        ctk.CTkLabel(
            dlg,
            text="若所選資料夾已有模型檔案，將直接使用，不會重複下載。",
            font=ctk.CTkFont(size=11),
            text_color="gray",
        ).pack(anchor="w", padx=20, pady=(6, 0))

        btn_row = ctk.CTkFrame(dlg, fg_color="transparent")
        btn_row.pack(pady=(14, 0))

        def _confirm():
            val = path_var.get().strip()
            chosen[0] = Path(val) if val else None
            dlg.destroy()
            evt.set()

        def _cancel():
            chosen[0] = None
            dlg.destroy()
            evt.set()

        ctk.CTkButton(btn_row, text="確認並繼續", width=120, command=_confirm).pack(side="left", padx=8)
        ctk.CTkButton(btn_row, text="取消", width=80, fg_color="#555", command=_cancel).pack(side="left", padx=8)
        dlg.protocol("WM_DELETE_WINDOW", _cancel)

    def _on_dl_progress(self, pct: float, msg: str):
        self.after(0, lambda: self.dl_bar.set(pct))
        self.after(0, lambda: self._set_status(f"⬇ {msg} ({pct*100:.0f}%)"))

    def _show_dl_bar(self):
        self.dl_bar.pack(side="left", padx=(0, 8), pady=12)

    def _hide_dl_bar(self):
        self.dl_bar.pack_forget()

    def _load_models(self):
        device = self.device_var.get()
        try:
            self.engine.load(device=device, model_dir=self._model_dir, cb=self._set_status)
            self.after(0, self._on_models_ready)
        except Exception as e:
            # 取得簡短錯誤訊息（OpenVINO 錯誤通常很長）
            first_line = str(e).splitlines()[0][:120]
            self.after(0, lambda d=device, r=first_line: self._on_models_failed(d, r))

    def _on_models_ready(self):
        self.device_combo.configure(state="readonly")
        self.reload_btn.configure(state="normal")
        self.convert_btn.configure(state="normal")
        self.rt_start_btn.configure(state="normal")
        device = self.device_var.get()
        self._set_status(f"✅ 就緒（{device}）")

    def _on_models_failed(self, device: str, reason: str):
        """模型載入失敗：還原 UI，讓使用者可以切換裝置後重試。"""
        self.device_combo.configure(state="readonly")
        self.reload_btn.configure(state="normal")   # 允許切換裝置後重試
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
            title="選擇音訊檔案",
            filetypes=[
                ("音訊檔案", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac"),
                ("所有檔案", "*.*"),
            ],
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
            messagebox.showwarning("提示", "找不到檔案，請重新選擇")
            return
        if not self.engine.ready:
            messagebox.showwarning("提示", "模型尚未載入完成")
            return

        self._audio_file = path
        self._converting = True
        self.convert_btn.configure(state="disabled", text="轉換中…")
        self.prog_bar.set(0)
        self._file_log_clear()
        threading.Thread(target=self._convert_worker, daemon=True).start()

    def _convert_worker(self):
        path = self._audio_file

        def prog_cb(done, total, msg):
            pct = done / total if total > 0 else 0
            self.after(0, lambda: self.prog_bar.set(pct))
            self.after(0, lambda: self.prog_label.configure(text=msg))
            self._file_log(msg)

        try:
            t0 = time.perf_counter()
            self._file_log(f"開始處理：{path.name}")
            srt     = self.engine.process_file(path, progress_cb=prog_cb)
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

    # ── 即時轉換操作 ───────────────────────────────────

    def _on_rt_start(self):
        name = self.rt_dev_combo.get()
        idx  = self._dev_idx_map.get(name)
        if idx is None:
            messagebox.showwarning("提示", "請選擇有效的音訊輸入裝置")
            return

        self._rt_mgr = RealtimeManager(
            asr=self.engine,
            device_idx=idx,
            on_text=self._on_rt_text,
            on_status=self._on_rt_status,
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
            messagebox.showinfo("提示", "目前沒有字幕內容可儲存")
            return
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
        # 轉換進行中：請使用者確認
        if self._converting:
            if not messagebox.askyesno(
                "確認關閉",
                "音訊轉換正在進行中。\n確定要強制關閉嗎？（目前進度將遺失）",
                icon="warning",
                default="no",
            ):
                return

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
    app = App()
    app.mainloop()
