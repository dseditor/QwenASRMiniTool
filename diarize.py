"""
diarize.py
─────────────────────────────────────────────────────────────────────
說話者分離（Speaker Diarization）引擎

兩個 ONNX 模型（共 32.5 MB，位於 ov_models/diarization/）：
  segmentation-community-1.onnx  — 偵測語音段落 + 粗略說話者類別
  embedding_model.onnx           — 提取說話者聲紋向量（WeSpeaker）

聚類演算法：
  兩階段設計 — 先收集所有段落的聲紋向量，再一次做全局聚類。
  避免貪心算法「鏈式漂移」導致同一說話者被拆成多個 ID 的問題。

  ┌─ 自動模式（n_speakers=None）
  │   Average-linkage 層次聚類，在 cosine distance 0.38 處切割樹狀圖
  │   → 自動決定說話者數量
  └─ 指定人數（n_speakers=N）
      強制分成 N 組（maxclust criterion）
      → 適合已知說話者數量的情境，避免過度分割

使用方式：
  from diarize import DiarizationEngine
  eng = DiarizationEngine(model_dir / "diarization")
  segments = eng.diarize(audio_float32_16khz, n_speakers=2)
  # → [(0.40, 4.55, "說話者1"), (4.85, 9.28, "說話者2"), ...]
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort

# ── 常數（從 pyannote-rs 原始碼 segment.rs 取得）─────────────────
SAMPLE_RATE    = 16_000
WINDOW_SAMPLES = SAMPLE_RATE * 10          # 160,000 samples = 10 秒
FRAME_SIZE     = 270                       # samples per output frame
FRAME_START    = 721                       # initial sample offset
MIN_SEG_SEC    = 0.8                       # 過短段落過濾門檻（秒）
MERGE_GAP_SEC  = 0.30                      # 相鄰同說話者合併間距（秒）

# 自動模式的 cosine distance 切割閾值
# cosine_dist = 1 - cosine_sim；值越小 → 聚類越嚴格（說話者數越多）
AUTO_DIST_THRESH = 0.38


class DiarizationEngine:
    """
    說話者分離引擎（thread-safe）。

    屬性：
        ready : bool  — 模型已載入且可用
    """

    def __init__(self, diar_dir: Path):
        self._lock    = threading.Lock()
        self.ready    = False
        self.seg_sess = None
        self.emb_sess = None
        self._diar_dir = diar_dir
        self._load()

    # ── 模型載入 ──────────────────────────────────────────────────

    def _load(self):
        seg_path = self._diar_dir / "segmentation-community-1.onnx"
        emb_path = self._diar_dir / "embedding_model.onnx"
        if not seg_path.exists() or not emb_path.exists():
            return   # 靜默失敗，self.ready 保持 False

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 2
        opts.log_severity_level   = 3   # 隱藏 ONNX Runtime 警告
        self.seg_sess = ort.InferenceSession(
            str(seg_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.emb_sess = ort.InferenceSession(
            str(emb_path), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.ready = True

    # ── 公開 API ─────────────────────────────────────────────────

    def diarize(
        self,
        audio: np.ndarray,
        n_speakers: int | None = None,
    ) -> list[tuple[float, float, str]]:
        """
        對 16kHz float32 音訊執行說話者分離。

        n_speakers : 指定說話者總人數（None = 自動偵測）。
                     已知人數時強烈建議指定，可避免過度分割。

        回傳：[(start_sec, end_sec, "說話者N"), ...]
        已過濾靜音與過短段落，可直接作為 ASR 的分段依據。
        """
        with self._lock:
            raw  = self._segment(audio)
            return self._embed_and_cluster(audio, raw, n_speakers=n_speakers)

    # ── 分割（Segmentation Model）────────────────────────────────

    def _segment(
        self, audio: np.ndarray
    ) -> list[tuple[float, float, int]]:
        """
        回傳：[(start_sec, end_sec, local_class), ...]
        local_class: 1-6（非 0 靜音）
        """
        input_name = self.seg_sess.get_inputs()[0].name   # "input_values"

        n   = len(audio)
        pad = (WINDOW_SAMPLES - n % WINDOW_SAMPLES) % WINDOW_SAMPLES
        padded = np.pad(audio.astype(np.float32), (0, pad))

        raw_segs: list[tuple[float, float, int]] = []

        for win_start in range(0, len(padded), WINDOW_SAMPLES):
            window = padded[win_start: win_start + WINDOW_SAMPLES]
            inp    = window[np.newaxis, np.newaxis, :]
            logits = self.seg_sess.run(None, {input_name: inp})[0]
            frame_labels = np.argmax(logits[0], axis=-1)

            def _frame_to_sec(fi: int, _ws: int = win_start) -> float:
                return (_ws + FRAME_START + fi * FRAME_SIZE) / SAMPLE_RATE

            cur_lbl   = int(frame_labels[0])
            seg_start = 0
            for fi in range(1, len(frame_labels)):
                lbl = int(frame_labels[fi])
                if lbl != cur_lbl:
                    if cur_lbl != 0:
                        raw_segs.append((_frame_to_sec(seg_start),
                                         _frame_to_sec(fi), cur_lbl))
                    seg_start = fi
                    cur_lbl   = lbl
            if cur_lbl != 0:
                raw_segs.append((_frame_to_sec(seg_start),
                                 _frame_to_sec(len(frame_labels)), cur_lbl))

        # 裁剪 + 過濾 + 合併相鄰同類段落
        total_dur = n / SAMPLE_RATE
        merged: list[tuple[float, float, int]] = []
        for t0, t1, lbl in raw_segs:
            t0 = min(t0, total_dur)
            t1 = min(t1, total_dur)
            if t1 - t0 < 0.1:
                continue
            if (merged and merged[-1][2] == lbl
                    and t0 - merged[-1][1] < MERGE_GAP_SEC):
                merged[-1] = (merged[-1][0], t1, lbl)
            else:
                merged.append((t0, t1, lbl))

        return merged

    # ── 嵌入提取（Embedding Model）──────────────────────────────

    def _kaldi_fbank(self, samples_f32: np.ndarray) -> np.ndarray:
        """Kaldi-style 80 維 mel filter bank（WeSpeaker 標準前處理）。"""
        import kaldi_native_fbank as knf

        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq       = float(SAMPLE_RATE)
        opts.frame_opts.frame_length_ms = 25.0
        opts.frame_opts.frame_shift_ms  = 10.0
        opts.mel_opts.num_bins          = 80
        opts.frame_opts.dither          = 0.0

        fbank = knf.OnlineFbank(opts)
        fbank.accept_waveform(
            float(SAMPLE_RATE),
            (samples_f32 * 32768.0).tolist(),
        )
        fbank.input_finished()

        n_frames = fbank.num_frames_ready
        if n_frames == 0:
            return np.zeros((1, 80), dtype=np.float32)

        feats = np.array(
            [fbank.get_frame(i) for i in range(n_frames)],
            dtype=np.float32,
        )
        feats -= feats.mean(axis=0, keepdims=True)   # CMN 正規化
        return feats

    def _get_embedding(
        self, audio: np.ndarray, t0: float, t1: float
    ) -> np.ndarray | None:
        """從 [t0, t1] 秒音訊提取 L2 正規化後的 256 維說話者向量。"""
        s0    = int(t0 * SAMPLE_RATE)
        s1    = int(t1 * SAMPLE_RATE)
        chunk = audio[s0:s1]
        if len(chunk) < SAMPLE_RATE * 0.5:
            return None

        feats = self._kaldi_fbank(chunk)
        out   = self.emb_sess.run(
            None, {"fbank_features": feats[np.newaxis, :]}
        )[0][0]
        norm = np.linalg.norm(out)
        return out / norm if norm > 1e-9 else out

    # ── 兩階段聚類 ───────────────────────────────────────────────

    def _embed_and_cluster(
        self,
        audio: np.ndarray,
        raw_segs: list[tuple[float, float, int]],
        n_speakers: int | None = None,
    ) -> list[tuple[float, float, str]]:
        """
        兩階段設計：
          Stage 1  — 對所有有效段落提取 embedding（先不分類）
          Stage 2  — 統一做層次聚類，避免貪心順序造成的漂移
        """
        # Stage 1：收集所有有效段落的 embedding
        segs_with_emb: list[tuple[float, float, np.ndarray]] = []
        for t0, t1, _ in raw_segs:
            if t1 - t0 < MIN_SEG_SEC:
                continue
            emb = self._get_embedding(audio, t0, t1)
            if emb is not None:
                segs_with_emb.append((t0, t1, emb))

        if not segs_with_emb:
            return []

        embeddings = [e for _, _, e in segs_with_emb]

        # Stage 2：聚類取得說話者標籤
        if len(embeddings) == 1:
            labels = [1]
        elif n_speakers is not None:
            labels = self._cluster_fixed_n(embeddings, n_speakers)
        else:
            labels = self._cluster_auto(embeddings)

        # 組合結果
        results = [
            (t0, t1, f"說話者{lbl}")
            for (t0, t1, _), lbl in zip(segs_with_emb, labels)
        ]

        # 合併相鄰的同說話者段落
        merged: list[tuple[float, float, str]] = []
        for t0, t1, spk in results:
            if (merged and merged[-1][2] == spk
                    and t0 - merged[-1][1] < MERGE_GAP_SEC):
                merged[-1] = (merged[-1][0], t1, spk)
            else:
                merged.append((t0, t1, spk))

        return merged

    def _cluster_fixed_n(
        self, embeddings: list[np.ndarray], n: int
    ) -> list[int]:
        """
        指定人數模式：Average-linkage 層次聚類，強制分成 n 組。
        回傳 1-indexed 標籤列表。
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        n = max(1, min(n, len(embeddings)))  # 限制在合理範圍
        if len(embeddings) <= n:
            return list(range(1, len(embeddings) + 1))

        emb_mat = np.stack(embeddings)           # (M, 256)
        sim     = emb_mat @ emb_mat.T            # cosine similarity
        dist    = np.clip(1.0 - sim, 0.0, 2.0)  # cosine distance
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)

        Z      = linkage(condensed, method="average")
        labels = fcluster(Z, t=n, criterion="maxclust")   # 1-indexed
        return labels.tolist()

    def _cluster_auto(
        self, embeddings: list[np.ndarray]
    ) -> list[int]:
        """
        自動模式：Average-linkage 層次聚類，在 AUTO_DIST_THRESH 處切割。
        回傳 1-indexed 標籤列表。
        """
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform

        if len(embeddings) < 2:
            return [1] * len(embeddings)

        emb_mat = np.stack(embeddings)
        sim     = emb_mat @ emb_mat.T
        dist    = np.clip(1.0 - sim, 0.0, 2.0)
        np.fill_diagonal(dist, 0.0)
        condensed = squareform(dist, checks=False)

        Z      = linkage(condensed, method="average")
        labels = fcluster(Z, t=AUTO_DIST_THRESH, criterion="distance")
        return labels.tolist()
