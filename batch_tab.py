"""batch_tab.py — 批次多音檔辨識頁籤

整合進 app.py 的 CTkTabview：
    from batch_tab import BatchTab
    tab = BatchTab(parent_frame, engine, open_subtitle_cb=lambda srt, audio, dz: ...)
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk

# ── 字型常數（與 app.py 相同）──────────────────────────────────────────
FONT_BODY  = ("Microsoft JhengHei", 13)
FONT_SMALL = ("Microsoft JhengHei", 11)
FONT_MONO  = ("Consolas", 12)

# ── 支援的音訊副檔名 ───────────────────────────────────────────────────
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".opus",
              ".wma", ".mp4", ".mkv", ".webm"}


# ══════════════════════════════════════════════════════════════════════
# 單一音檔的資料模型
# ══════════════════════════════════════════════════════════════════════

class BatchItem:
    """批次清單中代表單一音檔的資料容器。"""

    def __init__(self, path: Path):
        self.path      = path
        self.status    = "待處理"   # 待處理 / 辨識中 / 完成 / 失敗
        self.progress  = 0.0       # 0.0–1.0
        self.srt_path: Path | None = None
        self.error_msg = ""
        self.duration  = 0.0       # 秒（背景載入）

    @property
    def status_color(self):
        return {
            "待處理": ("gray35", "#555566"),
            "辨識中": ("#1A6DA0", "#5DADE2"),
            "完成":   ("#1E7A42", "#58D68D"),
            "失敗":   ("#AA3030", "#F1948A"),
        }.get(self.status, ("gray40", "#888899"))


# ══════════════════════════════════════════════════════════════════════
# 批次辨識頁籤（CTkFrame，嵌入主 App 的 CTkTabview）
# ══════════════════════════════════════════════════════════════════════

class BatchTab(ctk.CTkFrame):
    """批次多音檔辨識頁籤。

    參數
    ----
    engine          : ASREngine 或 ChatLLMASREngine 實例（可為 None，稍後注入）
    open_subtitle_cb: callable(srt_path, audio_path, diarize_mode)
                      用來開啟 SubtitleEditorWindow，由 App 傳入以避免循環 import
    """

    def __init__(self, parent, engine, open_subtitle_cb, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._engine         = engine
        self._open_subtitle  = open_subtitle_cb
        self._items: list[BatchItem] = []
        self._row_widgets: list[dict] = []   # 每行的 widget 參照

        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._futures:  list[concurrent.futures.Future] = []
        self._running   = False
        self._stop_evt  = threading.Event()
        self._out_dir:  Path | None = None   # 輸出目錄（None = 與音檔同目錄）

        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)
        self._build_toolbar()
        self._build_list()
        self._build_statusbar()

    # ── 工具列 ────────────────────────────────────────────────────────

    def _build_toolbar(self):
        bar = ctk.CTkFrame(self, fg_color=("gray88", "#181828"), corner_radius=8)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        # 新增 / 移除
        ctk.CTkButton(
            bar, text="⊕", width=38, height=32,
            fg_color="#1B4A1B", hover_color="#28602A",
            font=("Segoe UI Emoji", 16),
            command=self._add_files,
        ).pack(side="left", padx=(8, 2), pady=6)

        ctk.CTkButton(
            bar, text="⊖", width=38, height=32,
            fg_color="#4A1B1B", hover_color="#602828",
            font=("Segoe UI Emoji", 16),
            command=self._remove_idle,
        ).pack(side="left", padx=(2, 8), pady=6)

        _sep(bar)

        # 開始 / 停止
        self._start_btn = ctk.CTkButton(
            bar, text="▶ 開始辨識", width=110, height=32,
            fg_color="#1A3A5C", hover_color="#265A8A",
            font=FONT_SMALL,
            command=self._start_all,
        )
        self._start_btn.pack(side="left", padx=(8, 4), pady=6)

        self._stop_btn = ctk.CTkButton(
            bar, text="⏹ 停止", width=80, height=32,
            fg_color="#4A2A1A", hover_color="#6A3C24",
            font=FONT_SMALL,
            command=self._stop_all,
            state="disabled",
        )
        self._stop_btn.pack(side="left", padx=(0, 8), pady=6)

        _sep(bar)

        # 並行選項
        self._parallel_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(
            bar, text="並行辨識",
            variable=self._parallel_var,
            font=FONT_SMALL, text_color=("gray20", "#AAAACC"),
            command=self._on_parallel_toggle,
        ).pack(side="left", padx=(8, 4), pady=6)

        self._gpu_hint = ctk.CTkLabel(
            bar, text="（僅 GPU 後端）",
            font=("Microsoft JhengHei", 10), text_color=("gray50", "#665544"),
        )
        self._gpu_hint.pack(side="left", padx=(0, 4))

        ctk.CTkLabel(bar, text="線程:", font=FONT_SMALL,
                     text_color=("gray40", "#888899")).pack(side="left", padx=(4, 2))
        self._worker_var = ctk.StringVar(value="2")
        self._worker_cmb = ctk.CTkComboBox(
            bar, values=["1", "2", "3", "4"],
            variable=self._worker_var,
            width=56, height=28, font=FONT_SMALL,
            state="disabled",
        )
        self._worker_cmb.pack(side="left", padx=(0, 8), pady=6)

        _sep(bar)

        # 輸出目錄
        ctk.CTkButton(
            bar, text="📁 輸出目錄", width=100, height=32,
            fg_color="#282838", hover_color="#383850",
            font=FONT_SMALL,
            command=self._choose_out_dir,
        ).pack(side="left", padx=(8, 4), pady=6)

        self._out_dir_lbl = ctk.CTkLabel(
            bar, text="（音檔所在目錄）",
            font=("Microsoft JhengHei", 10), text_color=("gray45", "#444455"),
            anchor="w",
        )
        self._out_dir_lbl.pack(side="left", padx=(0, 8))

    # ── 清單區 ────────────────────────────────────────────────────────

    def _build_list(self):
        # 表頭
        hdr = ctk.CTkFrame(self, fg_color=("gray85", "#1E1E32"), corner_radius=0, height=26)
        hdr.grid(row=1, column=0, sticky="new", padx=8)
        hdr.pack_propagate(False)
        for txt, w in [
            ("  #", 32), ("檔名", 0), ("時長", 60),
            ("狀態", 68), ("進度", 136), ("操作", 112),
        ]:
            kw: dict = dict(
                text=txt, font=("Microsoft JhengHei", 11),
                text_color=("gray35", "#55556A"), anchor="w",
            )
            if w:
                kw["width"] = w
            ctk.CTkLabel(hdr, **kw).pack(side="left", padx=(4, 0))

        # 可捲動清單
        self._sf = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self._sf.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 4))

        self._rebuild_list()

    # ── 狀態列 ────────────────────────────────────────────────────────

    def _build_statusbar(self):
        bot = ctk.CTkFrame(self, fg_color=("gray90", "#14141E"), corner_radius=0, height=36)
        bot.grid(row=3, column=0, sticky="ew", padx=0)
        bot.grid_propagate(False)

        self._status_var = ctk.StringVar(value="尚無音檔")
        ctk.CTkLabel(
            bot, textvariable=self._status_var,
            font=FONT_SMALL, text_color=("gray35", "#555566"),
        ).pack(side="left", padx=12, pady=8)

        self._overall_bar = ctk.CTkProgressBar(bot, width=180, height=10)
        self._overall_bar.set(0)
        self._overall_bar.pack(side="left", padx=(0, 12), pady=8)

    # ── 重建清單 ──────────────────────────────────────────────────────

    def _rebuild_list(self):
        for w in self._sf.winfo_children():
            w.destroy()
        self._row_widgets.clear()
        for i, item in enumerate(self._items):
            self._build_row(i, item)

    def _build_row(self, idx: int, item: BatchItem):
        bg = ("gray95", "#1C1C1C") if idx % 2 == 0 else ("gray91", "#222228")
        fr = ctk.CTkFrame(self._sf, fg_color=bg, corner_radius=4)
        fr.pack(fill="x", padx=2, pady=1)
        fr.columnconfigure(1, weight=1)

        # 序號
        ctk.CTkLabel(
            fr, text=str(idx + 1), width=28, anchor="e",
            font=("Consolas", 11), text_color=("gray35", "#555566"),
        ).grid(row=0, column=0, padx=(6, 4), pady=5)

        # 檔名（可點擊工具提示用 tooltip 太複雜，改用完整路徑作 anchor title）
        name_lbl = ctk.CTkLabel(
            fr, text=item.path.name, anchor="w",
            font=FONT_SMALL, text_color=("gray20", "#BBBBCC"),
        )
        name_lbl.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        _tooltip(name_lbl, str(item.path))

        # 時長
        dur_lbl = ctk.CTkLabel(
            fr,
            text=_fmt_dur(item.duration) if item.duration else "—",
            width=56, anchor="center",
            font=FONT_MONO, text_color=("gray45", "#666677"),
        )
        dur_lbl.grid(row=0, column=2, padx=(0, 4))

        # 狀態
        status_lbl = ctk.CTkLabel(
            fr, text=item.status, width=64, anchor="center",
            font=FONT_SMALL, text_color=item.status_color,
        )
        status_lbl.grid(row=0, column=3, padx=(0, 4))

        # 進度條
        pbar = ctk.CTkProgressBar(fr, width=124, height=10)
        pbar.set(item.progress)
        pbar.grid(row=0, column=4, padx=(0, 6), pady=5)

        # 操作按鈕
        btn_fr = ctk.CTkFrame(fr, fg_color="transparent")
        btn_fr.grid(row=0, column=5, padx=(0, 6), pady=4)

        ctk.CTkButton(
            btn_fr, text="▶", width=32, height=26,
            fg_color="#1A3A5C", hover_color="#265A8A",
            font=("Segoe UI Emoji", 12),
            command=lambda p=item.path: self._preview(p),
        ).pack(side="left", padx=(0, 2))

        detail_state = "normal" if (item.srt_path and item.srt_path.exists()) else "disabled"
        detail_btn = ctk.CTkButton(
            btn_fr, text="⋯", width=32, height=26,
            fg_color="#2A1A4A", hover_color="#3D2870",
            font=("Segoe UI Emoji", 12),
            command=lambda it=item: self._open_detail(it),
            state=detail_state,
        )
        detail_btn.pack(side="left", padx=(0, 2))

        ctk.CTkButton(
            btn_fr, text="⊖", width=32, height=26,
            fg_color="#4A1B1B", hover_color="#602828",
            font=("Segoe UI Emoji", 12),
            command=lambda i=idx: self._remove_item(i),
        ).pack(side="left")

        self._row_widgets.append({
            "frame":      fr,
            "status_lbl": status_lbl,
            "pbar":       pbar,
            "dur_lbl":    dur_lbl,
            "detail_btn": detail_btn,
        })

    # ── 檔案管理 ──────────────────────────────────────────────────────

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            parent=self,
            title="選擇音檔（可多選）",
            filetypes=[
                ("音訊檔案",
                 "*.mp3 *.wav *.m4a *.flac *.ogg *.aac *.opus *.wma *.mp4 *.mkv *.webm"),
                ("所有檔案", "*.*"),
            ],
        )
        added = False
        for p in paths:
            pp = Path(p)
            if not any(it.path == pp for it in self._items):
                self._items.append(BatchItem(pp))
                added = True
        if added:
            self._rebuild_list()
            self._refresh_status()
            # 背景載入時長
            for item in self._items:
                if item.duration == 0.0:
                    threading.Thread(
                        target=self._load_dur, args=(item,), daemon=True,
                    ).start()

    def add_file(self, path: Path):
        """供 App 主視窗呼叫（如拖放 / 命令列），加入單一音檔。"""
        if not any(it.path == path for it in self._items):
            self._items.append(BatchItem(path))
            self._rebuild_list()
            self._refresh_status()
            threading.Thread(target=self._load_dur, args=(self._items[-1],),
                             daemon=True).start()

    def _remove_item(self, idx: int):
        if 0 <= idx < len(self._items):
            if self._items[idx].status == "辨識中":
                return  # 辨識中不可移除
            del self._items[idx]
            self._rebuild_list()
            self._refresh_status()

    def _remove_idle(self):
        """移除所有「待處理」與「失敗」的項目。"""
        self._items = [it for it in self._items
                       if it.status not in ("待處理", "失敗")]
        self._rebuild_list()
        self._refresh_status()

    def _load_dur(self, item: BatchItem):
        """背景執行緒：讀取音訊時長。"""
        try:
            import soundfile as sf
            info = sf.info(str(item.path))
            item.duration = info.duration
        except Exception:
            # soundfile 讀不了（m4a/影片等）→ audio_io（ffmpeg 後援，無 librosa）
            from audio_io import audio_duration
            item.duration = audio_duration(item.path)
            if not item.duration:
                return
        try:
            idx = self._items.index(item)
            self.after(0, lambda: self._row_widgets[idx]["dur_lbl"].configure(
                text=_fmt_dur(item.duration)
            ))
        except (ValueError, IndexError):
            pass

    # ── 辨識控制 ──────────────────────────────────────────────────────

    def _on_parallel_toggle(self):
        if self._parallel_var.get():
            self._worker_cmb.configure(state="normal")
        else:
            self._worker_cmb.configure(state="disabled")

    def _choose_out_dir(self):
        d = filedialog.askdirectory(parent=self, title="選擇 SRT 輸出目錄")
        if d:
            self._out_dir = Path(d)
            short = str(self._out_dir)
            if len(short) > 40:
                short = "…" + short[-38:]
            self._out_dir_lbl.configure(text=short)
        else:
            self._out_dir = None
            self._out_dir_lbl.configure(text="（音檔所在目錄）")

    def set_engine(self, engine):
        """模型載入完成後，由 App 注入引擎。"""
        self._engine = engine

    def _start_all(self):
        if self._engine is None:
            messagebox.showwarning("尚未載入模型", "請先完成模型載入再開始辨識。", parent=self)
            return

        pending = [it for it in self._items if it.status in ("待處理", "失敗")]
        if not pending:
            messagebox.showinfo("提示", "沒有待處理或失敗的音檔。", parent=self)
            return

        # 並行警告
        use_parallel = self._parallel_var.get()
        max_workers  = int(self._worker_var.get()) if use_parallel else 1
        backend_name = type(self._engine).__name__
        if use_parallel and max_workers > 1 and "ChatLLM" not in backend_name:
            ok = messagebox.askyesno(
                "並行辨識警告",
                f"目前後端為 {backend_name}，OpenVINO/CPU 後端不保證多線程安全。\n"
                "並行辨識可能導致結果錯誤或程式崩潰。\n\n"
                "建議只在 ChatLLM (GPU) 後端使用並行辨識。\n\n是否仍要繼續？",
                parent=self,
            )
            if not ok:
                return

        self._running = True
        self._stop_evt.clear()
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures.clear()

        for item in pending:
            if self._stop_evt.is_set():
                break
            f = self._executor.submit(self._run_one, item)
            f.add_done_callback(lambda _: self.after(0, self._check_all_done))
            self._futures.append(f)

    def _stop_all(self):
        self._stop_evt.set()
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        for item in self._items:
            if item.status == "辨識中":
                item.status   = "待處理"
                item.progress = 0.0
        self._running = False
        self.after(0, self._on_all_done)

    def _check_all_done(self):
        """future done callback：若所有 future 完成則恢復按鈕。"""
        if all(f.done() for f in self._futures):
            self._on_all_done()

    def _on_all_done(self):
        self._running = False
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._rebuild_list()
        self._refresh_status()

    def _run_one(self, item: BatchItem):
        """在 executor 執行緒中跑單一音檔辨識。"""
        if self._stop_evt.is_set():
            return

        item.status   = "辨識中"
        item.progress = 0.0
        self.after(0, lambda it=item: self._sync_row(it))

        # 真實 progress_cb（利用引擎的 chunk 回報）
        def _prog(i, total, _msg):
            if self._stop_evt.is_set():
                raise InterruptedError("使用者停止")
            item.progress = i / max(1, total)
            self.after(0, lambda it=item: self._update_pbar(it))

        import os, tempfile
        from ffmpeg_utils import is_video, find_ffmpeg, extract_audio_to_wav

        tmp_wav: "Path | None" = None
        try:
            proc_path = item.path

            # 影片音軌提取
            if is_video(item.path):
                ffmpeg = find_ffmpeg()
                if not ffmpeg:
                    raise RuntimeError(
                        "需要 ffmpeg 才能處理影片。\n"
                        "請在「音檔轉字幕」頁籤先完成 ffmpeg 下載，"
                        "或手動安裝 ffmpeg 並加入系統 PATH。"
                    )
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                os.close(tmp_fd)
                tmp_wav = Path(tmp_path)
                extract_audio_to_wav(item.path, tmp_wav, ffmpeg)
                proc_path = tmp_wav

            # 呼叫引擎
            srt = self._engine.process_file(
                proc_path,
                progress_cb=_prog,
                language=None,
                context=None,
            )

            # 若使用者指定了自訂輸出目錄，把 SRT 搬過去
            if srt and self._out_dir:
                dest = self._out_dir / srt.name
                try:
                    srt.rename(dest)
                    srt = dest
                except Exception:
                    pass
            item.srt_path = srt
            item.status   = "完成" if srt else "失敗"
            item.progress = 1.0 if srt else 0.0
            if not srt:
                item.error_msg = "未偵測到人聲，無輸出"
        except InterruptedError:
            item.status   = "待處理"
            item.progress = 0.0
        except Exception as e:
            item.status    = "失敗"
            item.error_msg = str(e)
            item.progress  = 0.0
        finally:
            if tmp_wav and tmp_wav.exists():
                try:
                    tmp_wav.unlink()
                except Exception:
                    pass

        self.after(0, lambda it=item: self._sync_row(it))
        self.after(0, self._refresh_status)

    # ── 列更新（必須在主執行緒呼叫）──────────────────────────────────

    def _sync_row(self, item: BatchItem):
        """同步單一行的狀態標籤 / 進度條 / 詳細按鈕。"""
        try:
            idx = self._items.index(item)
            w   = self._row_widgets[idx]
        except (ValueError, IndexError):
            return
        w["status_lbl"].configure(text=item.status, text_color=item.status_color)
        w["pbar"].set(item.progress)
        if item.srt_path and item.srt_path.exists():
            w["detail_btn"].configure(state="normal")
        # 失敗時顯示 tooltip
        if item.status == "失敗" and item.error_msg:
            _tooltip(w["status_lbl"], item.error_msg)

    def _update_pbar(self, item: BatchItem):
        try:
            idx = self._items.index(item)
            self._row_widgets[idx]["pbar"].set(item.progress)
        except (ValueError, IndexError):
            pass

    def _refresh_status(self):
        total  = len(self._items)
        done   = sum(1 for it in self._items if it.status == "完成")
        failed = sum(1 for it in self._items if it.status == "失敗")
        proc   = sum(1 for it in self._items if it.status == "辨識中")
        if total == 0:
            self._status_var.set("尚無音檔")
            self._overall_bar.set(0)
        else:
            parts = [f"完成 {done}/{total}"]
            if proc:
                parts.append(f"辨識中 {proc}")
            if failed:
                parts.append(f"失敗 {failed}")
            self._status_var.set("  ".join(parts))
            self._overall_bar.set(done / total)

    # ── 試聽與詳細 ────────────────────────────────────────────────────

    def _preview(self, path: Path):
        """試聽音檔前 30 秒。"""
        def _play():
            try:
                import soundfile as sf
                import sounddevice as sd
                sd.stop()
                data, sr = sf.read(str(path), always_2d=False, dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=1)
                sd.play(data[:sr * 30], sr)
            except Exception as e:
                self.after(0, lambda: messagebox.showerror(
                    "播放失敗", str(e), parent=self))
        threading.Thread(target=_play, daemon=True).start()

    def _open_detail(self, item: BatchItem):
        if item.srt_path and item.srt_path.exists():
            self._open_subtitle(item.srt_path, item.path, False)
        else:
            messagebox.showinfo(
                "尚未完成", "該音檔尚未辨識完成，無字幕可檢視。", parent=self)


# ── 輔助函式 ──────────────────────────────────────────────────────────

def _sep(parent):
    """在工具列插入分隔線。"""
    ctk.CTkFrame(parent, fg_color="#333344", width=1, height=28).pack(
        side="left", pady=6, padx=4)


def _fmt_dur(sec: float) -> str:
    """秒數格式化為 m:ss。"""
    m, s = int(sec // 60), int(sec % 60)
    return f"{m}:{s:02d}"


def _tooltip(widget: tk.BaseWidget, text: str):
    """最簡單的 hover tooltip（純 Tkinter）。"""
    tip: list[tk.Toplevel | None] = [None]

    def _enter(_):
        if not text:
            return
        x = widget.winfo_rootx() + 10
        y = widget.winfo_rooty() + widget.winfo_height() + 4
        t = tk.Toplevel(widget)
        t.wm_overrideredirect(True)
        t.wm_geometry(f"+{x}+{y}")
        tk.Label(t, text=text, bg="#2A2A3A", fg="#CCCCDD",
                 font=("Microsoft JhengHei", 10),
                 padx=6, pady=3, relief="solid", bd=1).pack()
        tip[0] = t

    def _leave(_):
        if tip[0]:
            tip[0].destroy()
            tip[0] = None

    widget.bind("<Enter>", _enter, add="+")
    widget.bind("<Leave>", _leave, add="+")
