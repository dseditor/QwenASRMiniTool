"""setting.py — 獨立設定 Tab

SettingsTab(CTkScrollableFrame) 整合：
  1. 版本與線上更新
  2. 外觀主題 / 介面縮放
  3. 辨識語系（全域）
  4. 中文輸出語言
  5. VAD 語音偵測閾值
  6. FFmpeg 路徑

（引擎 / 裝置 / 模型選擇、模型路徑、下載來源、CPU 效能已移至「模型」分頁 model_tab.py）

使用方式（app.py）：
    from setting import SettingsTab
    self._settings_tab = SettingsTab(self.tabs.tab("  設定  "), self)
    self._settings_tab.pack(fill="both", expand=True)

使用方式（app-gpu.py，語系於建立時預填完整清單）：
    self._settings_tab = SettingsTab(
        self.tabs.tab("  設定  "), self,
        lang_values=["自動偵測"] + SUPPORTED_LANGUAGES, lang_state="disabled")

對外 API：
    sync_prefs(settings: dict)  — 由 App._apply_ui_prefs 呼叫
"""
from __future__ import annotations

import os
import threading
import webbrowser
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

# ── 字型常數（與 app.py / app-gpu.py 保持一致）────────────────────────
FONT_BODY  = ("Microsoft JhengHei", 13)
FONT_SMALL = ("Microsoft JhengHei", 11)
FONT_MONO  = ("Consolas", 12)


def _hsep(parent):
    """水平分隔線。"""
    ctk.CTkFrame(
        parent, fg_color=("gray80", "gray25"), height=1, corner_radius=0,
    ).pack(fill="x", padx=0, pady=8)


# ══════════════════════════════════════════════════════════════════════
# SettingsTab
# ══════════════════════════════════════════════════════════════════════

class SettingsTab(ctk.CTkScrollableFrame):
    """設定頁籤：版本更新、外觀、縮放、辨識語系、中文輸出、VAD、FFmpeg。"""

    def __init__(self, parent, app, *,
                 lang_values: list[str] | None = None,
                 lang_state: str = "disabled"):
        super().__init__(parent, fg_color=("gray92", "gray17"))
        self._app           = app
        self._lang_values   = lang_values or ["自動偵測"]
        self._lang_state    = lang_state
        self._build()

    # ══ 建構 UI ══════════════════════════════════════════════════════

    def _build(self):
        # 辨識語系置頂（全域控制，最常切換）
        self._build_asr_language_section()
        _hsep(self)

        self._build_update_section()
        _hsep(self)

        self._build_appearance_section()
        _hsep(self)

        self._build_scale_section()
        _hsep(self)

        self._build_language_section()
        _hsep(self)

        self._build_vad_section()
        _hsep(self)

        self._build_ffmpeg_section()

    # ── 0. 版本與線上更新 ─────────────────────────────────────────────

    def _build_update_section(self):
        try:
            from version import __version__ as _ver
        except Exception:
            _ver = "?"
        self._app_version = _ver

        ctk.CTkLabel(
            self, text="🔄 版本與更新",
            font=("Microsoft JhengHei", 14, "bold"), anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 4))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 4))

        ctk.CTkLabel(
            row, text=f"目前版本：v{_ver}", font=FONT_BODY, anchor="w",
        ).pack(side="left")

        self._upd_check_btn = ctk.CTkButton(
            row, text="🔍 檢查更新", width=110, height=30, font=FONT_BODY,
            command=self._on_check_update,
        )
        self._upd_check_btn.pack(side="right")

        # 狀態文字（檢查結果 / 下載進度訊息）
        self._upd_status = ctk.CTkLabel(
            self, text="", font=FONT_SMALL,
            text_color=("gray40", "#AAAAAA"), anchor="w", justify="left",
            wraplength=520,
        )
        self._upd_status.pack(fill="x", padx=12, pady=(0, 2))

        # 進度條（下載時才顯示）
        self._upd_prog = ctk.CTkProgressBar(self, height=10)
        self._upd_prog.set(0)

        # 「立即更新」按鈕（發現新版時才顯示）
        self._upd_action_row = ctk.CTkFrame(self, fg_color="transparent")
        self._upd_do_btn = ctk.CTkButton(
            self._upd_action_row, text="⬇ 立即更新並重啟",
            width=170, height=32, font=FONT_BODY,
            fg_color=("#2563eb", "#1d4ed8"), hover_color=("#1d4ed8", "#1e40af"),
            command=self._on_do_update,
        )
        self._upd_do_btn.pack(side="left", padx=(0, 8))
        self._upd_page_btn = ctk.CTkButton(
            self._upd_action_row, text="🌐 發行頁",
            width=90, height=32, font=FONT_BODY,
            fg_color=("gray70", "gray35"), hover_color=("gray60", "gray28"),
            command=self._on_open_release_page,
        )
        self._upd_page_btn.pack(side="left")

        # 狀態暫存
        self._upd_info: dict | None = None

    def _set_upd_status(self, text: str, color=None):
        def _do():
            self._upd_status.configure(
                text=text,
                text_color=color or ("gray40", "#AAAAAA"),
            )
        try:
            self.after(0, _do)
        except Exception:
            pass

    def _on_open_release_page(self):
        url = (self._upd_info or {}).get("html_url") if self._upd_info else None
        if not url:
            try:
                from version import GITHUB_RELEASES_PAGE
                url = GITHUB_RELEASES_PAGE
            except Exception:
                return
        webbrowser.open(url)

    def _on_check_update(self):
        self._upd_check_btn.configure(state="disabled", text="檢查中…")
        self._upd_action_row.pack_forget()
        self._set_upd_status("正在向 GitHub 查詢最新版本…")

        def _worker():
            try:
                import updater
                info = updater.check_latest()
            except Exception as e:  # 連匯入都失敗（極少見）
                info = {"ok": False, "error": str(e)}
            self.after(0, lambda: self._after_check(info))

        threading.Thread(target=_worker, daemon=True).start()

    def _after_check(self, info: dict):
        self._upd_check_btn.configure(state="normal", text="🔍 檢查更新")
        self._upd_info = info

        if not info.get("ok"):
            self._set_upd_status(
                f"❌ 檢查失敗：{info.get('error', '未知錯誤')}",
                color=("#b91c1c", "#f87171"),
            )
            return

        latest = info.get("latest_tag", "?")
        if not info.get("has_update"):
            self._set_upd_status(
                f"✅ 已是最新版本（最新發佈：{latest}）",
                color=("#15803d", "#4ade80"),
            )
            return

        # 有更新
        body = (info.get("body") or "").strip()
        body_short = (body[:200] + "…") if len(body) > 200 else body
        msg = f"🎉 發現新版本 {latest}"
        if body_short:
            msg += f"\n更新說明：{body_short}"
        if not info.get("asset_url"):
            msg += "\n（此版本未附可自動更新的套件，請手動下載）"
        self._set_upd_status(msg, color=("#1d4ed8", "#93c5fd"))

        # 顯示動作列；無套件時禁用「立即更新」
        self._upd_action_row.pack(fill="x", padx=12, pady=(4, 6))
        if info.get("asset_url"):
            self._upd_do_btn.configure(state="normal")
        else:
            self._upd_do_btn.configure(state="disabled")

    def _on_do_update(self):
        import updater
        info = self._upd_info or {}
        url, name = info.get("asset_url"), info.get("asset_name")
        if not url or not name:
            return

        if not updater.is_frozen():
            self._set_upd_status(
                "⚠ 開發模式不支援自我更新，請改用 build.bat 重新編譯。",
                color=("#b45309", "#fbbf24"),
            )
            return

        self._upd_do_btn.configure(state="disabled", text="下載中…")
        self._upd_check_btn.configure(state="disabled")
        self._upd_prog.pack(fill="x", padx=12, pady=(2, 6))
        self._upd_prog.set(0)

        def _prog(done: int, total: int):
            if total:
                pct = done / total
                self.after(0, lambda: self._upd_prog.set(pct))
                self.after(0, lambda: self._set_upd_status(
                    f"下載中… {done/1_048_576:.1f} / {total/1_048_576:.1f} MB"))

        def _worker():
            try:
                archive = updater.download_asset(url, name, progress_cb=_prog)
                self._set_upd_status("下載完成，正在準備更新…")
                bat = updater.apply_update(archive, relaunch=True)
                self._set_upd_status("即將關閉並套用更新，請稍候…")
                updater.launch_and_exit(bat)
                # launch 後關閉本程式，讓 helper bat 完成覆蓋與重啟
                self.after(400, self._exit_for_update)
            except Exception as e:
                self.after(0, lambda: self._after_update_fail(str(e)))

        threading.Thread(target=_worker, daemon=True).start()

    def _after_update_fail(self, err: str):
        self._upd_prog.pack_forget()
        self._upd_do_btn.configure(state="normal", text="⬇ 立即更新並重啟")
        self._upd_check_btn.configure(state="normal")
        self._set_upd_status(
            f"❌ 更新失敗：{err}", color=("#b91c1c", "#f87171"))

    def _exit_for_update(self):
        """關閉應用程式，讓 helper bat 接手覆蓋檔案。"""
        try:
            self._app.destroy()
        finally:
            os._exit(0)

    # ── 2. 外觀主題 ───────────────────────────────────────────────────

    def _build_appearance_section(self):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(10, 8))

        ctk.CTkLabel(
            row, text="🎨 外觀主題", font=FONT_BODY, width=130, anchor="w",
        ).pack(side="left")

        self.appearance_seg = ctk.CTkSegmentedButton(
            row, values=["🌑 深色", "☀ 淺色"],
            width=160, height=30, font=FONT_BODY,
            command=self._on_appearance_seg,
        )
        self.appearance_seg.set("🌑 深色")
        self.appearance_seg.pack(side="left")

    def _on_appearance_seg(self, value: str):
        # 映射為 App._on_appearance_change 接受的值
        mapped = "☀" if "淺" in value else "🌑"
        self._app._on_appearance_change(mapped)

    # ── 2b. 介面縮放（字體大小）──────────────────────────────────────────

    # 顯示標籤 → 縮放倍率
    _SCALE_MAP = {
        "小 (90%)":    0.9,
        "標準 (100%)": 1.0,
        "大 (115%)":   1.15,
        "特大 (130%)": 1.3,
        "超大 (150%)": 1.5,
    }

    def _build_scale_section(self):
        ctk.CTkLabel(
            self, text="🔍 介面縮放（字體大小）", font=FONT_BODY, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        ctk.CTkLabel(
            self,
            text="等比放大整個介面與文字，適合高解析度螢幕。調整後立即生效。",
            font=FONT_SMALL, text_color=("gray40", "#AAAAAA"), anchor="w",
            wraplength=480, justify="left",
        ).pack(fill="x", padx=12, pady=(0, 4))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 8))

        self._scale_seg = ctk.CTkSegmentedButton(
            row, values=list(self._SCALE_MAP.keys()),
            height=30, font=FONT_BODY,
            command=self._on_scale_seg,
        )
        self._scale_seg.set("標準 (100%)")
        self._scale_seg.pack(side="left")

    def _on_scale_seg(self, value: str):
        scale = self._SCALE_MAP.get(value, 1.0)
        if hasattr(self._app, "_on_ui_scale_change"):
            self._app._on_ui_scale_change(scale)

    # ── 2c. 辨識語系（全域）─────────────────────────────────────────────

    def _build_asr_language_section(self):
        """ASR 強制語系選擇器（全域，音檔 / 批次 / 錄製共用）。

        widget 建立後回寫 app.lang_var / app.lang_combo，沿用既有
        _on_models_ready 對 lang_combo 的 values/state 填充邏輯。
        """
        ctk.CTkLabel(
            self, text="🌐 辨識語系", font=FONT_BODY, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        ctk.CTkLabel(
            self,
            text="強制指定辨識語系；「自動偵測」交由模型判斷。此設定對音檔轉字幕、批次辨識、錄製轉換全域生效。",
            font=FONT_SMALL, text_color=("gray40", "#AAAAAA"), anchor="w",
            wraplength=480, justify="left",
        ).pack(fill="x", padx=12, pady=(0, 4))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 8))

        ctk.CTkLabel(
            row, text="語系：", font=FONT_BODY, width=130, anchor="w",
        ).pack(side="left")

        self._app.lang_var   = ctk.StringVar(value="自動偵測")
        self._app.lang_combo = ctk.CTkComboBox(
            row, values=self._lang_values, variable=self._app.lang_var,
            width=200, state=self._lang_state, font=FONT_BODY,
        )
        self._app.lang_combo.pack(side="left")

    # ── 3. 中文輸出語言 ───────────────────────────────────────────────

    def _build_language_section(self):
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(10, 4))

        ctk.CTkLabel(
            row, text="🈶 中文輸出", font=FONT_BODY, width=130, anchor="w",
        ).pack(side="left")

        self.chinese_seg = ctk.CTkSegmentedButton(
            row, values=["繁體中文", "簡體中文"],
            width=160, height=30, font=FONT_BODY,
            command=self._on_chinese_seg,
        )
        self.chinese_seg.set("繁體中文")
        self.chinese_seg.pack(side="left")

        # 簡繁詞彙轉換（僅繁體模式生效）
        vrow = ctk.CTkFrame(self, fg_color="transparent")
        vrow.pack(fill="x", padx=12, pady=(0, 2))

        ctk.CTkLabel(vrow, text="", width=130).pack(side="left")  # 對齊用佔位
        self._vocab_var = ctk.BooleanVar(value=True)
        self._vocab_chk = ctk.CTkCheckBox(
            vrow, text="簡繁詞彙轉換", variable=self._vocab_var,
            font=FONT_BODY, command=self._on_vocab_chk,
        )
        self._vocab_chk.pack(side="left")

        ctk.CTkLabel(
            self,
            text="開啟：連用詞也在地化（軟件→軟體、質量→品質）；關閉：僅字形轉換，保留原始用詞。簡體模式下不適用。",
            font=FONT_SMALL, text_color=("gray40", "#AAAAAA"), anchor="w",
            wraplength=480, justify="left",
        ).pack(fill="x", padx=12, pady=(0, 8))

    def _on_chinese_seg(self, value: str):
        # 映射為 App._on_chinese_mode_change 接受的值
        mapped = "簡體" if "簡" in value else "繁體"
        self._app._on_chinese_mode_change(mapped)
        self._sync_vocab_state()

    def _on_vocab_chk(self):
        if hasattr(self._app, "_on_vocab_convert_change"):
            self._app._on_vocab_convert_change(self._vocab_var.get())

    def _sync_vocab_state(self):
        """簡體模式下停用詞彙轉換勾選（無作用）。"""
        simp = "簡" in self.chinese_seg.get()
        try:
            self._vocab_chk.configure(state="disabled" if simp else "normal")
        except Exception:
            pass

    # ── 4. VAD 語音偵測阈値 ─────────────────────────────

    def _build_vad_section(self):
        ctk.CTkLabel(
            self, text="🎤 語音偵測阈値（VAD Threshold）",
            font=FONT_BODY, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        ctk.CTkLabel(
            self,
            text="降低阈値可減少漏識（部分被判定為空白的片段可能有聲音）；提高則減少假陽性。預設：0.50。",
            font=FONT_SMALL, text_color=("gray40", "#AAAAAA"), anchor="w",
            wraplength=480, justify="left",
        ).pack(fill="x", padx=12, pady=(0, 4))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 8))

        self._vad_val_var = ctk.StringVar(value="0.50")
        ctk.CTkLabel(row, textvariable=self._vad_val_var,
                     font=FONT_BODY, width=44, anchor="e").pack(side="left")

        self._vad_slider = ctk.CTkSlider(
            row, from_=0.30, to=0.80, number_of_steps=50,
            width=280, height=18,
            command=self._on_vad_change,
        )
        self._vad_slider.set(0.50)
        self._vad_slider.pack(side="left", padx=(8, 8))

        ctk.CTkLabel(row, text="0.30",
                     font=FONT_SMALL, text_color=("gray50", "#888888")).pack(side="left")
        ctk.CTkLabel(row, text="–",
                     font=FONT_SMALL, text_color=("gray50", "#888888")).pack(side="left", padx=2)
        ctk.CTkLabel(row, text="0.80",
                     font=FONT_SMALL, text_color=("gray50", "#888888")).pack(side="left")

    def _on_vad_change(self, value: float):
        """VAD 閾値即時同步到全域變數與設定檔。"""
        self._vad_val_var.set(f"{value:.2f}")
        # 同步到 app 模組的 VAD_THRESHOLD
        import sys as _sys
        app_module = _sys.modules.get(type(self._app).__module__)
        if app_module and hasattr(app_module, "VAD_THRESHOLD"):
            app_module.VAD_THRESHOLD = value   # type: ignore
        self._app._patch_setting("vad_threshold", round(value, 2))

    # ── 6. FFmpeg ─────────────────────────────────────────────────────

    def _build_ffmpeg_section(self):
        ctk.CTkLabel(
            self, text="🎞 FFmpeg", font=FONT_BODY, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 12))

        self._ffmpeg_status_lbl = ctk.CTkLabel(
            row, text="（載入中…）",
            font=FONT_SMALL, anchor="w",
            text_color=("gray30", "gray70"),
        )
        self._ffmpeg_status_lbl.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            row, text="選擇 ffmpeg.exe", width=130, height=28, font=FONT_SMALL,
            command=self._on_choose_ffmpeg,
        ).pack(side="right")

    def _on_choose_ffmpeg(self):
        p = filedialog.askopenfilename(
            parent=self,
            title="選擇 ffmpeg.exe",
            filetypes=[("可執行檔", "ffmpeg.exe *.exe"), ("所有檔案", "*.*")],
        )
        if not p:
            return
        self._app._patch_setting("ffmpeg_path", p)  # type: ignore
        if hasattr(self._app, "_ffmpeg_exe"):
            self._app._ffmpeg_exe = Path(p)  # type: ignore
        self._ffmpeg_status_lbl.configure(
            text=f"✅ {p}", text_color=("green", "#88CC88"),
        )

    # ══ 對外 API ══════════════════════════════════════════════════════

    def sync_prefs(self, settings: dict):
        """由 App._apply_ui_prefs 呼叫，同步所有 UI 控件狀態。"""
        # 外觀主題
        mode = settings.get("appearance_mode", "dark")
        self.appearance_seg.set("☀ 淺色" if mode == "light" else "🌑 深色")

        # 中文輸出
        self.chinese_seg.set(
            "簡體中文" if settings.get("output_simplified") else "繁體中文"
        )

        # 簡繁詞彙轉換
        if hasattr(self, "_vocab_var"):
            self._vocab_var.set(bool(settings.get("vocab_convert", True)))
            self._sync_vocab_state()

        # 介面縮放
        if hasattr(self, "_scale_seg"):
            scale = float(settings.get("ui_scale", 1.0))
            label = min(
                self._SCALE_MAP,
                key=lambda k: abs(self._SCALE_MAP[k] - scale),
            )
            self._scale_seg.set(label)

        # VAD 閾值
        vad = float(settings.get("vad_threshold", 0.50))
        vad = max(0.30, min(0.80, vad))
        self._vad_slider.set(vad)
        self._vad_val_var.set(f"{vad:.2f}")

        # FFmpeg 狀態：自動偵測（含編譯版 bundled <app>/ffmpeg/ffmpeg.exe）
        # 偵測順序：使用者手動指定 → App 已偵測 → find_ffmpeg()（系統 PATH／/ffmpeg）
        detected = ""
        ffpath = settings.get("ffmpeg_path", "")
        if ffpath and Path(ffpath).exists():
            detected = ffpath
        else:
            ffexe = getattr(self._app, "_ffmpeg_exe", None)
            if ffexe and Path(ffexe).exists():
                detected = str(ffexe)
            else:
                try:
                    from ffmpeg_utils import find_ffmpeg
                    found = find_ffmpeg()
                    if found:
                        detected = str(found)
                        # 回寫至 App，供影片轉換與後續顯示直接取用
                        if hasattr(self._app, "_ffmpeg_exe"):
                            self._app._ffmpeg_exe = found  # type: ignore
                except Exception:
                    detected = ""

        if detected:
            self._ffmpeg_status_lbl.configure(
                text=f"✅ {detected}", text_color=("green", "#88CC88"),
            )
        else:
            self._ffmpeg_status_lbl.configure(
                text="❌ 未配置", text_color=("red", "#CC6666"),
            )

