"""model_tab.py — 模型分頁（引擎 / 裝置 / 模型選擇 + 模型路徑 / 下載來源 / CPU 效能）

ModelTab(CTkScrollableFrame) 把原本散在標題列（dev_bar）的「引擎 / 裝置 / 模型 /
重新載入」與原本在「設定」分頁的「模型路徑 / 下載來源 / CPU 推理效能」集中到單一
分頁，讓上方標題列只保留狀態摘要，騰出版面空間。

設計重點：
  • 引擎 / 裝置 / 模型的 combo 仍綁在 App 持有的 StringVar 上（device_var /
    model_var），widget 建立後**回寫**到 App（app.device_combo / app.model_combo /
    app.reload_btn），讓既有所有 `self.device_combo.configure(...)` 呼叫零修改。
  • CPU 版（app.py）有模型選擇（show_model_select=True）；GPU 版（app-gpu.py）模型
    固定，僅顯示裝置選擇（show_model_select=False）。
  • 模型路徑 / 鏡像站 / CPU 三節由 setting.py 整段移入，行為與相依（_app._patch_setting /
    _app._on_mirror_change）完全沿用。

對外 API：
    sync_prefs(settings: dict)  — 由 App._apply_ui_prefs 呼叫（同步路徑 / 鏡像 / CPU）
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

# ── 字型常數（與 app.py / setting.py 保持一致）────────────────────────
FONT_BODY  = ("Microsoft JhengHei", 13)
FONT_SMALL = ("Microsoft JhengHei", 11)
FONT_HEAD  = ("Microsoft JhengHei", 14, "bold")


def _hsep(parent):
    """水平分隔線。"""
    ctk.CTkFrame(
        parent, fg_color=("gray80", "gray25"), height=1, corner_radius=0,
    ).pack(fill="x", padx=0, pady=8)


class ModelTab(ctk.CTkScrollableFrame):
    """模型分頁：引擎 / 裝置 / 模型選擇、模型路徑、下載來源、CPU 推理效能。"""

    def __init__(
        self,
        parent,
        app,
        *,
        show_model_select: bool = True,
        device_default: str = "CPU",
        show_cpu_section: bool = True,
    ):
        super().__init__(parent, fg_color=("gray92", "gray17"))
        self._app               = app
        self._show_model_select = show_model_select
        self._device_default    = device_default
        self._show_cpu_section  = show_cpu_section
        self._build()

    # ══ 建構 UI ══════════════════════════════════════════════════════

    def _build(self):
        self._build_engine_section()
        _hsep(self)
        self._build_model_path_section()
        if self._show_cpu_section:
            _hsep(self)
            self._build_cpu_section()

    # ── 1. 引擎 / 裝置 / 模型選擇 ─────────────────────────────────────

    def _build_engine_section(self):
        ctk.CTkLabel(
            self, text="🖥 推理引擎 / 裝置", font=FONT_HEAD, anchor="w",
        ).pack(fill="x", padx=12, pady=(12, 4))

        # 裝置選擇列
        drow = ctk.CTkFrame(self, fg_color="transparent")
        drow.pack(fill="x", padx=12, pady=(0, 4))
        ctk.CTkLabel(
            drow, text="推理裝置：", font=FONT_BODY, width=72, anchor="w",
        ).pack(side="left")

        self._app.device_var   = ctk.StringVar(value=self._device_default)
        self._app.device_combo = ctk.CTkComboBox(
            drow, values=[self._device_default], variable=self._app.device_var,
            width=300, state="disabled", font=FONT_BODY,
        )
        self._app.device_combo.pack(side="left", padx=(4, 0))

        # GPU 版無模型選擇 → 重新載入鈕放在裝置列
        if not self._show_model_select:
            self._app.reload_btn = ctk.CTkButton(
                drow, text="重新載入", width=100, state="disabled", font=FONT_BODY,
                fg_color="gray35", hover_color="gray25",
                command=self._app._on_reload_models,
            )
            self._app.reload_btn.pack(side="left", padx=(12, 0))

        # 模型選擇列（CPU 版才有）
        if self._show_model_select:
            mrow = ctk.CTkFrame(self, fg_color="transparent")
            mrow.pack(fill="x", padx=12, pady=(0, 4))
            ctk.CTkLabel(
                mrow, text="模型：", font=FONT_BODY, width=72, anchor="w",
            ).pack(side="left")

            self._app.model_var   = ctk.StringVar(value="Qwen3-ASR-0.6B")
            self._app.model_combo = ctk.CTkComboBox(
                mrow,
                values=["Qwen3-ASR-0.6B", "Qwen3-ASR-1.7B INT8"],
                variable=self._app.model_var,
                width=220, state="readonly", font=FONT_BODY,
            )
            self._app.model_combo.pack(side="left", padx=(4, 0))

            self._app.reload_btn = ctk.CTkButton(
                mrow, text="重新載入", width=100, state="disabled", font=FONT_BODY,
                fg_color="gray35", hover_color="gray25",
                command=self._app._on_reload_models,
            )
            self._app.reload_btn.pack(side="left", padx=(12, 0))

        ctk.CTkLabel(
            self,
            text="切換裝置或模型後，點「重新載入」套用。GPU（Vulkan / CUDA）需安裝對應驅動。",
            font=FONT_SMALL, text_color=("gray40", "#AAAAAA"), anchor="w",
            wraplength=480, justify="left",
        ).pack(fill="x", padx=12, pady=(0, 6))

    # ── 2. 模型路徑 + 下載來源（鏡像站）──────────────────────────────

    def _build_model_path_section(self):
        ctk.CTkLabel(
            self, text="📦 模型路徑", font=FONT_HEAD, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 10))

        self._model_path_lbl = ctk.CTkLabel(
            row, text=self._get_model_path_text(),
            font=FONT_SMALL, anchor="w",
            text_color=("gray30", "gray70"),
            wraplength=400, justify="left",
        )
        self._model_path_lbl.pack(side="left", fill="x", expand=True)

        ctk.CTkButton(
            row, text="更改…", width=72, height=28, font=FONT_SMALL,
            command=self._on_change_model_dir,
        ).pack(side="right")

        # ── 下載來源（HuggingFace 鏡像站）─────────────────────────────
        ctk.CTkLabel(
            self, text="模型下載來源", font=FONT_SMALL,
            text_color=("gray40", "#AAAAAA"), anchor="w",
        ).pack(fill="x", padx=12, pady=(2, 2))

        src_row = ctk.CTkFrame(self, fg_color="transparent")
        src_row.pack(fill="x", padx=12, pady=(0, 2))

        self._mirror_seg = ctk.CTkSegmentedButton(
            src_row, values=["官方 HF", "鏡像站"],
            width=160, height=28, font=FONT_SMALL,
            command=self._on_mirror_seg,
        )
        self._mirror_seg.set("官方 HF")
        self._mirror_seg.pack(side="left")

        self._mirror_var   = ctk.StringVar(value="https://hf-mirror.com")
        self._mirror_entry = ctk.CTkEntry(
            src_row, textvariable=self._mirror_var,
            width=230, height=28, font=FONT_SMALL,
        )
        self._mirror_entry.pack(side="left", padx=(8, 0))
        self._mirror_entry.bind("<FocusOut>", lambda _e: self._apply_mirror_if_on())
        self._mirror_entry.bind("<Return>",   lambda _e: self._apply_mirror_if_on())

        ctk.CTkLabel(
            self,
            text="直連 huggingface.co 緩慢或逾時時，可切換鏡像站（預設 hf-mirror.com）。",
            font=FONT_SMALL, text_color=("gray40", "#AAAAAA"), anchor="w",
            wraplength=480, justify="left",
        ).pack(fill="x", padx=12, pady=(0, 10))

    def _on_mirror_seg(self, value: str):
        use_mirror = ("鏡" in value)
        try:
            self._mirror_entry.configure(state="normal" if use_mirror else "disabled")
        except Exception:
            pass
        base = self._mirror_var.get().strip() if use_mirror else ""
        if hasattr(self._app, "_on_mirror_change"):
            self._app._on_mirror_change(base)

    def _apply_mirror_if_on(self):
        """編輯鏡像網址後，若目前為鏡像模式則即時套用。"""
        if "鏡" in self._mirror_seg.get():
            base = self._mirror_var.get().strip()
            if hasattr(self._app, "_on_mirror_change"):
                self._app._on_mirror_change(base)

    def _get_model_path_text(self) -> str:
        """取得顯示用模型路徑文字（相容 app.py / app-gpu.py）。"""
        app_module = sys.modules.get(type(self._app).__module__)
        # GPU 版（app-gpu.py）有 GPU_MODEL_DIR 模組全域
        gpu_dir  = getattr(app_module, "GPU_MODEL_DIR",  None)
        asr_name = getattr(app_module, "ASR_MODEL_NAME", None)
        if gpu_dir and asr_name:
            return str(gpu_dir / asr_name)
        # CPU 版（app.py）使用 self._settings
        if hasattr(self._app, "_settings"):
            md = self._app._settings.get("model_dir", "")  # type: ignore
            return str(md) if md else "（尚未設定）"
        return "（尚未設定）"

    def _on_change_model_dir(self):
        d = filedialog.askdirectory(parent=self, title="選擇模型目錄")
        if not d:
            return
        app_module = sys.modules.get(type(self._app).__module__)
        # GPU 版儲存至 gpu_model_dir
        if getattr(app_module, "GPU_MODEL_DIR", None):
            self._app._patch_setting("gpu_model_dir", d)  # type: ignore
        else:
            self._app._patch_setting("model_dir", d)  # type: ignore
        self._model_path_lbl.configure(text=d)

    # ── 3. CPU 推理效能 ───────────────────────────────────────────────

    def _build_cpu_section(self):
        _logical = os.cpu_count() or 1
        ctk.CTkLabel(
            self, text="⚡ CPU 推理效能", font=FONT_HEAD, anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 2))

        ctk.CTkLabel(
            self,
            text=(
                "「自動」使用 OpenVINO 預設（約 50–70% 核心）。"
                f"「全速」啟用所有 {_logical} 個邏輯核心，可加快推理 30–80%，"
                "但佔用更多 CPU 資源。修改後需重新載入模型。"
            ),
            font=FONT_SMALL, text_color=("gray40", "#AAAAAA"), anchor="w",
            wraplength=480, justify="left",
        ).pack(fill="x", padx=12, pady=(0, 6))

        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(0, 8))

        self._cpu_seg = ctk.CTkSegmentedButton(
            row,
            values=["自動（省電）", f"全速（{_logical} 執行緒）"],
            width=240, height=30, font=FONT_BODY,
            command=self._on_cpu_change,
        )
        self._cpu_seg.set("自動（省電）")
        self._cpu_seg.pack(side="left")

        self._cpu_reload_hint = ctk.CTkLabel(
            row, text="", font=FONT_SMALL,
            text_color=("gray40", "#AAAAAA"),
        )
        self._cpu_reload_hint.pack(side="left", padx=(10, 0))

    def _on_cpu_change(self, value: str):
        _logical = os.cpu_count() or 1
        n = _logical if "全速" in value else 0
        self._app._patch_setting("cpu_threads", n)
        self._cpu_reload_hint.configure(text="↺ 需重新載入模型生效")
        self.after(4000, lambda: self._cpu_reload_hint.configure(text=""))

    # ══ 對外 API ══════════════════════════════════════════════════════

    def sync_prefs(self, settings: dict):
        """由 App._apply_ui_prefs 呼叫，同步路徑 / 鏡像 / CPU 控件狀態。"""
        # 下載來源（鏡像站）
        if hasattr(self, "_mirror_seg"):
            mirror = (settings.get("hf_mirror", "") or "").strip()
            if mirror:
                self._mirror_var.set(mirror)
                self._mirror_seg.set("鏡像站")
                self._mirror_entry.configure(state="normal")
            else:
                self._mirror_seg.set("官方 HF")
                self._mirror_entry.configure(state="disabled")

        # CPU 效能
        if hasattr(self, "_cpu_seg"):
            cpu_threads = int(settings.get("cpu_threads", 0))
            _logical = os.cpu_count() or 1
            self._cpu_seg.set(
                f"全速（{_logical} 執行緒）" if cpu_threads > 0 else "自動（省電）"
            )

        # 模型路徑
        if hasattr(self, "_model_path_lbl"):
            self._model_path_lbl.configure(text=self._get_model_path_text())
