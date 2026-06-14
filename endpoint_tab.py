"""endpoint_tab.py — 「端點」分頁（OpenAI 相容服務的控制台）

整合：
  • API 服務開關（啟動/停止 api_server.TranscribeServer，與既有引擎共用）
  • 本機 / 區網網址 + 一鍵開啟網頁（stdlib webbrowser）
  • QR code（segno，純 python；未安裝時優雅降級為提示）
  • trycloudflare 對外臨時網址（cloudflared 隨用下載 + 子程序通道 + 解析網址）

app.py / app-gpu.py 共用：
    from endpoint_tab import EndpointTab
    self._endpoint_tab = EndpointTab(self.tabs.tab("  端點  "), self)
    self._endpoint_tab.pack(fill="both", expand=True)

對外 API：
    start_api_if_enabled()  — 由 App._on_models_ready 呼叫
    stop_all()              — 由 App._on_close 呼叫
"""
from __future__ import annotations

import re
import socket
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path
from tkinter import messagebox, filedialog

import customtkinter as ctk

FONT_BODY  = ("Microsoft JhengHei", 13)
FONT_SMALL = ("Microsoft JhengHei", 11)
FONT_MONO  = ("Consolas", 12)

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_CF_URL = ("https://github.com/cloudflare/cloudflared/releases/latest/"
           "download/cloudflared-windows-amd64.exe")
_TRYCF_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def _local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class EndpointTab(ctk.CTkScrollableFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color=("gray92", "gray17"))
        self._app = app
        self._cf_proc: subprocess.Popen | None = None
        self._cf_url: str | None = None
        self._cf_qr_img = None       # 保留右側 QR 面板的 CTkImage 參照（避免 GC）
        self._qr_png_bytes = None    # 目前面板顯示的 QR PNG 位元組（供下載）
        self._qr_data = None         # 目前面板顯示的 QR 對應網址
        self._build()

    # ══ UI ════════════════════════════════════════════════════════════
    def _build(self):
        ctk.CTkLabel(self, text="🌐 OpenAI 相容轉錄端點", font=("Microsoft JhengHei", 16, "bold"),
                     anchor="w").pack(fill="x", padx=14, pady=(14, 2))
        ctk.CTkLabel(
            self,
            text="啟用後可在瀏覽器上傳音檔轉錄，或供外部程式呼叫 "
                 "POST /v1/audio/transcriptions（與目前載入的引擎共用，支援 chatllm / OpenVINO）",
            font=FONT_SMALL, anchor="w", justify="left",
            text_color=("gray35", "gray65"), wraplength=600,
        ).pack(fill="x", padx=14, pady=(0, 10))

        # ── 服務開關 + 埠號 ──────────────────────────────────────────
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=(0, 6))
        self._api_enable_var = ctk.BooleanVar(value=False)
        ctk.CTkSwitch(row, text="啟用服務", variable=self._api_enable_var,
                      font=FONT_BODY, command=self._on_api_toggle).pack(side="left")
        ctk.CTkLabel(row, text="埠號：", font=FONT_SMALL,
                     text_color=("gray35", "gray65")).pack(side="left", padx=(18, 2))
        self._api_port_var = ctk.StringVar(value="11435")
        ctk.CTkEntry(row, textvariable=self._api_port_var, width=72,
                     font=FONT_SMALL).pack(side="left")
        self._api_status_lbl = ctk.CTkLabel(row, text="（未啟用）", font=FONT_SMALL,
                                             text_color=("gray35", "gray65"))
        self._api_status_lbl.pack(side="left", padx=(14, 0))

        # ── 本機 / 區網網址 ──────────────────────────────────────────
        self._url_box = ctk.CTkFrame(self, fg_color=("gray88", "gray20"))
        self._url_box.pack(fill="x", padx=14, pady=(6, 6))
        self._build_url_row("本機：", "_local")
        self._build_url_row("區網：", "_lan")

        # ── 對外臨時網址（trycloudflare）— 左控制 / 右 QR 兩欄 ────────
        #    QR 放在區塊右側，建立網址後即時顯示，免向下捲動。
        cf_sec = ctk.CTkFrame(self, fg_color="transparent")
        cf_sec.pack(fill="x", padx=14, pady=(10, 2))
        cf_left = ctk.CTkFrame(cf_sec, fg_color="transparent")
        cf_left.pack(side="left", fill="x", expand=True)
        cf_right = ctk.CTkFrame(cf_sec, fg_color="transparent")
        cf_right.pack(side="right", anchor="n", padx=(12, 0))

        # 右側：QR 面板（區網 / 外網共用；外網啟用時以外網 QR 為主）
        self._cf_qr_lbl = ctk.CTkLabel(
            cf_right, text="QR\n（按 QR 或建立外網後顯示）", font=FONT_SMALL,
            width=180, height=180, corner_radius=8,
            fg_color=("gray85", "gray22"), text_color=("gray45", "gray60"),
        )
        self._cf_qr_lbl.pack()
        self._qr_cap = ctk.CTkLabel(cf_right, text="", font=FONT_SMALL,
                                    text_color=("gray35", "gray65"))
        self._qr_cap.pack(pady=(4, 0))
        self._qr_dl_btn = ctk.CTkButton(
            cf_right, text="下載 QR", width=120, height=26, font=FONT_SMALL,
            state="disabled", command=self._on_qr_download)
        self._qr_dl_btn.pack(pady=(6, 0))

        ctk.CTkLabel(cf_left, text="🌍 對外臨時網址（trycloudflare）", font=FONT_BODY,
                     anchor="w").pack(fill="x", pady=(0, 2))
        ctk.CTkLabel(
            cf_left, text="透過 Cloudflare 臨時通道讓手機在外網也能連線（免帳號、自帶 https、"
                       "每次重啟網址會變）。首次使用會下載 cloudflared（約 25MB）。",
            font=FONT_SMALL, anchor="w", justify="left",
            text_color=("gray35", "gray65"), wraplength=420,
        ).pack(fill="x", pady=(0, 2))
        # 風險警告（醒目）
        ctk.CTkLabel(
            cf_left,
            text="⚠ 風險提醒：開啟對外網址＝把服務公開到網際網路。網址含金鑰（等同密碼），"
                 "任何取得完整網址的人都能上傳音檔、消耗你的 CPU/GPU。用完請立即停止，勿外流網址/QR。",
            font=("Microsoft JhengHei", 11, "bold"), anchor="w", justify="left",
            text_color=("#B00", "#F0A0A0"), wraplength=420,
        ).pack(fill="x", pady=(0, 6))
        cfrow = ctk.CTkFrame(cf_left, fg_color="transparent")
        cfrow.pack(fill="x", pady=(0, 4))
        self._cf_btn = ctk.CTkButton(cfrow, text="建立對外網址", width=130, height=30,
                                     font=FONT_SMALL, command=self._on_cf_toggle)
        self._cf_btn.pack(side="left")
        self._cf_open_btn = ctk.CTkButton(cfrow, text="開啟", width=64, height=30,
                                          font=FONT_SMALL, state="disabled",
                                          command=lambda: self._open(self._cf_url))
        self._cf_open_btn.pack(side="left", padx=(8, 0))
        self._cf_qr_btn = ctk.CTkButton(cfrow, text="QR", width=64, height=30,
                                        font=FONT_SMALL, state="disabled",
                                        command=self._show_cf_qr)
        self._cf_qr_btn.pack(side="left", padx=(8, 0))
        self._cf_status = ctk.CTkLabel(cf_left, text="", font=FONT_MONO, anchor="w",
                                       text_color=("gray35", "gray65"), wraplength=420,
                                       justify="left")
        self._cf_status.pack(fill="x", pady=(0, 6))

        self._refresh_urls()

    def _build_url_row(self, prefix: str, which: str):
        row = ctk.CTkFrame(self._url_box, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(row, text=prefix, font=FONT_SMALL, width=46,
                     text_color=("gray35", "gray65")).pack(side="left")
        lbl = ctk.CTkLabel(row, text="（服務未啟用）", font=FONT_MONO, anchor="w")
        lbl.pack(side="left", fill="x", expand=True)
        setattr(self, which + "_lbl", lbl)
        ctk.CTkButton(row, text="開啟", width=58, height=26, font=FONT_SMALL,
                      command=lambda: self._open(getattr(self, which + "_url", None))
                      ).pack(side="left", padx=(6, 0))
        if which == "_lan":
            ctk.CTkButton(row, text="QR", width=58, height=26, font=FONT_SMALL,
                          command=self._show_lan_qr
                          ).pack(side="left", padx=(6, 0))

    # ══ API 服務 ══════════════════════════════════════════════════════
    def _api_token(self) -> str:
        srv = getattr(self._app, "_api_server", None)
        return getattr(srv, "token", "") if srv else ""

    def _refresh_urls(self):
        port = self._api_port_var.get()
        running = self._api_running()
        q = f"/?k={self._api_token()}" if running else "/"
        self._local_url = f"http://127.0.0.1:{port}{q}" if running else None
        self._lan_url = f"http://{_local_ip()}:{port}{q}" if running else None
        self._local_lbl.configure(text=self._local_url or "（服務未啟用）")
        self._lan_lbl.configure(text=self._lan_url or "（服務未啟用）")

    def _api_running(self) -> bool:
        srv = getattr(self._app, "_api_server", None)
        return bool(srv and getattr(srv, "running", False))

    def _on_api_toggle(self):
        enable = bool(self._api_enable_var.get())
        try:
            port = int(self._api_port_var.get())
        except ValueError:
            port = 11435
            self._api_port_var.set("11435")
        self._app._patch_setting("api_enabled", enable)
        self._app._patch_setting("api_port", port)
        if enable:
            ok, msg = self._start_api(port)
            self._api_status_lbl.configure(
                text=msg, text_color=("green", "#3a9") if ok else ("red", "#c55"))
            if not ok:
                self._api_enable_var.set(False)
        else:
            self._stop_api()
            self._api_status_lbl.configure(text="（已停止）",
                                           text_color=("gray35", "gray65"))
        self._refresh_urls()

    def _start_api(self, port: int):
        from api_server import TranscribeServer
        self._stop_api()
        srv = TranscribeServer(get_engine=lambda: getattr(self._app, "engine", None),
                               port=port)
        try:
            srv.start()
        except OSError:
            return False, f"❌ 埠 {port} 可能被占用"
        except Exception as e:
            return False, f"❌ 啟動失敗：{e}"
        self._app._api_server = srv
        return True, "✅ 執行中"

    def _stop_api(self):
        srv = getattr(self._app, "_api_server", None)
        if srv is not None:
            try:
                srv.stop()
            except Exception:
                pass
        self._app._api_server = None

    def start_api_if_enabled(self):
        """由 App._on_models_ready 呼叫。"""
        s = getattr(self._app, "_settings", None) or {}
        try:
            self._api_port_var.set(str(int(s.get("api_port", 11435))))
        except (ValueError, TypeError):
            pass
        if not s.get("api_enabled"):
            self._refresh_urls()
            return
        self._api_enable_var.set(True)
        ok, msg = self._start_api(int(self._api_port_var.get()))
        self._api_status_lbl.configure(
            text=msg, text_color=("green", "#3a9") if ok else ("red", "#c55"))
        if not ok:
            self._api_enable_var.set(False)
        self._refresh_urls()

    # ══ 開啟網頁 ══════════════════════════════════════════════════════
    def _open(self, url: str | None):
        if url:
            webbrowser.open(url)

    # ══ QR（segno；區網 / 外網共用右側面板，外網優先，可下載）═════════
    def _qr_png(self, data: str):
        """產生 QR 的 PNG 位元組（segno，純 python）；未安裝回 None。"""
        try:
            import io
            import segno
            buf = io.BytesIO()
            segno.make(data, error="m").save(buf, kind="png", scale=10, border=2)
            return buf.getvalue()
        except Exception:
            return None

    def _render_qr(self, data: str, caption: str):
        """把 QR 畫到右側面板，並記錄 PNG 位元組供下載。"""
        png = self._qr_png(data)
        if png is None:
            self._cf_qr_lbl.configure(
                image=None, text="未安裝 segno\npip install segno",
                text_color=("red", "#c55"))
            self._qr_dl_btn.configure(state="disabled")
            return
        try:
            import io
            from PIL import Image
            img = Image.open(io.BytesIO(png)).convert("RGB")
            self._cf_qr_img = ctk.CTkImage(light_image=img, dark_image=img,
                                           size=(170, 170))
        except Exception:
            self._cf_qr_img = None
        self._qr_png_bytes = png
        self._qr_data = data
        self._cf_qr_lbl.configure(image=self._cf_qr_img, text="")
        self._qr_cap.configure(text=caption)
        self._qr_dl_btn.configure(state="normal")

    def _show_cf_qr(self):
        """顯示對外（trycloudflare）網址 QR。"""
        if not self._cf_url:
            return
        self._render_qr(self._cf_url, "外網臨時網址")

    def _show_lan_qr(self):
        """顯示區網 QR；外網通道啟用時改以外網 QR 為主，避免兩個 QR 混淆。"""
        if self._cf_url:
            self._show_cf_qr()
            self._set_cf("外網通道啟用中，已優先顯示外網 QR。")
            return
        if not getattr(self, "_lan_url", None):
            return
        self._render_qr(self._lan_url, "區網（同網段裝置可掃）")

    def _clear_cf_qr(self):
        """還原 QR 面板為佔位狀態（外網通道停止時）。"""
        self._cf_qr_img = None
        self._qr_png_bytes = None
        self._qr_data = None
        self._cf_qr_lbl.configure(
            image=None, text="QR\n（按 QR 或建立外網後顯示）",
            text_color=("gray45", "gray60"))
        self._qr_cap.configure(text="")
        self._qr_dl_btn.configure(state="disabled")

    def _on_qr_download(self):
        """把目前面板顯示的 QR 存成 PNG 檔。"""
        if not self._qr_png_bytes:
            return
        f = filedialog.asksaveasfilename(
            parent=self, title="儲存 QR 圖片",
            defaultextension=".png", initialfile="qwenasr_qr.png",
            filetypes=[("PNG 圖片", "*.png"), ("所有檔案", "*.*")])
        if not f:
            return
        try:
            with open(f, "wb") as fp:
                fp.write(self._qr_png_bytes)
            self._set_cf(f"✅ QR 已儲存：{f}", ok=True)
        except Exception as e:
            self._set_cf(f"❌ QR 儲存失敗：{e}", err=True)

    # ══ cloudflared 對外通道 ═══════════════════════════════════════════
    def _find_cloudflared(self) -> Path | None:
        import shutil
        which = shutil.which("cloudflared")
        if which:
            return Path(which)
        for c in (_base_dir() / "cloudflared" / "cloudflared.exe",
                  Path("C:/Program Files (x86)/cloudflared/cloudflared.exe")):
            if c.exists():
                return c
        return None

    def _on_cf_toggle(self):
        if self._cf_proc is not None:
            self._stop_tunnel()
            return
        if not self._api_running():
            self._cf_status.configure(text="請先啟用上方的 API 服務", text_color=("red", "#c55"))
            return
        # 開啟前的風險確認（公開到網際網路、無密碼驗證）
        if not messagebox.askyesno(
            "確認開啟對外網址？",
            "即將透過 Cloudflare 臨時通道把轉錄服務公開到網際網路。\n\n"
            "• 網址含存取金鑰（等同密碼）；任何取得「完整網址」的人都能使用\n"
            "• 會消耗你的 CPU/GPU\n"
            "• 網址/QR 請勿外流；用完請按「停止對外」關閉\n\n"
            "確定要開啟嗎？",
            icon="warning", default="no",
        ):
            return
        self._cf_btn.configure(state="disabled", text="準備中…")
        threading.Thread(target=self._cf_worker, daemon=True).start()

    def _cf_worker(self):
        cf = self._find_cloudflared()
        if cf is None:
            self._set_cf("⬇ 下載 cloudflared（約 25MB）…")
            try:
                cf = self._download_cloudflared()
            except Exception as e:
                self._set_cf(f"❌ cloudflared 下載失敗：{e}", err=True)
                self.after(0, lambda: self._cf_btn.configure(state="normal", text="建立對外網址"))
                return
        port = self._api_port_var.get()
        self._set_cf("建立通道中…（首次連線約需數秒）")
        # 用空設定檔覆蓋預設 ~/.cloudflared/config.yml，避免使用者既有具名
        # tunnel 的 ingress 規則（含 catch-all http_status:404）攔截我們的快速通道。
        empty_cfg = _base_dir() / "cloudflared" / "quicktunnel.yml"
        try:
            empty_cfg.parent.mkdir(parents=True, exist_ok=True)
            empty_cfg.write_text(
                "# QwenASR quick tunnel - intentionally empty to bypass "
                "~/.cloudflared/config.yml ingress rules\n",
                encoding="utf-8")
        except Exception:
            pass
        try:
            self._cf_proc = subprocess.Popen(
                [str(cf), "--config", str(empty_cfg),
                 "tunnel", "--url", f"http://127.0.0.1:{port}",
                 "--no-autoupdate"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                creationflags=_NO_WINDOW,
            )
        except Exception as e:
            self._set_cf(f"❌ 啟動 cloudflared 失敗：{e}", err=True)
            self.after(0, lambda: self._cf_btn.configure(state="normal", text="建立對外網址"))
            return

        self.after(0, lambda: self._cf_btn.configure(state="normal", text="停止對外"))
        # 讀 stdout 找 trycloudflare 網址
        for line in self._cf_proc.stdout:                      # type: ignore
            m = _TRYCF_RE.search(line)
            if m:
                tok = self._api_token()
                self._cf_url = m.group(0) + (f"/?k={tok}" if tok else "/")
                self._set_cf(f"✅ 對外網址（含金鑰，等同密碼）：{self._cf_url}", ok=True)
                self.after(0, lambda: (self._cf_open_btn.configure(state="normal"),
                                       self._cf_qr_btn.configure(state="normal"),
                                       self._show_cf_qr()))  # 右側面板即時顯示 QR
                break
        # 持續耗用輸出避免管線阻塞，直到程序結束
        try:
            for _ in self._cf_proc.stdout:                     # type: ignore
                pass
        except Exception:
            pass

    def _download_cloudflared(self) -> Path:
        import ssl
        import urllib.request
        try:
            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = ssl.create_default_context()
        dest_dir = _base_dir() / "cloudflared"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / "cloudflared.exe"
        req = urllib.request.Request(_CF_URL, headers={"User-Agent": "QwenASR"})
        with urllib.request.urlopen(req, context=ctx, timeout=60) as r, \
             open(str(dest) + ".tmp", "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            while True:
                b = r.read(65536)
                if not b:
                    break
                f.write(b)
                done += len(b)
                if total:
                    self._set_cf(f"⬇ 下載 cloudflared… {done/1048576:.0f}/{total/1048576:.0f} MB")
        import os
        os.replace(str(dest) + ".tmp", str(dest))
        return dest

    def _stop_tunnel(self):
        if self._cf_proc is not None:
            try:
                self._cf_proc.terminate()
            except Exception:
                pass
            self._cf_proc = None
        self._cf_url = None
        self._cf_btn.configure(text="建立對外網址", state="normal")
        self._cf_open_btn.configure(state="disabled")
        self._cf_qr_btn.configure(state="disabled")
        self._clear_cf_qr()
        self._set_cf("（對外通道已停止）")

    def _set_cf(self, text: str, ok=False, err=False):
        color = ("green", "#3a9") if ok else (("red", "#c55") if err else ("gray35", "gray65"))
        self.after(0, lambda: self._cf_status.configure(text=text, text_color=color))

    # ══ 生命週期 ══════════════════════════════════════════════════════
    def stop_all(self):
        """由 App._on_close 呼叫。"""
        try:
            self._stop_tunnel()
        except Exception:
            pass
        self._stop_api()
