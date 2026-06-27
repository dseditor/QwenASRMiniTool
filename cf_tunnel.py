"""cf_tunnel.py — cloudflared 快速通道 + QR 產生（UI／傳輸無關，可共用）

把「對外臨時網址（trycloudflare）」與「QR 圖」的純邏輯從桌面版
endpoint_tab.py 抽出，讓 WebView 後端（webview_backend）也能重用，
不依賴 customtkinter / tkinter。桌面版維持原本內嵌實作不變。

公開介面：
    find_cloudflared()                  → 找本機 cloudflared（PATH／常見路徑）
    download_cloudflared(progress_cb)   → 隨用下載 cloudflared.exe（約 25MB）
    make_qr_png(data)                   → segno 產 QR 的 PNG 位元組（未安裝回 None）
    CloudflareTunnel                    → 管理單一 quick tunnel 子程序
"""
from __future__ import annotations

import io
import os
import re
import shutil
import ssl
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_CF_URL = ("https://github.com/cloudflare/cloudflared/releases/latest/"
           "download/cloudflared-windows-amd64.exe")
_TRYCF_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


def find_cloudflared() -> Path | None:
    """找本機 cloudflared：系統 PATH → App 目錄 → 常見安裝路徑。"""
    which = shutil.which("cloudflared")
    if which:
        return Path(which)
    for c in (_base_dir() / "cloudflared" / "cloudflared.exe",
              Path("C:/Program Files (x86)/cloudflared/cloudflared.exe")):
        if c.exists():
            return c
    return None


def download_cloudflared(progress_cb=None) -> Path:
    """下載官方 cloudflared.exe 到 <app>/cloudflared/。progress_cb(frac, msg)。"""
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
            if total and progress_cb:
                progress_cb(done / total,
                            f"下載 cloudflared… {done/1048576:.0f}/{total/1048576:.0f} MB")
    os.replace(str(dest) + ".tmp", str(dest))
    return dest


def make_qr_png(data: str) -> bytes | None:
    """產生 QR 的 PNG 位元組（segno，純 python）；未安裝或失敗回 None。"""
    try:
        import segno
        buf = io.BytesIO()
        segno.make(data, error="m").save(buf, kind="png", scale=10, border=2)
        return buf.getvalue()
    except Exception:
        return None


class CloudflareTunnel:
    """管理單一 cloudflared quick tunnel 子程序（指向本機某埠）。"""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self.url: str | None = None          # 含金鑰的完整外網網址
        self.status: str = ""                # 最近一次狀態訊息（"ready"=已就緒）
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._proc is not None

    def start(self, port: int, token: str | None = None, status_cb=None) -> str | None:
        """建立指向 http://127.0.0.1:<port> 的快速通道，阻塞直到取得網址或失敗。

        建議於背景執行緒呼叫。status_cb(msg) 回報進度（含 "ready"）。
        缺 cloudflared 會自動下載（約 25MB）。回傳含金鑰的完整外網網址。
        """
        def _set(msg: str):
            self.status = msg
            if status_cb:
                try:
                    status_cb(msg)
                except Exception:
                    pass

        with self._lock:
            if self._proc is not None:
                return self.url

        cf = find_cloudflared()
        if cf is None:
            _set("下載 cloudflared（約 25MB）…")
            cf = download_cloudflared(progress_cb=lambda frac, m: _set(m))

        _set("建立通道中…（首次連線約需數秒）")
        # 空設定檔覆蓋預設 ~/.cloudflared/config.yml，避免既有具名 tunnel 的
        # ingress 規則（含 catch-all http_status:404）攔截我們的快速通道。
        empty_cfg = _base_dir() / "cloudflared" / "quicktunnel.yml"
        try:
            empty_cfg.parent.mkdir(parents=True, exist_ok=True)
            empty_cfg.write_text(
                "# QwenASR quick tunnel - intentionally empty to bypass "
                "~/.cloudflared/config.yml ingress rules\n", encoding="utf-8")
        except Exception:
            pass

        proc = subprocess.Popen(
            [str(cf), "--config", str(empty_cfg), "tunnel",
             "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            creationflags=_NO_WINDOW,
        )
        with self._lock:
            self._proc = proc

        # 讀 stdout 找 trycloudflare 網址
        for line in proc.stdout:                                  # type: ignore
            m = _TRYCF_RE.search(line)
            if m:
                self.url = m.group(0) + (f"/?k={token}" if token else "/")
                _set("ready")
                break
        # 持續排空輸出避免管線阻塞，直到程序結束
        threading.Thread(target=self._drain, args=(proc,), daemon=True).start()
        return self.url

    def _drain(self, proc: subprocess.Popen):
        try:
            for _ in proc.stdout:                                 # type: ignore
                pass
        except Exception:
            pass

    def stop(self):
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        self.url = None
        self.status = ""
