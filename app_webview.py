"""app_webview.py — 桌面 WebView 啟動器（本機 HTTP 伺服器 + Edge --app 視窗）

架構（與 Eel / flaskwebgui 同類，但零第三方相依、純標準庫，方便 PyInstaller
打包，亦與既有 api_server.py 一致）：
  1. 起本機伺服器 webview_server（只綁 127.0.0.1，外部連不到）
  2. 背景載入 ASR 模型
  3. 用 Microsoft Edge 的 --app 模式開「無網址列」視窗指向本機網址
     （找不到 Edge → fallback 開預設瀏覽器）
  4. 等視窗（Edge 程序）關閉 → 收掉伺服器 → 結束

刻意不使用 pywebview：其 Windows EdgeChromium 後端（pythonnet）在本機有
無限遞迴 bug，且在 EXE 內需額外打包 .NET 相依，是不必要的風險來源。
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from webview_server import WebViewServer

APP_NAME = "聲音辨識"
WIN_W, WIN_H = 1180, 820


# ── 尋找 Microsoft Edge ────────────────────────────────────────────────
def _find_edge() -> str | None:
    candidates = [
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return str(c)
    # 退而求其次：找 Chrome
    for c in [
        Path(os.environ.get("PROGRAMFILES", "")) / "Google/Chrome/Application/chrome.exe",
        Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google/Chrome/Application/chrome.exe",
    ]:
        if c and Path(c).is_file():
            return str(c)
    return None


def _profile_dir() -> str:
    """Edge --app 專屬 user-data-dir：固定路徑 → 記住視窗大小/位置，且強制獨立實例
    （讓我們能等這個程序結束來判定視窗關閉）。"""
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME / "edge-profile"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def open_window(url: str) -> subprocess.Popen | None:
    """以 Edge --app 開無框視窗；回傳該程序（供等待關閉）。找不到 → 開預設瀏覽器回 None。"""
    edge = _find_edge()
    if edge:
        args = [
            edge,
            f"--app={url}",
            f"--user-data-dir={_profile_dir()}",
            f"--window-size={WIN_W},{WIN_H}",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        # 隱藏可能的主控台旗標（凍結環境）
        flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
        try:
            return subprocess.Popen(args, creationflags=flags)
        except Exception:
            pass
    # fallback：預設瀏覽器（有網址列，但保證能開）
    webbrowser.open(url)
    return None


def main():
    srv = WebViewServer(host="127.0.0.1", port=0)   # port=0 → 自動挑空閒埠
    srv.start()
    srv.backend.start_load()                        # 背景載入模型
    url = srv.url
    print(f"[{APP_NAME}] {url}")

    proc = open_window(url)

    try:
        if proc is not None:
            proc.wait()                              # 等 Edge 視窗關閉
        else:
            threading.Event().wait()                 # fallback：無從偵測關閉，常駐
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
