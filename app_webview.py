"""app_webview.py — 桌面 WebView 啟動器（本機 HTTP server + 原生 WebView2 視窗）

視窗用 pywebview 開「只載入本機網址」的原生 WebView2 視窗：系統內建
WebView2 runtime（不打包 Chromium → EXE 小）、原生標題列、無瀏覽器感、
無登入提示。前端完全透過 HTTP/SSE 與 server 溝通，**不使用 pywebview 的
js_api** —— 那一層（pythonnet 序列化 .NET 物件）正是先前踩到無限遞迴
死鎖的元兇；改成純載入網址後視窗穩定（實測無遞迴、視窗持續存活）。

啟動順序：起本機 server(只綁 127.0.0.1) → 背景載入模型 → 開原生視窗。
WebView2 不可用時 fallback：系統 Edge --app 無痕視窗 → 再不行開預設瀏覽器。
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


# ════════════════════════════════════════════════════════
# 主要：原生 WebView2 視窗（pywebview，只載入網址、無 js_api）
# ════════════════════════════════════════════════════════
def run_native_window(url: str) -> bool:
    """以原生 WebView2 視窗載入 url，阻塞至關窗回 True；不可用/失敗回 False。"""
    try:
        import webview
    except Exception:
        return False
    try:
        webview.create_window(
            APP_NAME, url=url,
            width=WIN_W, height=WIN_H, min_size=(960, 680),
        )
        webview.start()        # 阻塞至視窗關閉（在主執行緒）
        return True
    except Exception as e:
        print(f"[{APP_NAME}] 原生視窗失敗，改用 Edge：{e}")
        return False


# ════════════════════════════════════════════════════════
# Fallback：系統 Microsoft Edge --app 無痕視窗
# ════════════════════════════════════════════════════════
def _find_edge() -> str | None:
    for c in [
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Edge/Application/msedge.exe",
    ]:
        if c and Path(c).is_file():
            return str(c)
    return None


def _profile_dir() -> str:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME / "edge-profile"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def open_edge_app(url: str) -> subprocess.Popen | None:
    edge = _find_edge()
    if edge:
        args = [
            edge, f"--app={url}", "--inprivate",
            f"--user-data-dir={_profile_dir()}",
            f"--window-size={WIN_W},{WIN_H}",
            "--no-first-run", "--no-default-browser-check", "--disable-sync",
            "--disable-background-networking",
            "--disable-features=msImplicitSignin,msEdgeSyncEnabled,msEdgeWelcomePage,EdgeFollowEnabled",
        ]
        flags = 0x08000000 if sys.platform == "win32" else 0
        try:
            return subprocess.Popen(args, creationflags=flags)
        except Exception:
            pass
    webbrowser.open(url)
    return None


def main():
    srv = WebViewServer(host="127.0.0.1", port=0)   # 隨機空閒埠，只綁回環
    srv.start()
    srv.backend.start_load()                        # 背景載入模型
    url = srv.url
    print(f"[{APP_NAME}] {url}")

    try:
        if not run_native_window(url):              # 主要：原生 WebView2 視窗
            proc = open_edge_app(url)               # fallback：Edge --app 無痕
            if proc is not None:
                proc.wait()
            else:
                threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()


if __name__ == "__main__":
    main()
