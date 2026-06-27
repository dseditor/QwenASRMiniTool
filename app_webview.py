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

import ctypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from ctypes import wintypes
from pathlib import Path

from proc_guard import setup_kill_on_close_job
from webview_server import WebViewServer

APP_NAME = "聲音辨識小工具"
WIN_W, WIN_H = 1180, 820


# ════════════════════════════════════════════════════════
# 標題列配色：配合淺色頁面主題 → 白底黑字
# ════════════════════════════════════════════════════════
#   原生視窗（WebView2）與 Edge --app 視窗的標題列由 OS（DWM）繪製，預設會
#   跟隨系統的深／淺色設定 —— 系統若為深色，標題列就是深底白字，與本程式
#   固定的淺色頁面主題不搭。Windows 11（build 22000+）提供 DWM 屬性可直接
#   指定標題列配色，故在視窗顯示後呼叫 DwmSetWindowAttribute 強制：
#     • DWMWA_USE_IMMERSIVE_DARK_MODE(20)=0 → 關閉深色模式（淺色標題列）
#     • DWMWA_CAPTION_COLOR(35)=白          → 標題列底色（COLORREF 0x00BBGGRR）
#     • DWMWA_TEXT_COLOR(36)=黑             → 標題列文字／圖示色
#   兩種視窗（原生 / Edge fallback）的標題都是頁面 <title>「聲音辨識」，故可用
#   同一支以視窗標題尋找 HWND 的輔助函式套用。失敗（舊版 Windows / 非 win32）
#   一律靜默略過，不影響視窗開啟。
_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36
_COLOR_WHITE = 0x00FFFFFF        # COLORREF：白底（淺色標題列底）
_COLOR_BLACK = 0x00000000        # COLORREF：黑字
_COLOR_DARK_BG = 0x0026201E      # COLORREF 0x00BBGGRR ← #1E2026（深色面板底，與 CSS 一致）

_initial_theme = "light"         # main() 啟動時由設定填入，供視窗裝飾 worker 取用初始深淺


def _os_dark() -> bool:
    """讀 Windows 個人化設定判斷系統是否為深色（AppsUseLightTheme==0）。"""
    if sys.platform != "win32":
        return False
    try:
        import winreg
        with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize") as k:
            v, _ = winreg.QueryValueEx(k, "AppsUseLightTheme")
            return v == 0
    except Exception:
        return False


def _resolve_dark(theme: str) -> bool:
    """外觀偏好（light/dark/system）→ 是否深色。"""
    if theme == "dark":
        return True
    if theme == "light":
        return False
    return _os_dark()              # system → 跟隨 OS


def _set_titlebar(hwnd: int, dark: bool) -> bool:
    """對 HWND 套用標題列配色：dark→深底白字、light→白底黑字。回傳是否設成底色。"""
    if not hwnd or sys.platform != "win32":
        return False
    try:
        dwm = ctypes.windll.dwmapi
    except Exception:
        return False

    def _set(attr: int, value: int) -> int:
        v = ctypes.c_int(value)
        return dwm.DwmSetWindowAttribute(
            wintypes_hwnd(hwnd), attr, ctypes.byref(v), ctypes.sizeof(v))

    cap = _COLOR_DARK_BG if dark else _COLOR_WHITE
    txt = _COLOR_WHITE if dark else _COLOR_BLACK
    ok = False
    try:
        _set(_DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)
    except Exception:
        pass
    try:
        if _set(_DWMWA_CAPTION_COLOR, cap) == 0:        # S_OK
            ok = True
        _set(_DWMWA_TEXT_COLOR, txt)
    except Exception:
        pass
    return ok


def apply_titlebar_theme(theme: str):
    """供 webview_backend 回呼：依新外觀設定即時切換視窗標題列深淺（兩種視窗皆套）。"""
    dark = _resolve_dark(theme or "light")
    hwnd = _find_app_hwnd()
    if hwnd:
        _set_titlebar(hwnd, dark)


def wintypes_hwnd(hwnd: int):
    """把 Python int 包成 DWM API 接受的 HWND（c_void_p）。"""
    return ctypes.c_void_p(int(hwnd))


def _find_app_hwnd() -> int:
    """以視窗標題（APP_NAME）尋找頂層視窗 HWND；找不到回 0。"""
    if sys.platform != "win32":
        return 0
    try:
        return int(ctypes.windll.user32.FindWindowW(None, APP_NAME) or 0)
    except Exception:
        return 0


# ────────────────────────────────────────────────────────
# 視窗 / 工作列圖示（貓耳毛絨小耳機）
# ────────────────────────────────────────────────────────
#   為何需要程式化設定：PyInstaller 的 --icon 只內嵌進「打包後的 EXE」，
#   開發模式 `python app_webview.py` 的宿主行程是 python.exe → 工作列顯示
#   python 預設圖示。pywebview 6.x 也沒有可靠的 per-window 圖示 API。故改用
#   Win32 `WM_SETICON` 在視窗建立後把 icon.ico 套上去（標題列小圖 + 工作列大圖），
#   並設定獨立的 AppUserModelID，讓工作列不與 python.exe 共用同一顆按鈕／圖示。
_APP_USER_MODEL_ID = "dseditor.QwenASR.VoiceTool"
_WM_SETICON = 0x0080
_ICON_SMALL, _ICON_BIG = 0, 1
_IMAGE_ICON = 1
_LR_LOADFROMFILE, _LR_DEFAULTSIZE = 0x00000010, 0x00000040


def set_app_user_model_id():
    """讓 Windows 把本行程視為獨立 App（工作列圖示不跟 python.exe 綁在一起）。"""
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(_APP_USER_MODEL_ID)
    except Exception:
        pass


def _resolve_icon_path() -> Path | None:
    """找出 icon.ico：開發模式在 assets/；凍結模式在 _MEIPASS／exe 旁。"""
    cands = []
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            cands.append(Path(meipass) / "icon.ico")
        cands.append(Path(sys.executable).parent / "icon.ico")
    cands.append(Path(__file__).resolve().parent / "assets" / "icon.ico")
    for c in cands:
        if c and c.exists():
            return c
    return None


def _set_window_icon(hwnd: int, ico: Path) -> bool:
    """以 WM_SETICON 把 icon.ico 套到視窗（同時更新標題列小圖與工作列大圖）。"""
    if not hwnd or sys.platform != "win32":
        return False
    try:
        u = ctypes.windll.user32
        # 設定 restype/argtypes：HICON 是 64 位 handle，未指定會被 ctypes 截成 int
        # → handle 失效。LoadImageW 回 HANDLE、SendMessageW 收 HWND/WPARAM/LPARAM。
        u.LoadImageW.restype = wintypes.HANDLE
        u.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                 ctypes.c_int, ctypes.c_int, wintypes.UINT]
        u.SendMessageW.restype = ctypes.c_ssize_t
        u.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                   wintypes.WPARAM, wintypes.LPARAM]
        path = str(ico)
        big = u.LoadImageW(None, path, _IMAGE_ICON, 0, 0,
                           _LR_LOADFROMFILE | _LR_DEFAULTSIZE)        # 大圖（工作列）
        small = u.LoadImageW(None, path, _IMAGE_ICON, 16, 16, _LR_LOADFROMFILE)  # 小圖（標題列）
        h = wintypes_hwnd(hwnd)
        if big:
            u.SendMessageW(h, _WM_SETICON, _ICON_BIG, big)
        if small:
            u.SendMessageW(h, _WM_SETICON, _ICON_SMALL, small)
        return bool(big or small)
    except Exception:
        return False


def _decorate_window_async():
    """背景輪詢，待視窗建立後套用淺色標題列 + 程式圖示（原生 / Edge 皆適用）。

    視窗建立有延遲，且 DWM／WM_SETICON 都需 HWND 存在後才能套 —— 故短輪詢，
    兩者皆完成（或逾時）即停。獨立 daemon 緒執行，不阻塞主流程。
    """
    ico = _resolve_icon_path()
    dark = _resolve_dark(_initial_theme)

    def worker():
        themed = False
        iconed = ico is None          # 無 icon 檔則視為「已處理」（只做標題列）
        for _ in range(40):           # 最多 ~6 秒（40 × 0.15s）
            hwnd = _find_app_hwnd()
            if hwnd:
                if not themed:
                    themed = _set_titlebar(hwnd, dark)
                if not iconed:
                    iconed = _set_window_icon(hwnd, ico)
                if themed and iconed:
                    return
            time.sleep(0.15)
    threading.Thread(target=worker, name="window-decorate", daemon=True).start()


# ════════════════════════════════════════════════════════
# 前置偵測：系統是否具備 WebView2 Runtime（原生視窗的前提）
# ════════════════════════════════════════════════════════
#   為何需要：缺 WebView2 Runtime 時，pywebview 仍會「開出視窗」但 WebView2
#   控制項初始化失敗 —— edgechromium 只記 log 後 return，**不丟例外、不關窗**，
#   於是 webview.start() 既不返回也不報錯 → 我們的 Edge fallback 永遠不會觸發，
#   使用者只看到一個空白視窗（像整個程式壞掉）。故啟動前先用官方註冊表鍵判定，
#   不具備就直接走 Edge --app（純 Chromium，不需 WebView2/.NET）。
#   （缺 .NET 的情況則由 run_native_window 內的例外處理乾淨退回 Edge。）
_WEBVIEW2_CLIENT = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"


def webview2_available() -> bool:
    """偵測系統是否已安裝 WebView2 Runtime（Evergreen 或 per-user）。"""
    if sys.platform != "win32":
        return False
    import winreg
    for hive, path in (
        (winreg.HKEY_LOCAL_MACHINE,
         rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLIENT}"),
        (winreg.HKEY_LOCAL_MACHINE,
         rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLIENT}"),
        (winreg.HKEY_CURRENT_USER,
         rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{_WEBVIEW2_CLIENT}"),
    ):
        try:
            with winreg.OpenKey(hive, path) as k:
                pv, _ = winreg.QueryValueEx(k, "pv")
                if pv and pv not in ("", "0.0.0.0"):
                    return True
        except OSError:
            continue
    return False


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
        # 允許下載：原生 WebView2 預設 ALLOW_DOWNLOADS=False 會「靜默取消」<a download>，
        # 導致「字幕存檔」按鈕沒有任何反應、也不跳存檔視窗。開啟後 edgechromium 會彈
        # 原生 Save 視窗讓使用者選位置。
        try:
            webview.settings['ALLOW_DOWNLOADS'] = True
        except Exception:
            pass
        webview.create_window(
            APP_NAME, url=url,
            width=WIN_W, height=WIN_H, min_size=(960, 680),
        )
        _decorate_window_async()   # 視窗顯示後背景套用白底黑字標題列 + 程式圖示
        webview.start()            # 阻塞至視窗關閉（在主執行緒）
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
            proc = subprocess.Popen(args, creationflags=flags)
            _decorate_window_async()   # Edge --app 視窗同樣套白底黑字標題列 + 圖示
            return proc
        except Exception:
            pass
    webbrowser.open(url)
    return None


def main():
    # 先把自己綁進 Job Object：此後 start_load()/轉錄衍生的 crispasr.exe、
    # chatllm main.exe(FA) 等子程序都會自動納入同一 Job，關窗時連帶被殺。
    # handle 由 proc_guard 模組級變數持有（生命週期 == 行程），無須在此保管。
    setup_kill_on_close_job()

    # 設定獨立 AppUserModelID（要在開任何視窗前）→ 工作列圖示不跟 python.exe 共用。
    set_app_user_model_id()

    srv = WebViewServer(host="127.0.0.1", port=0)   # 隨機空閒埠，只綁回環
    srv.start()
    # 視窗外觀：讀持久化外觀偏好決定初始標題列深淺；註冊回呼，讓 UI 切換主題時
    # 視窗標題列即時跟著深/淺。
    global _initial_theme
    try:
        _initial_theme = srv.backend.persisted_theme()
        srv.backend.set_theme_callback(apply_titlebar_theme)
    except Exception:
        pass
    # 自動載入策略：只有「目前選擇的模型已下載」時才自動載入（快速、無下載）。
    # 首次／模型未下載 → 不自動觸發下載，停在模型頁等使用者選好（可改 Whisper 等）
    # 並按「下載並載入模型」才開始下載。避免一開啟就硬抓預設模型。
    try:
        if srv.backend.selected_model_present():
            srv.backend.start_load()
        else:
            print(f"[{APP_NAME}] 選擇的模型尚未下載 → 等使用者於模型頁確認後再下載。")
    except Exception:
        srv.backend.start_load()                     # 保底：判斷失敗仍嘗試載入
    url = srv.url
    print(f"[{APP_NAME}] {url}")

    # 只有在系統具備 WebView2 Runtime 時才試原生視窗（否則會卡在空白窗，
    # 見 webview2_available 註解）；缺則直接走 Edge --app，不需 WebView2/.NET。
    has_wv2 = webview2_available()
    if not has_wv2:
        print(f"[{APP_NAME}] 未偵測到 WebView2 Runtime，改用 Edge --app 視窗。")
    native_ok = has_wv2 and run_native_window(url)
    try:
        if not native_ok:                           # fallback：Edge --app 無痕 → 預設瀏覽器
            proc = open_edge_app(url)
            if proc is not None:
                proc.wait()
            else:
                threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        # 視窗已關 → 立即收網。先停 HTTP server（釋放埠），再硬退出行程。
        # 為什麼用 os._exit 而非正常 return：chatllm 走常駐 libchatllm.dll，
        # Vulkan context 刻意永不釋放（見 vulkan-dual-context-crash）；正常
        # 直譯器關閉會去清理這顆 DLL → 可能卡死成殭屍行程、持續佔顯存。
        # os._exit 跳過 atexit/GC/DLL 卸載，由 OS 直接回收行程與 GPU 資源；
        # 同一瞬間 Job handle 關閉 → 殘餘子程序一併被終止。
        try:
            srv.stop()
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
