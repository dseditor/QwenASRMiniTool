"""version.py — 應用程式版本與更新來源設定（單一事實來源）

此檔同時被 app.py / app-gpu.py / setting.py / updater.py 引用。
打包時務必以 --add-data 一併納入 EXE，讓自動更新能正確比對版本。

版本規則：
    語意化版本 MAJOR.MINOR.PATCH。
    每次發佈新編譯版時，先把 __version__ 往上加（例如 1.0.6 → 1.0.7），
    再到 GitHub 建立同名 tag 的 Release，並上傳整包 ZIP 資產。
"""
from __future__ import annotations

# 本次編譯版本（dist2）。1.0.8：模型分頁重構(引擎/裝置/模型/路徑/下載/CPU 集中)、
# 語系移至設定頁並置頂、標題列狀態摘要、Streamlit 退役(網頁服務統一由端點提供)、
# 端點網頁重設計+即時錄音(停頓自動上傳)、端點 QR 右側面板/外網優先/可下載、
# audio_io 取代 librosa(修 numpy 2.4 + numba 衝突，瘦身)、句尾字消失修復
# (去整秒裁切 + 尾段 0.35s + 瀏覽器停頓門檻 2.2s)。最新已發佈 Release 為 1.0.7。
__version__ = "1.0.8"

# 自動更新來源：GitHub repo（owner/name）
GITHUB_REPO = "dseditor/QwenASRMiniTool"

# GitHub Releases API（latest 端點，回傳最新「非預發佈」版本）
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# 發行頁（供「前往下載頁」按鈕使用）
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"
