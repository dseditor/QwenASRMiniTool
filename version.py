"""version.py — 應用程式版本與更新來源設定（單一事實來源）

此檔同時被 app.py / app-gpu.py / setting.py / updater.py 引用。
打包時務必以 --add-data 一併納入 EXE，讓自動更新能正確比對版本。

版本規則：
    語意化版本 MAJOR.MINOR.PATCH。
    每次發佈新編譯版時，先把 __version__ 往上加（例如 1.0.6 → 1.0.7），
    再到 GitHub 建立同名 tag 的 Release，並上傳整包 ZIP 資產。
"""
from __future__ import annotations

# 本次編譯版本（dist2）。1.0.9：新增 Whisper/Breeze-ASR-26 推理核心
# (CrispASR / Vulkan，-nf -bo1 達 ~35x，繁中/台語特化)、qwen3 ForcedAligner
# GGUF 字級時間軸對齊、核心切換 Vulkan context 防當機修正、字幕分行全引擎統一
# (subtitle_lines，標點/空白邊界切 + MAX_CHARS 保護 + 孤兒行合併)。
# 最新已發佈 Release 為 1.0.8。
__version__ = "1.0.9"

# WebView 版獨立版本字串（與 CTk 桌面版的語意化 __version__ 分流）。
# WebView 介面為全新重構，從 0.1 重新起算；顯示於 WebView 設定頁的版本徽章。
# （CTk 桌面版仍沿用上面的 __version__ 做語意化更新比對，互不干擾。）
#
# ── WebView 0.1 更新彙整（本版所有變更集中於此）────────────────────────────
#   介面：全新淺色 teal WebView 介面（本機 stdlib HTTP server + 原生 WebView2 窗），
#         python / 端點 / 獨立 EXE 三版共用同一前端。標題「聲音辨識小工具」+ 貓耳耳機圖示。
#   核心：CrispASR(Vulkan) 為主，預設「Qwen3-ASR-1.7B Q4」；OpenVINO(CPU) 共存；
#         chatllm 保留向下相容（有檔才現身、不隨包附帶）。裝置偵測改走 crispasr --diagnostics。
#   流程：開機若選擇的模型未下載 → 停模型頁不自動抓，待使用者按「下載並載入」（可改 Whisper），
#         首次免重啟就地下載載入；已下載則自動載入並進語音轉文字。
#   功能：音檔 / 批次 / 錄製（麥克風選擇＋即時存檔）/ 端點（QR＋Cloudflare）/ 模型自檢。
#   外觀：i18n（繁中／简体／English）、深淺色主題（視窗標題列同步）、介面縮放，全部實裝。
#   打包：crispasr 核心隨包、VAD onnx 隨包；ffmpeg / chatllm 按需下載或外帶（不入安裝包）。
WEBVIEW_VERSION = "webview 0.1"

# 自動更新來源：GitHub repo（owner/name）
GITHUB_REPO = "dseditor/QwenASRMiniTool"

# GitHub Releases API（latest 端點，回傳最新「非預發佈」版本）
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# 發行頁（供「前往下載頁」按鈕使用）
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"
