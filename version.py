"""version.py — 應用程式版本與更新來源設定（單一事實來源）

此檔同時被 app.py / app-gpu.py / setting.py / updater.py 引用。
打包時務必以 --add-data 一併納入 EXE，讓自動更新能正確比對版本。

版本規則：
    語意化版本 MAJOR.MINOR.PATCH。
    每次發佈新編譯版時，先把 __version__ 往上加（例如 1.0.6 → 1.0.7），
    再到 GitHub 建立同名 tag 的 Release，並上傳整包 ZIP 資產。
"""
from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════
# 2.0.0 起：WebView 版成為唯一主線，CTk 桌面版不再更新
# ══════════════════════════════════════════════════════════════════════
#   1.0.9 之前有兩條版本線：CTk 走語意化 __version__、WebView 走 "webview 0.x"
#   字串，WebView 的 Release 一律 prerelease 以免搶走 CTk 的 releases/latest。
#
#   ⚠ 為什麼一定要跳號到 2.0.0（而不是繼續 webview 0.2.x）：
#     updater.parse_version() 用的是 re.search（非 match），"webview 0.2.2" 會被
#     解析成 (0,2,2)，**比 1.0.9 小**。也就是說即使把它設成 latest，每個 CTk
#     使用者按「檢查更新」都只會看到「已是最新版」，永遠不會知道新版存在——
#     不是報錯，是靜默的死路。跳到 2.0.0 才解得出大於 1.0.9 的版本，主版號跳號
#     也正確表達「這是不相容的新一代」。
#
#   兩條線自此合流：__version__ 與 WEBVIEW_VERSION 同值，WebView 的 Release 改
#   為正式版（prerelease=false）成為 releases/latest。CTk 使用者按更新會收到
#   「遷移提示」而非自動覆蓋（見 updater.is_migration_package）。
__version__ = "2.0.0"

# CTk 桌面版最後一個發佈版本為 1.0.9（新增 Whisper/Breeze-ASR-26 推理核心、
# qwen3 ForcedAligner 字級對齊、核心切換防當機、字幕分行全引擎統一）。
# 2.0.0 起 CTk 不再更新，上面的 __version__ 由兩線共用。
_CTK_LAST_VERSION = "1.0.9"

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
#
# ── WebView 0.2 更新彙整（在 0.1 基礎上）──────────────────────────────────────
#   核心：CrispASR 核心升級 v0.7.1→v0.8.8（GPU 偵測/日語 qwen3 後端驗證）。
#   日語：新增 Qwen3-ASR-1.7B 日語動漫特化模型（cstr/ja-anime，Q4/Q8），日文歌詞/
#         台詞辨識明顯較佳；語言選日語時跳過 OpenCC 繁化，保留日文原生漢字。
#   體驗：卡拉OK逐字模式（字級時間軸貫通、播放逐字高亮）、真實波形+播放頭+分段標註、
#         設定頁「每段最長秒數」滑桿（減少長段字級段尾漂移）。
#
# ── WebView 0.2.1 更新彙整（hotfix）──────────────────────────────────────────
#   修復：1.7B INT8 (CPU/OpenVINO) 無法載入——ce1f8b0 重構誤刪 KV-cache 實作，
#         導致選 1.7B 實際讀取 0.6B 目錄（只裝 1.7B 直接報錯、兩者皆裝則默默
#         跑 0.6B）。復原 prefill+decode KV-cache 推理並改掛勾式覆寫，1.7B 同步
#         取得 cpu_threads／FA 字級對齊等後續功能。
#   改善：說話者分離聚類改版——長段落切 3 秒子窗逐窗提聲紋、自動人數改
#         silhouette 選擇＋雙重單人防護，修「同人被拆／異人被併」；短段落
#         不再丟棄（修 OpenVINO 路徑短句漏字）。
#
# ── WebView 0.2.2 更新彙整 ──────────────────────────────────────────────────
#   核心：CrispASR 升級 v0.8.8→v0.8.32。新增「推理加速版本」可選——程式以
#         Win32_VideoController + nvidia-smi 偵測顯示卡後給推薦（一律預設 Vulkan，
#         34MB 且實測與 CUDA 速度接近），NVIDIA 使用者可自行改選 CUDA 12/13；
#         另備 CPU 版與無 AVX2 的 CPU 相容版。Windows 無 ROCm/SYCL，故 AMD／Intel
#         的最佳解就是 Vulkan。換版本＝換核心，需重啟；模型檔不必重抓。
#   模型：新增 TEA-ASR-1.1 台灣國語特化（Qwen3-ASR-1.7B 微調，原生輸出繁體與台灣
#         用語、中英夾雜佳）。HF 上現成的 mmproj 拆檔版 CrispASR 讀不了，故本專案
#         以官方轉換器重轉為單檔融合式 GGUF 上架（dseditor/TEA-ASR-1.1-CrispASR-GGUF）。
#         ⚠ 該模型限乾淨人聲，背景配樂較重的段落或歌曲可能整段無輸出（UI 已標示）。
#   介面：模型頁改為「基礎／進階」雙模式，預設基礎——依用途（中文標準／繁體中文
#         TEA／日本語／國台語混合）並依偵測到的硬體自動給建議模型與量化；進階維持
#         原本的核心＋模型下拉與裝置／加速版本／診斷。兩模式共用同一載入流程。
#   自動化：新增無頭 CLI（帶子命令執行即不開視窗）——transcribe／apply／status／
#         profiles，stdout 只放乾淨 JSON、進度走 stderr，供外部 Agent 與腳本呼叫；
#         `apply --base` 讓校正只改文字、時間軸以原始檔為準。附 Claude Code skill。
#   修復：TEA 這類「原生輸出繁體」的模型與 qwen3 CTC 對齊器不相容——對齊器的類別表
#         建立於簡體輸出，遇繁體字會輸出未映射類別 ID，被渲染成私用區亂碼並使時間軸
#         對歪。改為對此類模型自動停用對齊器並以等時內插補回字級斷句；另在字幕輸出
#         總出口統一濾除私用區字元，任何引擎的同類亂碼都不再流入 SRT／JSON。
#
# ── 2.0.0（原 webview 0.2.2，改為主線版本號）─────────────────────────────────
#   內容同上方 webview 0.2.2 條目；版本號改為 2.0.0 使 WebView 成為唯一主線，
#   Release 改為正式版並成為 releases/latest。CTk 使用者按「檢查更新」會偵測到
#   2.0.0 > 1.0.9，但因套件不含其主執行檔，updater 會顯示遷移說明並導向發行頁，
#   **不會**自動覆蓋既有安裝（見 updater.is_migration_package / apply_update）。
#
# 兩條線已合流：此值與 __version__ 同步，保留常數名是為了不動既有引用
# （webview_backend._app_version / setting.py 版本徽章）。
WEBVIEW_VERSION = __version__

# 自動更新來源：GitHub repo（owner/name）
GITHUB_REPO = "dseditor/QwenASRMiniTool"

# GitHub Releases API（latest 端點，回傳最新「非預發佈」版本）
GITHUB_API_LATEST = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

# 發行頁（供「前往下載頁」按鈕使用）
GITHUB_RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"
