# 聲音辨識小工具（QwenASR WebView 版）

本地語音辨識字幕生成工具 —— **資料不離開你的電腦**。以 Qwen3-ASR / Whisper(Breeze-ASR-26) 為核心，
音檔、影片、麥克風錄音都能轉成 SRT 字幕。全新 **WebView 介面**（淺色 teal 風格、原生 WebView2 視窗），
支援純 CPU（OpenVINO INT8）與 GPU 加速（CrispASR / Vulkan，NVIDIA・AMD・Intel 通吃）。

> 目標很單純：讓你的同事、你的同學、你的阿嬤，都能免費、離線、在自己的電腦上做語音辨識。

- 介面標題：**聲音辨識小工具**（貓耳耳機圖示）
- 版本：**2.0.0**
- **自 2.0.0 起，WebView 版為唯一維護的版本**；舊的 CustomTkinter 桌面版停留在 `1.0.9`，
  不再更新。兩者可並存於不同資料夾、共用同一份模型；舊版使用者按「檢查更新」會看到
  遷移提示（**不會**覆蓋既有安裝，請解壓到新資料夾直接執行）。
- 舊版桌面版（CustomTkinter）說明文件已移至 [`oldreadme.md`](oldreadme.md)

---

## 主要特色

- 🎧 **音檔 / 影片 → SRT 字幕**：MP3・WAV・FLAC・M4A・OGG 等音訊，及 MP4・MKV・MOV 等影片（自動用 ffmpeg 抽音軌）
- 🎙️ **麥克風錄製轉換**：偵測說話停頓自動分段辨識（非逐字串流），可即時存檔
- 🌊 **真實波形 + 播放頭 + 分段標註**：Suno 式波形，點波形／字幕／區塊即可跳播
- 🎤 **卡拉OK逐字模式**：字級時間軸貫通，播放時逐字高亮歌詞（見下方截圖）
- 🌏 **多語系辨識**：中文、日文、英文等 30 種語系，可自動偵測或鎖定語言
- 🇯🇵 **日語強化**：內建 **Qwen3-ASR-1.7B 日語動漫**特化模型，日文歌詞／台詞辨識明顯較佳；日語輸出保留原生漢字（不誤繁化）
- 🇹🇼 **台灣國語強化（新）**：內建 **TEA-ASR-1.1** 特化模型，台灣用語／中英夾雜辨識更準，且**原生輸出繁體**（限乾淨人聲，見說明）
- 👥 **說話者分離**：可指定人數，SRT 自動標記「說話者1／2…」
- 💬 **辨識提示（參考文字）**：貼入歌詞、關鍵字或背景說明，提升辨識準確度
- 📚 **批次辨識**：一次匯入大量音訊／影片，依序自動辨識
- 🔌 **端點服務**：內建 OpenAI 相容轉錄 API + 手機掃碼上傳頁 + Cloudflare 對外通道
- 🧩 **多推理核心**：CrispASR（GPU：Vulkan／CUDA 可選）／OpenVINO（純 CPU）／chatllm（向下相容），可於「模型」頁切換
- ⚡ **加速版本自動推薦（新）**：偵測你的顯示卡後推薦最合適的 CrispASR build，NVIDIA 用戶可改選 CUDA
- 🎨 **深淺色主題 + 介面縮放 + 多語介面**（繁中／简体／English）
- 📦 **開箱即用**：CrispASR 核心與 VAD 隨包附帶；各模型／ffmpeg 按需自動下載

---

## 介面導覽

### 音檔轉字幕

![音檔轉字幕介面](Readme/webview/01.jpg)

主畫面。拖入或選擇音檔（影片亦可）後：

1. 上方可設定 **語言**（預設自動）、**說話者分離**（可選人數）、**時間軸對齊**（字級時間軸）
2. 可在「**辨識提示**」貼入歌詞／關鍵字，或「讀入 TXT」載入參考文字
3. 點「**▶ 開始轉換**」，進度列即時顯示；完成後可「**開啟輸出資料夾**」或「**字幕存檔**」
4. 下方呈現**真實波形 + 播放頭 + 分段標註**，點波形、字幕條或區塊都能跳播

### 卡拉OK逐字播放

![卡拉OK逐字模式](Readme/webview/07.jpg)

啟用時間軸對齊後，播放時字幕會以**大字逐字高亮**呈現（卡拉OK效果），下一句以淡色預覽，
適合對歌詞、做字幕校對或教學展示。

---

### 批次辨識

![批次辨識介面](Readme/webview/02.jpg)

切到「**批次**」分頁，「**+ 加入檔案**」可多選音訊／影片，「**▶ 全部開始**」依序辨識，
每列即時顯示進度與狀態（待處理／辨識中／完成／失敗），完成後可由「**⋯**」開啟字幕編輯。
批次過程中可隨時再加檔案。適合整批會議錄音、多集 Podcast、課堂錄影一次轉字幕。

---

### 錄製轉換

![錄製轉換介面](Readme/webview/03.jpg)

切到「**錄製**」分頁，選擇麥克風後按下中央麥克風鈕開始。**系統在偵測到說話停頓時才辨識**
（句中短暫停頓不會中斷），結果逐段出現在下方「即時字幕」。可開「**即時存檔**」邊錄邊存，
或錄完按「**儲存字幕**」。再按一次麥克風鈕結束並完成最後一段。

---

### 端點服務（內網／手機上傳）

![端點服務介面](Readme/webview/04.jpg)

切到「**端點**」分頁，開啟「**OpenAI 相容轉錄服務**」後：

- **手機掃碼上傳**：同網段裝置掃 QR 或開啟網址，直接上傳音訊／影片辨識
- **OpenAI 相容 API**：`POST /v1/audio/transcriptions`，可被腳本、OpenWebUI 等串接
- **存取金鑰（等同密碼）**：可顯示／重設，保護你的算力
- **對外臨時網址（Cloudflare）**：一鍵開臨時通道讓非同網段也能上傳（首次會下載 cloudflared 約 25MB）

> ⚠️ 對外網址內含金鑰，任何取得完整網址的人都能上傳、消耗你的 CPU/GPU。用完請立即關閉。

---

### 模型與裝置

![模型與裝置介面](Readme/webview/05.jpg)

切到「**模型**」分頁：

- **系統自檢**：逐核心 × 逐能力列出檔案就緒狀況（綠＝已備、黃＝啟用時自動下載）
- **基礎／進階雙模式（新）**：預設為**基礎** —— 不必懂模型名稱，直接依「你要辨識什麼」挑：

  | 用途 | 建議模型（依偵測到的硬體自動給量化）|
  |------|------|
  | **中文（標準）** | Qwen3-ASR-1.7B（無顯卡時改推純 CPU 的 0.6B）|
  | **繁體中文 TEA（乾淨聲音）** | TEA-ASR-1.1 台灣國語 |
  | **日本語** | Qwen3-ASR-1.7B 日語動漫 |
  | **國台語混合（高速）** | Whisper Breeze-ASR-26 |

  有獨立顯卡給最準的 Q8、只有內顯給較輕的 Q4／Q5。切到**進階**則維持原本的
  「推理核心 ＋ 模型下拉 ＋ 裝置／加速版本／診斷」全手動介面。兩種模式共用同一顆
  載入按鈕與同一份設定，切換不會產生不一致。
- **推理核心**（進階）：可選 **Qwen** 或 **Whisper (Breeze)**；下方「模型」下拉依核心動態切換
- **切換即重啟**：切核心／模型會持久化設定並提示重新啟動（避免就地切換 Vulkan context 造成當機）

- **推理加速版本**：偵測顯示卡後推薦 CrispASR 的 Vulkan／CUDA／CPU build，可自行改選（換版本需重啟）

四張核心卡片：**Qwen · OpenVINO（CPU）**、**CRISPASR（Whisper/Breeze + Qwen + TEA）**、
**Qwen · chatllm（Vulkan · 相容）**、**共用元件（FFmpeg／說話者分離）**。

---

### 設定

![設定介面](Readme/webview/06.jpg)

切到「**設定**」分頁，集中管理偏好：

| 設定項 | 說明 |
|--------|------|
| **介面縮放** | 調整文字與控制項大小 |
| **輸出格式** | 全域預設 SRT 字幕／純文字（影響單檔・批次・端點）|
| **語音偵測靈敏度（VAD）** | 降低閾值減少漏識、提高減少假陽性（預設 0.50）|
| **每段最長秒數（字級對齊）** | 調短可減少長段歌詞的「段尾字漸漸跑掉」漂移；依模型自動夾低（0.6B 30s／1.7B 10s）。CrispASR 走自身視窗不受此設定影響 |
| **簡繁詞彙轉換** | 關閉／台灣用語（s2twp）／標準（s2t）|
| **HuggingFace 鏡像站** | 下載較慢時可改用鏡像（如 hf-mirror.com）|
| **FFmpeg 路徑** | 處理影片音軌用；留空＝自動偵測／需要時下載 |
| **外觀主題** | 淺色／深色／跟隨系統（視窗標題列同步）|
| **介面語言** | 繁體中文／简体中文／English |

底部顯示版本徽章與「檢查更新」（WebView 版檢查更新只開啟發行頁）。

---

## 推理核心與模型

本工具「模型驅動核心」—— 在「模型」頁選哪顆模型，就自動用對應的推理後端。

### CRISPASR（GPU）— 主力核心

以 [CrispASR](https://github.com/CrispStrobe/CrispASR)（whisper.cpp 家族的多後端 runtime，ggml C++）
加速推理，本版使用 **CrispASR v0.8.32** 核心。

**推理加速版本可選**（模型頁「推理加速版本」）：程式會偵測你的顯示卡（Windows 的
`Win32_VideoController` + `nvidia-smi`）後給出推薦，你可自行改選：

| 加速版本 | 下載量 | 適用 |
|---------|-------|------|
| **Vulkan**（預設・建議）| 約 34 MB | NVIDIA／AMD／Intel 通吃，免裝 CUDA／ROCm；無獨顯時自動退回 CPU |
| **CUDA 13** | 約 484 MB | NVIDIA 專用（需 580 以上驅動），自帶 CUDA runtime |
| **CUDA 12** | 約 690 MB | NVIDIA 專用（較舊驅動），自帶 CUDA runtime |
| **CPU** | 約 8 MB | 完全不碰顯卡，驅動有問題時的保底 |
| **CPU 相容版** | 約 8 MB | 給不支援 AVX2 的老處理器（一般 CPU 版閃退時才需要）|

> CrispASR 的 Windows 版沒有 ROCm／SYCL，因此 AMD 與 Intel 顯卡的最佳解就是 Vulkan；
> 只有 NVIDIA 才多一條 CUDA 可選。實測（RTX 5090）Vulkan 與 CUDA 速度接近，故預設仍為
> 體積最小的 Vulkan。換加速版本＝換核心，需重新啟動；模型檔不必重抓。

| 模型 | 用途 |
|------|------|
| **Qwen3-ASR-1.7B Q4 / Q8** | 通用高辨識率，繁中斷句佳 |
| **Qwen3-ASR-1.7B 日語動漫 Q4 / Q8** | 日語／動漫特化，日文歌詞・台詞辨識明顯較佳 |
| **TEA-ASR-1.1 台灣國語 Q8**（新）| 台灣國語／繁體中文／中英夾雜特化，**原生輸出繁體**（免簡繁轉換）。⚠️ 見下方適用範圍 |
| **Whisper Breeze-ASR-26 Q4 / Q5 / Q8** | 繁中／台語特化（large-v2 等級）|
| **qwen3 ForcedAligner GGUF** | 字級時間軸對齊（卡拉OK逐字、精準字幕）|

> **TEA-ASR 的適用範圍**：它是「凍結音訊編碼器 ＋ 解碼器小 LoRA、不到 10 小時公開語料」的
> 輕量微調（見[模型卡](https://huggingface.co/JacobLinCool/TEA-ASR-1.1)）。本專案實測：
> **乾淨人聲**（訪談、質詢、講座）辨識最準且原生吐繁體；但**背景配樂較重的段落或歌曲**
> 可能整段無輸出（模型直接結束生成）。此行為在未量化的 F16 版本同樣出現，**與量化無關**。
> 音樂類素材請改用 Qwen3-ASR-1.7B（通用）或 Whisper Breeze。程式在模型下拉旁會顯示同樣的提醒。

### Qwen · OpenVINO（純 CPU）

免顯卡、免驅動，純 CPU 即可執行。**0.6B**（輕量，開箱即用）與 **1.7B INT8 KV-Cache**（更準，按需下載）。
適合沒有獨顯的環境。

### Qwen · chatllm（Vulkan · 向下相容）

保留供既有使用者向下相容（曾用過 chatllm 或自桌面版沿用者）。核心二進位**不隨安裝包附帶**，
只有機器上實際備有 `libchatllm.dll` 時才會在清單中現身。

---

## 日語辨識強化（webview 0.2 新增）

新增 **Qwen3-ASR-1.7B 日語動漫**特化模型（[cstr/qwen3-asr-1.7b-ja-anime-GGUF](https://huggingface.co/cstr/qwen3-asr-1.7b-ja-anime-GGUF)），
針對日語／動漫語音微調，日文歌詞、台詞辨識明顯優於標準模型。實測日文歌曲對比：

| 標準 Qwen3-ASR-1.7B | 日語動漫版 |
|---------------------|-----------|
| An なまぶしさ（英日混雜）| **あんな眩しさ**（正確漢字）|
| 輪切りかけたコヨーテ（胡言）| **わかりかけた**（較合理）|
| どこ。（截斷）| **どこかで会えるように**（完整）|

同時修正：**語言選日語時跳過 OpenCC 繁化**（原本會把日文漢字 会／静／図 誤轉成繁中字形 會／靜／圖），
保留日文原生字形；中文／台語辨識維持繁化不變。

---

## 命令列模式（給 Agent／腳本）

`QwenASR-WebView.exe` **帶子命令執行時是無頭 CLI**：不開視窗、跑完即退，
stdout 只放結果、進度走 stderr，適合被 AI Agent 或批次腳本直接呼叫。
不帶子命令則照舊開圖形介面。

```bash
QwenASR-WebView.exe status --json                  # 核心／模型／加速版本狀態
QwenASR-WebView.exe profiles --json                # 可用模型（含硬體判斷）
QwenASR-WebView.exe transcribe audio.mp3 --json    # 轉錄 → 乾淨 JSON
QwenASR-WebView.exe apply fixed.json --base work.json -o final.srt
```

`transcribe` 的 JSON：

```json
{"input":"...","ok":true,"srtPath":"...\subtitles\audio.srt",
 "segments":[{"i":1,"start":0.48,"end":1.84,"text":"再看立法院的質詢"}],
 "text":"整段連續逐字稿"}
```

常用選項：`-l Chinese`（指定語言）、`--profile zh|tea|ja|twmix`（依用途換模型，
**只影響這次執行**、不改 GUI 設定）、`--hint "人名、術語"`、`--diarize [--speakers N]`、
`-o out.srt -f srt|txt|json`。退出碼 `0` 成功／`1` 轉錄失敗／`2` 參數錯誤。

### 為什麼要有 `apply`

讓 Agent 修字幕時**只准改 `text`**。若直接把整份 SRT 交給語言模型重寫，
時間戳很容易被順手改壞且難以察覺。正確流程是三步：

1. `transcribe ... -o work.json -f json` 取得 segments
2. Agent 只改每個物件的 `text`，存成 `fixed.json`
3. `apply fixed.json --base work.json -o final.srt`

`--base` 以**原始檔的時間軸為準**、只依 `i` 回填文字 —— 就算 Agent 動到時間也會被蓋回去。
沒帶 `--base` 且時間軸看起來退化（全零／倒退）時，CLI 會在 stderr 出聲警告。

> 摘要／改寫由 Agent 自己做 —— 本工具只負責交出準確的逐字稿。

### Claude Code Skill

repo 內附 `.claude/skills/qwenasr/SKILL.md`，把上述用法（含各 profile 的
適用情境與校正守則）教給 Agent。要讓任何專案的 Agent 都能用，複製到
`~/.claude/skills/qwenasr/` 即可。

---

## 安裝與更新

### 免安裝 EXE（推薦）

至 [GitHub Releases](https://github.com/dseditor/QwenASRMiniTool/releases) 下載 **QwenASR-WebView** 版壓縮包，
解壓後執行 `QwenASR-WebView.exe`。首次啟動：

- 已內含 CrispASR 核心與 VAD，選定模型後若尚未下載會提示，按「下載並載入」即自動下載
- 影片辨識首次會提示下載 ffmpeg；說話者分離、字級對齊模型皆按需下載

> 💡 若解壓後 Windows 阻擋執行，於 exe 按右鍵 →「內容」→ 勾選「解除封鎖」即可（Mark-of-the-Web）。

### 原始碼執行 / 自行編譯

```bash
git clone https://github.com/dseditor/QwenASRMiniTool.git
cd QwenASRMiniTool
# 建立虛擬環境並安裝需求後：
python app_webview.py          # 啟動 WebView 介面
# 或編譯為單一 EXE：
build_webview.bat              # 產出 dist2\QwenASR-WebView\QwenASR-WebView.exe
```

---

## 系統需求

| 項目 | CPU 模式（OpenVINO）| GPU 模式（CrispASR / Vulkan）|
|------|--------------------|------------------------------|
| 作業系統 | Windows 10 / 11（64-bit）| Windows 10 / 11（64-bit）|
| RAM | 6 GB（峰值約 4.8 GB）| 8 GB 以上 |
| 硬碟 | 2 GB（0.6B 約 1.2 GB）| 依模型：Qwen 1.7B GGUF Q8 約 2.5 GB |
| GPU | 不需要 | Vulkan 1.2+（NVIDIA / AMD / Intel）|
| WebView2 | Windows 11 內建；Windows 10 需安裝 Edge WebView2 Runtime | 同左 |

> ⚠️ 安裝路徑請使用**全英文**（例如 `C:\QwenASR`），含中文字元可能無法正常執行。

---

## 模型來源

| 項目 | 連結 |
|------|------|
| Qwen3-ASR-1.7B GGUF（CrispASR/Vulkan）| [cstr/qwen3-asr-1.7b-GGUF](https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF) |
| **Qwen3-ASR-1.7B 日語動漫 GGUF** | [cstr/qwen3-asr-1.7b-ja-anime-GGUF](https://huggingface.co/cstr/qwen3-asr-1.7b-ja-anime-GGUF) |
| **TEA-ASR-1.1 台灣國語 GGUF**（新）| [dseditor/TEA-ASR-1.1-CrispASR-GGUF](https://huggingface.co/dseditor/TEA-ASR-1.1-CrispASR-GGUF)（原始模型：[JacobLinCool/TEA-ASR-1.1](https://huggingface.co/JacobLinCool/TEA-ASR-1.1)）|
| Whisper Breeze-ASR-26 GGML | [phate334/Breeze-ASR-26-GGML](https://huggingface.co/phate334/Breeze-ASR-26-GGML) |
| qwen3 ForcedAligner GGUF | [cstr/qwen3-forced-aligner-0.6b-GGUF](https://huggingface.co/cstr/qwen3-forced-aligner-0.6b-GGUF) |
| 0.6B OpenVINO INT8（主／備）| [dseditor](https://huggingface.co/dseditor/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO) ／ [Echo9Zulu](https://huggingface.co/Echo9Zulu/Qwen3-ASR-0.6B-INT8_ASYM-OpenVINO) |
| 1.7B OpenVINO INT8 KV-Cache | [dseditor/Qwen3-ASR-1.7B-INT8_OpenVINO](https://huggingface.co/dseditor/Qwen3-ASR-1.7B-INT8_OpenVINO) |
| 原始 PyTorch 模型 | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) ／ [1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| VAD 模型 | [snakers4/silero-vad v4.0](https://github.com/snakers4/silero-vad) |
| 說話者分離模型 | [altunenes/speaker-diarization-community-1-onnx](https://huggingface.co/altunenes/speaker-diarization-community-1-onnx) |

---

## 開發者說明

### 架構（WebView 版）

```
app_webview.py          # WebView 版進入點（原生 WebView2 視窗 + 本機 HTTP server）
webview_server.py       # 本機 stdlib HTTP 伺服器（serve webview/ + /api，含 Host 白名單防 rebinding）
webview_backend.py      # WebView 共用後端邏輯（核心/模型目錄、載入、自檢、端點）
webview/                # 前端（HTML / CSS / JS，i18n），python·端點·EXE 三版共用
  index.html  css/app.css  js/{app,bridge,i18n}.js
app.py                  # 引擎核心 ASREngine（OpenVINO）+ 既有桌面版共用邏輯
crisp_engine.py         # CrispASR（Vulkan）推理引擎：Whisper/Breeze + Qwen3-ASR GGUF 子程序包裝
chatllm_engine.py       # chatllm（Vulkan）推理引擎（向下相容）
subtitle_lines.py       # 全引擎共用字幕分行（字級時間軸→字幕行，標點/空白切 + 孤兒合併）
fa_aligner.py           # ForcedAligner 字級時間軸（OpenVINO/chatllm 共用）
subtitle_editor.py      # 字幕驗證與編輯
batch_tab.py            # 批次排程邏輯
diarize.py              # 說話者分離引擎（外部 ONNX，兩階段聚類，不依賴 torch）
audio_io.py             # 音訊讀取（16k mono，不依賴 librosa）
processor_numpy.py      # 純 numpy Mel / BPE 處理器
api_server.py           # OpenAI 相容轉錄端點（純標準庫）
cf_tunnel.py            # Cloudflare 臨時通道
downloader.py           # 模型完整性檢查與按需下載（含 LFS pointer 偵測、HF 鏡像）
version.py / updater.py # 版本單一事實來源 / 自動更新
build_webview.bat       # PyInstaller onefile 打包（WebView EXE）
crispasr/               # CrispASR 核心（exe + dll，隨包）；模型 *.bin/*.gguf 按需下載
ov_models/              # OpenVINO 模型 / VAD / Qwen GGUF / 說話者分離 ONNX
```

### 為什麼是 WebView + onefile

- **WebView**：改用本機 stdlib HTTP server + 原生 WebView2 視窗（棄 pywebview 的 pythonnet 遞迴 bug），前端純網頁、跨三版共用。
- **onefile**：onedir 的散落 DLL 在解壓時會沾上 Mark-of-the-Web，.NET 拒載 → 退回 Edge。onefile 執行時解壓 DLL 到暫存（無 MOTW），使用者只需解封鎖一個 exe。

### 第三方致謝

- **[CrispASR](https://github.com/CrispStrobe/CrispASR)**（CrispStrobe）— Vulkan 多後端 ASR runtime（whisper / qwen3），本專案使用其官方 Windows 預編譯版。
- **[chatllm.cpp](https://github.com/foldl/chatllm.cpp)**（foldl，MIT）— Vulkan GPU 推理後端（向下相容），使用官方預編譯二進位。
- **[Breeze-ASR-26](https://huggingface.co/phate334/Breeze-ASR-26-GGML)**（phate334）— 繁中／台語特化 Whisper 模型。
- **[FFmpeg](https://ffmpeg.org/)**（GPL）— 影片音軌提取，按需下載，二進位不內嵌。
- **[silero-vad](https://github.com/snakers4/silero-vad)** — 語音活動偵測。

---

## 授權

本專案程式碼以 **MIT** 授權釋出。模型權重與第三方預編譯二進位依各自來源的授權條款。
