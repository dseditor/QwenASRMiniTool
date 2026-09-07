---
name: qwenasr
description: 用本機「聲音辨識小工具」(QwenASR) 把音訊／影片轉成字幕，並在取得逐字稿後做校正、摘要或改寫。當使用者說「轉錄」「上字幕」「聽打」「這段錄音講什麼」「幫我做逐字稿」「校正字幕」「摘要這段音檔」，或給了 .mp3/.wav/.m4a/.mp4 等檔案要求處理內容時使用。全程在本機執行，音訊不外傳。
---

# QwenASR — 本機語音辨識（Agent 用法）

`QwenASR-WebView.exe` 帶子命令執行時是**無頭 CLI**：不開視窗、跑完即退、
stdout 只有結果、進度走 stderr。適合直接在 Bash 工具裡呼叫。

## 找到執行檔

依序尋找，找到第一個就用（**請用絕對路徑呼叫**）：

1. 使用者明講的路徑
2. `F:\AIStudio\QwenASR\dist2\QwenASR-WebView\QwenASR-WebView.exe`
3. 開發環境（沒有 EXE 時）：`F:\AIStudio\QwenASR\venv\Scripts\python.exe F:\AIStudio\QwenASR\app_webview.py`

找不到就問使用者 EXE 在哪，**不要**自己去裝別的 ASR 工具。

## 先確認狀態

```bash
"<EXE>" status --json
```
回傳已選核心、模型是否就緒、加速版本。`selectedReady:false` 代表模型還沒下載——
第一次轉錄會自動下載（1～2.5 GB，需數分鐘），先告知使用者再繼續。

## 轉錄

```bash
"<EXE>" transcribe "D:\path\audio.mp3" --json
```

stdout 是一份 JSON：

```json
{"input":"...","ok":true,"srtPath":"...\\subtitles\\audio.srt",
 "segments":[{"i":1,"start":0.48,"end":1.84,"text":"再看立法院的質詢"}, ...],
 "text":"整段連續逐字稿"}
```

常用選項：

| 選項 | 說明 |
|------|------|
| `-l Chinese` / `Japanese` / `English` | 指定語言（留空＝自動偵測；已知語言時指定較準） |
| `--profile zh\|tea\|ja\|twmix` | 依用途換模型，**只影響這次執行**，不會改到使用者 GUI 的設定 |
| `--hint "人名、術語、背景說明"` | 提示詞，能明顯改善專有名詞 |
| `--diarize [--speakers N]` | 說話者分離（segments 會多 `speaker` 欄位） |
| `-o out.srt` / `-f srt\|txt\|json` | 指定輸出檔與格式（僅單一輸入時有效） |

`--profile` 選哪個（不確定就用預設，別亂換）：

- `zh` 中文標準 — 一般情況的預設，斷句標點最完整
- `tea` 台灣國語 — 訪談／質詢／講座等**乾淨人聲**，原生輸出繁體
  ⚠️ 背景配樂較重的段落或歌曲可能整段無輸出，音樂類素材不要用它
- `ja` 日本語 — 日語／動漫語音
- `twmix` 國台語混合 — 速度極快，但**不輸出標點**

多檔：`transcribe a.mp3 b.mp3 c.mp3 --json` → `{"results":[...]}`。

## 校正字幕（重要：時間軸不可碰）

流程是**三步**，中間那步才是你做的事：

```bash
# ① 轉錄並存成 JSON
"<EXE>" transcribe "audio.mp3" -o work.json -f json --json > result.json

# ② 你讀 work.json，只改每個物件的 "text"，存成 fixed.json
#    絕對不要改 i / start / end，也不要增刪段落

# ③ 用原始檔當時間軸基準寫回 SRT
"<EXE>" apply fixed.json --base work.json -o final.srt
```

`--base` 會**以原始檔的時間軸為準**，只從你的檔案取 `text` 依 `i` 對應回填 ——
就算你不小心動到時間，也會被基準檔蓋回去。**校正時一律加 `--base`。**

校正時該做與不該做：

- ✅ 錯字、同音字、專有名詞、人名、公司名、術語
- ✅ 明顯的語音誤判（「修醉」→「修碎」之類，依上下文判斷）
- ❌ 不要潤稿、不要改語氣、不要刪贅字 —— 除非使用者明確要求
- ❌ 不要合併或拆分段落（會與時間軸脫節）

## 摘要 / 改寫

EXE 只負責交出逐字稿，**摘要是你自己做**。用 `--json` 拿 `text` 欄位
（整段連續逐字稿）或 `segments`（要引用時間點時用）直接處理即可，
不需要任何額外命令。要標時間點時引用 `start`，格式化成 `mm:ss` 給使用者。

## 注意事項

- **第一次執行會下載模型**（1～2.5 GB）。長音檔轉錄可能數分鐘 —— 用背景執行
  或把逾時放寬，不要中途重試（會重跑整個檔案）。
- **stdout 是乾淨 JSON**；載入與進度訊息都在 stderr。解析前不需要清洗。
- 退出碼：`0` 成功／`1` 轉錄失敗／`2` 參數錯誤／`130` 中斷。
- 影片檔（mp4/mkv/…）可直接餵，會自動抽音軌（首次需下載 ffmpeg）。
- 產出的 SRT 預設落在 EXE 旁的 `subtitles\`，路徑在 `srtPath`。
- 「未產生字幕（未偵測到人聲）」通常是**素材與模型不合**（例如用 `tea`
  轉歌曲），改用 `--profile zh` 重試。

## 查可用模型

```bash
"<EXE>" profiles --json     # 含本機硬體判斷與各 profile 是否已下載
```
