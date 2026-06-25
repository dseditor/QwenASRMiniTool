# webview/ — 共用前端（聲音辨識）

三個版本（桌面 EXE / Python / 端點服務）共用**同一份**前端介面。
設計方向：淺色清爽編輯室風，深湖綠 teal 強調色（B 版）。

## 結構

```
webview/
├── index.html        殼 + 六個視圖（音檔 / 批次 / 錄製 / 端點 / 模型 / 設定）
├── css/app.css       設計系統（色彩 token、元件、響應式）
└── js/
    ├── bridge.js     window.QwenAPI 統一橋接層（傳輸抽象 + event bus）
    └── app.js        UI 邏輯（只透過 QwenAPI，與後端解耦）
```

## 架構：本機伺服器 + 瀏覽器渲染（同 Eel / flaskwebgui，但純標準庫）

不使用 pywebview（其 Windows pythonnet 後端有遞迴 bug，且 EXE 內要打包 .NET）。
改採「本機 HTTP 伺服器 + 系統 Edge `--app` 開窗」，零第三方相依、易 PyInstaller 打包。

| 後端檔 | 角色 |
|---|---|
| `webview_backend.py` | 與視窗/傳輸無關的業務邏輯（status/settings/devices/endpoint/transcribe），重用 ASREngine |
| `webview_server.py`  | stdlib HTTP：serve 本資料夾 + `/api/*` + SSE `/api/events`；只綁 127.0.0.1 |
| `app_webview.py`     | 起 server → 背景載入模型 → 原生 WebView2 視窗（pywebview 只載入網址、**無 js_api**；fallback Edge `--app --inprivate` → 預設瀏覽器）→ 等關閉收 server |

## bridge.js 傳輸偵測

| 條件 | 模式 | 傳輸 |
|---|---|---|
| HTTP 且 `/health` 回 200 | `web` | `/api/*`（桌面本機 server / 端點 LAN）；進度經 SSE `/api/events` |
| 其餘（含純靜態伺服 / file://）| `mock` | 內建範例資料（設計預覽） |

UI 只呼叫 `QwenAPI.transcribe()` 等抽象方法。檔案選擇走瀏覽器原生
`<input type=file>` → 以 multipart 上傳到本機 server，與端點版走相同路徑。

## 本機預覽（純設計，無後端）

```bash
python -m http.server 8777 --bind 127.0.0.1   # 於 webview/ 下執行
# 開 http://127.0.0.1:8777/index.html → /health 探測失敗 → 自動 mock 模式
```

## 桌面實機

```bash
python app_webview.py          # 起本機 server + 載入模型 + Edge --app 開窗
# 除錯：python webview_server.py 固定起在 :8765，再用瀏覽器開
```
