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

## 橋接層：一份前端，三種傳輸

`bridge.js` 啟動時自動偵測並路由：

| 偵測條件 | 模式 | 傳輸 |
|---|---|---|
| `window.pywebview.api` 存在 | `desktop` | pywebview js_api（`app_webview.py`）|
| HTTP 且 `/health` 回 200 | `web` | 端點 HTTP（`api_server.py`）|
| 其餘（含純靜態伺服）| `mock` | 內建範例資料（設計預覽）|

UI 只呼叫 `QwenAPI.transcribe()` 等抽象方法，不需知道底層。
桌面獨有能力（如 `pickFile` 原生對話框）在 web/mock 回 `null`，
UI 自動 fallback（改用 `<input type=file>`）。

進度等長時間事件走內建 event bus：後端以
`window.evaluate_js("QwenAPI._emit('progress', {...})")` 主動推回，
三種傳輸共用同一組 `QwenAPI.on('progress', …)`。

## 本機預覽（無需後端）

```bash
python -m http.server 8777 --bind 127.0.0.1   # 於 webview/ 下執行
# 開 http://127.0.0.1:8777/index.html → 自動進 mock 模式，可完整點選
```

## 桌面實機

```bash
python app_webview.py     # 載入本資料夾，js_api 接上真實 ASREngine
```
