/* ============================================================
   bridge.js — 統一前端橋接層 window.QwenAPI
   三版共用同一前端的關鍵：UI 只呼叫 QwenAPI.*，由本層偵測並路由：
     • web  : 本機/區網 HTTP（桌面 = 本機 server + Edge；端點 = LAN）
              方法打 /api/*，進度經 SSE /api/events 推回
     • mock : 純展示（file:// 或無後端，回範例資料供設計預覽）
   進度等長時間事件透過內建 event bus；web 模式由伺服器以 SSE 推送，
   mock 模式自行模擬 —— UI 兩者皆以 QwenAPI.on('progress', …) 接收。
   ============================================================ */
(function () {
  "use strict";

  // ── event bus ───────────────────────────────────────────
  const listeners = {};
  function on(ev, fn) { (listeners[ev] ||= new Set()).add(fn); return () => off(ev, fn); }
  function off(ev, fn) { listeners[ev]?.delete(fn); }
  function emit(ev, payload) { listeners[ev]?.forEach(fn => { try { fn(payload); } catch (e) { console.error(e); } }); }

  let MODE = "mock";
  const KEY = new URLSearchParams(location.search).get("k") || "";
  const isHttp = () => location.protocol === "http:" || location.protocol === "https:";

  // HTTP 時探測 /health：有後端 → web；純靜態/無 API → 降級 mock。
  async function probeHttp() {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 1500);
      const r = await fetch("/health", { signal: ctrl.signal, headers: authHeaders() });
      clearTimeout(t);
      return r.ok;
    } catch { return false; }
  }

  const ready = new Promise(resolve => {
    async function settle() {
      MODE = (isHttp() && await probeHttp()) ? "web" : "mock";
      console.info("[QwenAPI] transport =", MODE);
      if (MODE === "web") connectSSE();
      resolve(MODE);
    }
    if (document.readyState === "complete") setTimeout(settle, 0);
    else window.addEventListener("load", () => setTimeout(settle, 0), { once: true });
  });

  // ── web：SSE 進度/狀態串流 ──────────────────────────────
  function connectSSE() {
    try {
      const es = new EventSource("/api/events" + (KEY ? "?k=" + encodeURIComponent(KEY) : ""));
      es.onmessage = e => {
        try { const m = JSON.parse(e.data); if (m && m.event) emit(m.event, m.payload); }
        catch { /* 心跳/註解行，略過 */ }
      };
      es.onerror = () => { /* EventSource 會自動重連 */ };
    } catch (e) { console.warn("[QwenAPI] SSE 無法建立", e); }
  }

  // ── web：fetch 輔助 ─────────────────────────────────────
  function authHeaders(extra) {
    const h = Object.assign({}, extra || {});
    if (KEY) h["Authorization"] = "Bearer " + KEY;
    return h;
  }
  function withKey(path) { return KEY ? path + (path.includes("?") ? "&" : "?") + "k=" + encodeURIComponent(KEY) : path; }
  async function apiGet(path) {
    const r = await fetch(withKey(path), { headers: authHeaders() });
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }
  async function apiPost(path, body) {
    const r = await fetch(withKey(path), {
      method: "POST", headers: authHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) {
      let msg = "HTTP " + r.status;
      try { msg = (await r.json()).error?.message || msg; } catch {}
      throw new Error(msg);
    }
    return r.json();
  }

  // ════════════════════════════════════════════════════════
  // 公開 API（web 打 /api/*；mock 回範例資料）
  // 桌面獨有能力（pickFile）在 web/mock 回 null → UI 改用 <input> fallback
  // ════════════════════════════════════════════════════════
  const api = {
    on, off, _emit: emit,
    get mode() { return MODE; },
    ready,

    async getStatus() {
      if (MODE === "web") return apiGet("/api/status");
      return MOCK.status;
    },

    async pickFile() { return null; },            // 瀏覽器環境一律用 <input type=file>
    async loadHintTxt() { return null; },

    // opts: {file, language, diarize, nSpeakers, align, hint}
    async transcribe(opts) {
      if (MODE === "web") return webTranscribe(opts);
      return mockTranscribe(opts);
    },
    async cancel() {
      if (MODE === "web") { try { await apiPost("/api/cancel", {}); } catch {} return true; }
      MOCK.cancelled = true; return true;
    },

    async openOutputDir() {
      if (MODE === "web") { try { await apiPost("/api/open-output", {}); } catch {} return true; }
      return false;
    },
    async checkUpdate() {
      if (MODE === "web") { try { return await apiPost("/api/check-update", {}); } catch {} }
      window.open("https://github.com/dseditor/QwenASRMiniTool/releases/latest", "_blank");
      return { ok: true };
    },

    async listDevices() {
      if (MODE === "web") return apiGet("/api/devices");
      return MOCK.devices;
    },
    async setBackend(id) {
      if (MODE === "web") return apiPost("/api/backend", { index: id });
      return true;
    },
    async getModelOptions() {
      if (MODE === "web") return apiGet("/api/model-options");
      return MOCK.modelOptions;
    },
    async setModel(core, model) {
      if (MODE === "web") return apiPost("/api/model", { core, model });
      return { ok: true, core, model, arch: "（mock）", restartRequired: false, canLoadNow: true,
               message: `（展示模式）已選「${core} · ${model}」，點「下載並載入模型」開始` };
    },
    // 首次選定模型 → 就地下載並載入（進度走 SSE "progress"，完成走 "status"）
    async startLoad() {
      if (MODE === "web") return apiPost("/api/load", {});
      // mock：模擬載入完成
      setTimeout(() => { MOCK.status.modelReady = true; emit("status", { modelReady: true }); }, 1200);
      return { ok: true, loading: true };
    },
    async getLanguages() {
      if (MODE === "web") return apiGet("/api/languages");
      return MOCK.languages;
    },
    async getHealthCheck() {
      if (MODE === "web") return apiGet("/api/health-check");
      return MOCK.health;
    },

    async getSettings() {
      if (MODE === "web") return apiGet("/api/settings");
      return MOCK.settings;
    },
    async setSettings(patch) {
      if (MODE === "web") return apiPost("/api/settings", patch);
      Object.assign(MOCK.settings, patch); return MOCK.settings;
    },

    async getEndpoint() {
      if (MODE === "web") return apiGet("/api/endpoint");
      return MOCK.endpoint;
    },
    async toggleEndpoint(on_, port) {
      if (MODE === "web") return apiPost("/api/endpoint", { action: on_ ? "start" : "stop", port });
      MOCK.endpoint.running = on_; if (port) MOCK.endpoint.port = +port; return MOCK.endpoint;
    },
    async regenKey() {
      if (MODE === "web") return apiPost("/api/endpoint", { action: "regen" });
      MOCK.endpoint.key = "k_" + Math.random().toString(36).slice(2, 14);
      MOCK.endpoint.url = `http://${MOCK.endpoint.host}:${MOCK.endpoint.port}/?k=${MOCK.endpoint.key}`;
      return MOCK.endpoint;
    },

    // 對外臨時網址（Cloudflare）— 狀態/網址另經 SSE "tunnel" 事件即時推送
    async getTunnel() {
      if (MODE === "web") return apiGet("/api/tunnel");
      return MOCK.tunnel;
    },
    async toggleTunnel(on_) {
      if (MODE === "web") return apiPost("/api/tunnel", { action: on_ ? "start" : "stop" });
      MOCK.tunnel.running = on_;
      MOCK.tunnel.url = on_ ? "https://demo-fluffy-cloud.trycloudflare.com/?k=•••••" : "";
      MOCK.tunnel.status = on_ ? "ready" : "";
      setTimeout(() => emit("tunnel", MOCK.tunnel), 600);   // 模擬非同步就緒
      return { running: on_, url: "", status: on_ ? "建立中…" : "" };
    },
    // QR 圖網址：web 走後端 /api/qr（segno）；mock 無後端 → 空字串（前端顯示佔位）
    qrSrc(data) {
      if (!data || MODE !== "web") return "";
      return withKey("/api/qr?d=" + encodeURIComponent(data));
    },

    async getBatch() {
      if (MODE === "web") { try { return await apiGet("/api/batch"); } catch { return { summary: { done: 0, total: 0 }, items: [] }; } }
      return MOCK.batch;
    },
    async addBatchFiles() { return this.getBatch(); },
    async runBatch() { return true; },
  };

  // ── web 轉錄：POST /api/transcribe（進度走 SSE）──────────
  async function webTranscribe(opts) {
    if (!opts.file) throw new Error("請先選擇檔案");
    const fd = new FormData();
    fd.append("file", opts.file, opts.file.name);
    if (opts.language) fd.append("language", opts.language);
    fd.append("align", opts.align ? "1" : "0");
    fd.append("diarize", opts.diarize ? "1" : "0");
    if (opts.nSpeakers && opts.nSpeakers !== "auto") fd.append("n_speakers", String(opts.nSpeakers));
    if (opts.hint) fd.append("hint", opts.hint);
    emit("progress", { pct: 3, status: "上傳中…" });
    const r = await fetch(withKey("/api/transcribe"), { method: "POST", body: fd, headers: authHeaders() });
    if (!r.ok) {
      let msg = "HTTP " + r.status;
      try { msg = (await r.json()).error?.message || msg; } catch {}
      throw new Error(msg);
    }
    return r.json();   // {segments, srtPath}
  }

  // ════════════════════════════════════════════════════════
  // mock：純展示資料（讓設計稿在瀏覽器中完全可互動）
  // ════════════════════════════════════════════════════════
  const MOCK = {
    cancelled: false,
    status: { modelReady: true, backend: "GPU · CRISPASR（Vulkan）", device: "NVIDIA GeForce RTX", version: "webview 0.1", appName: "聲音辨識小工具", hasAnyModel: false, selectedReady: false },
    settings: { scale: 100, format: "srt", vocab: "s2twp", mirror: "", ffmpeg: "", theme: "light", uiLang: "繁體中文", vad: 0.5 },
    endpoint: { running: true, host: "192.168.1.20", port: 11435, key: "k_8x2pf3qd7m1c", url: "" },
    tunnel: { running: false, url: "", status: "" },
    devices: {
      devices: [
        { kind: "cpu", name: "AMD Ryzen 5 9600X 6-Core", note: "使用中" },
        { kind: "gpu", name: "NVIDIA GeForce RTX 4070", note: "8.0 GB 可用" },
      ],
      diag: { level: "info", text: "（展示資料）GPU 由 CrispASR 偵測；實機會列出可用的獨立顯示卡與可用記憶體。" },
    },
    modelOptions: {
      cores: [
        { label: "Qwen", models: [
          { label: "Qwen3-ASR-0.6B", backend: "openvino", arch: "CPU · OpenVINO INT8", note: "" },
          { label: "Qwen3-ASR-1.7B INT8", backend: "openvino", arch: "CPU · OpenVINO INT8", note: "" },
          { label: "Qwen3-ASR-1.7B Q4 (CRISPASR/Vulkan)", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）", note: "" },
          { label: "Qwen3-ASR-1.7B Q8 (CRISPASR/Vulkan)", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）", note: "" },
          { label: "Qwen3-ASR-1.7B Q8（Vulkan · 相容）", backend: "chatllm", arch: "GPU · chatllm Vulkan", note: "⚠️ chatllm 核心在部分 AMD 內顯／APU 有已知相容問題，且核心二進位未隨安裝包提供（保留供既有使用者向下相容）；若遇當機或無輸出，建議改用「Qwen · CRISPASR/Vulkan」核心。" },
        ]},
        { label: "Whisper (Breeze)", models: [
          { label: "Breeze Q4 (輕量)", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）" },
          { label: "Breeze Q5 (標準)", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）" },
          { label: "Breeze Q8 (精確)", backend: "crispasr", arch: "GPU · CRISPASR（Vulkan）" },
        ]},
      ],
      current: { core: "Qwen", model: "Qwen3-ASR-0.6B" },
      activeArch: "CPU · OpenVINO INT8",
    },
    languages: {
      languages: [
        { label: "自動偵測", value: "" }, { label: "Chinese", value: "Chinese" },
        { label: "English", value: "English" }, { label: "Japanese", value: "Japanese" },
        { label: "Korean", value: "Korean" }, { label: "Cantonese", value: "Cantonese" },
      ],
    },
    health: {
      summary: { red: 0, yellow: 3, ok: true }, activeBackend: "openvino",
      cores: [
        { label: "Qwen · OpenVINO（CPU）", backend: "openvino", items: [
          { key: "model", label: "ASR 模型（0.6B）", status: "green", detail: "已下載" },
          { key: "vad", label: "語音分段 VAD（silero）", status: "green", detail: "已內建" },
          { key: "fa", label: "時間軸對齊 FA", status: "yellow", detail: "未下載（約 939MB，啟用對齊時下載）" },
          { key: "diar", label: "說話者分離（外部 ONNX）", status: "yellow", detail: "未下載（約 32MB，啟用分離時下載）" },
        ]},
        { label: "CRISPASR（Vulkan：Whisper/Breeze + Qwen）", backend: "crispasr", items: [
          { key: "core", label: "CrispASR 核心（crispasr.exe）", status: "green", detail: "已下載" },
          { key: "breeze", label: "Whisper/Breeze 模型（Q5）", status: "green", detail: "已下載" },
          { key: "qwen", label: "Qwen3-ASR-1.7B 模型（Q8）", status: "yellow", detail: "未下載（啟用時下載）" },
          { key: "fa", label: "時間軸對齊 FA（aligner gguf Q5）", status: "green", detail: "已下載" },
          { key: "diar", label: "說話者分離（外部 ONNX，與 OpenVINO 共用）", status: "yellow", detail: "未下載" },
        ]},
      ],
      shared: [
        { key: "ffmpeg", label: "FFmpeg（影片抽音軌用）", status: "green", detail: "已偵測：C:/ffmpeg/ffmpeg.exe" },
        { key: "diar_shared", label: "說話者分離模型（共用）", status: "yellow", detail: "未下載（啟用分離時自動下載）" },
      ],
    },
    batch: {
      summary: { done: 3, total: 7 },
      items: [
        { name: "ep01_intro.mp3", status: "done", progress: 1 },
        { name: "ep02_guest.m4a", status: "done", progress: 1 },
        { name: "會議_0612.wav", status: "done", progress: 1 },
        { name: "podcast_long.mp4", status: "running", progress: 0.42 },
        { name: "lecture_3.wav", status: "pending", progress: 0 },
        { name: "interview.m4a", status: "pending", progress: 0 },
        { name: "broken_amd.wav", status: "failed", progress: 0, error: "模型輸出異常，此核心可能不相容，建議改用 CPU(OpenVINO) 核心。" },
      ],
    },
    segments: [
      { start: 3.0, end: 7.2, speaker: 1, text: "大家好，今天我們來聊聊本地語音辨識這個主題。" },
      { start: 7.4, end: 11.8, speaker: 2, text: "對啊，最吸引我的是資料完全不用上傳雲端，隱私有保障。" },
      { start: 12.0, end: 16.5, speaker: 1, text: "沒錯，而且這套支援說話者分離，會議記錄整理起來特別方便。" },
      { start: 16.8, end: 21.0, speaker: 2, text: "時間軸對齊也很實用，字幕可以直接匯出成 SRT 或純文字。" },
      { start: 21.3, end: 26.0, speaker: 1, text: "如果顯卡相容，用 GPU 核心速度會再快上好幾倍。" },
      { start: 26.2, end: 31.0, speaker: 2, text: "那我們下半段就來實際示範一次完整的轉錄流程吧。" },
    ],
  };
  MOCK.endpoint.url = `http://${MOCK.endpoint.host}:${MOCK.endpoint.port}/?k=${MOCK.endpoint.key}`;

  async function mockTranscribe() {
    MOCK.cancelled = false;
    const steps = [
      [12, "載入音訊…"], [25, "語音分段（VAD）…"], [44, "辨識中 · 第 3/8 段"],
      [68, "辨識中 · 第 5/8 段"], [86, "時間軸對齊…"], [100, "完成"],
    ];
    for (const [pct, status] of steps) {
      if (MOCK.cancelled) throw new Error("已取消");
      await sleep(420);
      emit("progress", { pct, status });
    }
    return { segments: MOCK.segments, srtPath: "subtitles/B.srt" };
  }
  const sleep = ms => new Promise(r => setTimeout(r, ms));

  window.QwenAPI = api;
})();
