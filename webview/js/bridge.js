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

    async listDevices() {
      if (MODE === "web") return apiGet("/api/devices");
      return MOCK.devices;
    },
    async setBackend(id) {
      if (MODE === "web") return apiPost("/api/backend", { index: id });
      return true;
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
    async toggleEndpoint(on_) {
      if (MODE === "web") return apiPost("/api/endpoint", { action: on_ ? "start" : "stop" });
      MOCK.endpoint.running = on_; return MOCK.endpoint;
    },
    async regenKey() {
      if (MODE === "web") return apiPost("/api/endpoint", { action: "regen" });
      MOCK.endpoint.key = "k_" + Math.random().toString(36).slice(2, 14);
      MOCK.endpoint.url = `http://${MOCK.endpoint.host}:${MOCK.endpoint.port}/?k=${MOCK.endpoint.key}`;
      return MOCK.endpoint;
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
    status: { modelReady: true, backend: "CPU · OpenVINO INT8", device: "AMD Ryzen 5 9600X", version: "1.0.9", appName: "聲音辨識" },
    settings: { scale: 100, format: "srt", vocab: "s2twp", mirror: "", ffmpeg: "ffmpeg/ffmpeg.exe", theme: "light", uiLang: "繁體中文" },
    endpoint: { running: true, host: "192.168.1.20", port: 11435, key: "k_8x2pf3qd7m1c", url: "" },
    devices: {
      devices: [
        { kind: "cpu", name: "AMD Ryzen 5 9600X 6-Core", note: "使用中" },
        { kind: "gpu", name: "NVIDIA GeForce RTX 4070", note: "8.0 GB 可用" },
      ],
      diag: { level: "warn", text: "GPU 偵測未完成：main.exe --show_devices 逾時。已自動改用 CPU 推理；如需 GPU 請更新顯卡 / Vulkan 驅動。" },
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
