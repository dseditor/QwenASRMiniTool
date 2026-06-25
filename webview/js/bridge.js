/* ============================================================
   bridge.js — 統一前端橋接層 window.QwenAPI
   三版共用同一前端的關鍵：UI 只呼叫 QwenAPI.*，由本層偵測
   並路由到對應傳輸：
     • desktop : pywebview（window.pywebview.api.*）
     • web     : 端點 HTTP（fetch /v1, /api）
     • mock    : 純展示（file:// 無後端，回傳範例資料）
   進度等長時間事件透過內建 event bus 回報；桌面後端以
   window.QwenAPI._emit(event, payload) 主動推送。
   ============================================================ */
(function () {
  "use strict";

  // ── event bus ───────────────────────────────────────────
  const listeners = {};                       // event -> Set<fn>
  function on(ev, fn) { (listeners[ev] ||= new Set()).add(fn); return () => off(ev, fn); }
  function off(ev, fn) { listeners[ev]?.delete(fn); }
  function emit(ev, payload) { listeners[ev]?.forEach(fn => { try { fn(payload); } catch (e) { console.error(e); } }); }

  // ── 傳輸偵測 ────────────────────────────────────────────
  // pywebview 會在 window.pywebview.api 就緒前短暫不存在，
  // 故 ready 以 pywebviewready 事件或 fetch /health 為準。
  let MODE = "mock";
  const KEY = new URLSearchParams(location.search).get("k") || "";

  function hasPywebview() { return !!(window.pywebview && window.pywebview.api); }
  function isHttp() { return location.protocol === "http:" || location.protocol === "https:"; }

  // HTTP 時先探測 /health：有後端 → web；純靜態伺服（無 API）→ 降級 mock。
  // 這讓同一份檔案在「python -m http.server 預覽」時自動走展示資料。
  async function probeHttp() {
    try {
      const ctrl = new AbortController();
      const t = setTimeout(() => ctrl.abort(), 1200);
      const r = await fetch("/health", { signal: ctrl.signal });
      clearTimeout(t);
      return r.ok;                       // 200 → 真有端點後端
    } catch { return false; }
  }

  const ready = new Promise(resolve => {
    async function settle() {
      if (hasPywebview()) MODE = "desktop";
      else if (isHttp()) MODE = (await probeHttp()) ? "web" : "mock";
      else MODE = "mock";
      console.info("[QwenAPI] transport =", MODE);
      resolve(MODE);
    }
    if (hasPywebview()) return settle();
    // pywebview 注入較慢時等其事件；否則靠 protocol 判斷
    window.addEventListener("pywebviewready", settle, { once: true });
    if (document.readyState === "complete") setTimeout(settle, 60);
    else window.addEventListener("load", () => setTimeout(settle, 60), { once: true });
  });

  // ── desktop：呼叫 pywebview js_api ───────────────────────
  function py(method, ...args) {
    if (!hasPywebview() || typeof window.pywebview.api[method] !== "function") {
      return Promise.reject(new Error(`桌面後端缺少方法：${method}`));
    }
    return window.pywebview.api[method](...args);
  }

  // ── web：fetch 輔助 ─────────────────────────────────────
  function authHeaders(extra) {
    const h = Object.assign({}, extra || {});
    if (KEY) h["Authorization"] = "Bearer " + KEY;
    return h;
  }
  async function apiGet(path) {
    const r = await fetch(path + (path.includes("?") ? "&" : "?") + "k=" + encodeURIComponent(KEY),
      { headers: authHeaders() });
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  }

  // ════════════════════════════════════════════════════════
  // 公開 API
  // 每個方法都「優雅降級」：web/mock 缺的桌面能力回傳 null，
  // 讓 UI 自行 fallback（例如 pickFile 為 null → 改用 <input>）。
  // ════════════════════════════════════════════════════════
  const api = {
    on, off, _emit: emit,
    get mode() { return MODE; },
    ready,

    // ── 狀態 ──────────────────────────────────────────────
    async getStatus() {
      if (MODE === "desktop") return py("get_status");
      if (MODE === "web") {
        try { const h = await apiGet("/health"); return { modelReady: !!h.model_ready, backend: "—", device: "—", version: "", appName: "聲音辨識" }; }
        catch { return MOCK.status; }
      }
      return MOCK.status;
    },

    // ── 選檔（桌面原生對話框）。web/mock → null，UI 改用 <input> ──
    async pickFile() {
      if (MODE === "desktop") return py("pick_file");
      return null;
    },

    async loadHintTxt() {
      if (MODE === "desktop") return py("load_hint_txt");
      return null;
    },

    // ── 轉錄 ──────────────────────────────────────────────
    // opts: {path?, file?, language, diarize, nSpeakers, align, hint}
    // 進度經 on('progress', {pct, status})；完成 resolve {segments, srtPath}
    async transcribe(opts) {
      if (MODE === "desktop") return py("transcribe", sanitize(opts));
      if (MODE === "web") return webTranscribe(opts);
      return mockTranscribe(opts);
    },
    async cancel() {
      if (MODE === "desktop") return py("cancel");
      MOCK.cancelled = true; return true;
    },

    async openOutputDir() {
      if (MODE === "desktop") return py("open_output_dir");
      return false;
    },

    // ── 裝置 / 後端 ──────────────────────────────────────
    async listDevices() {
      if (MODE === "desktop") return py("list_devices");
      return MOCK.devices;
    },
    async setBackend(id) {
      if (MODE === "desktop") return py("set_backend", id);
      return true;
    },

    // ── 設定 ──────────────────────────────────────────────
    async getSettings() {
      if (MODE === "desktop") return py("get_settings");
      return MOCK.settings;
    },
    async setSettings(patch) {
      if (MODE === "desktop") return py("set_settings", patch);
      Object.assign(MOCK.settings, patch); return MOCK.settings;
    },

    // ── 端點服務 ──────────────────────────────────────────
    async getEndpoint() {
      if (MODE === "desktop") return py("get_endpoint");
      return MOCK.endpoint;
    },
    async toggleEndpoint(on_) {
      if (MODE === "desktop") return py("toggle_endpoint", on_);
      MOCK.endpoint.running = on_; return MOCK.endpoint;
    },
    async regenKey() {
      if (MODE === "desktop") return py("regen_key");
      MOCK.endpoint.key = "k_" + Math.random().toString(36).slice(2, 14); return MOCK.endpoint;
    },

    // ── 批次 ──────────────────────────────────────────────
    async getBatch() { if (MODE === "desktop") return py("get_batch"); return MOCK.batch; },
    async addBatchFiles() { if (MODE === "desktop") return py("add_batch_files"); return MOCK.batch; },
    async runBatch() { if (MODE === "desktop") return py("run_batch"); return true; },
  };

  // ── 工具：pywebview 傳參需可序列化（剝離 File 物件等）──
  function sanitize(o) {
    const c = Object.assign({}, o);
    delete c.file;                 // File 物件無法跨橋；桌面以 path 為準
    return c;
  }

  // ════════════════════════════════════════════════════════
  // web：走既有 /v1/audio/transcriptions（OpenAI 相容）
  // 端點目前回完整結果（非串流）→ 以單次進度事件模擬
  // ════════════════════════════════════════════════════════
  async function webTranscribe(opts) {
    if (!opts.file) throw new Error("web 模式需提供檔案物件");
    emit("progress", { pct: 5, status: "上傳中…" });
    const fd = new FormData();
    fd.append("file", opts.file, opts.file.name);
    if (opts.language) fd.append("language", opts.language);
    fd.append("response_format", "verbose_json");
    fd.append("align", opts.align ? "1" : "0");
    fd.append("diarize", opts.diarize ? "1" : "0");
    if (opts.nSpeakers && opts.nSpeakers !== "auto") fd.append("n_speakers", String(opts.nSpeakers));
    emit("progress", { pct: 30, status: "辨識中…" });
    const r = await fetch("/v1/audio/transcriptions?k=" + encodeURIComponent(KEY),
      { method: "POST", body: fd, headers: authHeaders() });
    const ct = r.headers.get("Content-Type") || "";
    if (!r.ok) {
      const msg = ct.includes("json") ? (await r.json()).error?.message : await r.text();
      throw new Error(msg || ("HTTP " + r.status));
    }
    const j = await r.json();
    emit("progress", { pct: 100, status: "完成" });
    const segments = (j.segments || []).map((s, i) => ({
      start: s.start, end: s.end, speaker: null, text: s.text,
    }));
    return { segments, srtPath: null };
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
  // 端點 url 組裝（含金鑰）
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
