/* ============================================================
   app.js — UI 邏輯（與後端解耦，全程只透過 window.QwenAPI）
   ============================================================ */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const API = window.QwenAPI;

  const VIEW_TITLES = {
    file: "音檔轉字幕", batch: "批次辨識", record: "錄製轉換",
    endpoint: "端點服務", model: "模型與裝置", settings: "設定",
  };

  // ── 導覽切換 ────────────────────────────────────────────
  function switchView(name) {
    $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === name));
    $$(".view").forEach(v => v.classList.toggle("active", v.dataset.view === name));
    $("#view-title").textContent = VIEW_TITLES[name] || "";
    $("#view-ctx").textContent = "";
    if (name === "batch") renderBatch();
    if (name === "model") renderModel();
  }
  $("#nav").addEventListener("click", e => {
    const btn = e.target.closest(".nav-item");
    if (btn) switchView(btn.dataset.view);
  });

  // ── 狀態列 ──────────────────────────────────────────────
  async function refreshStatus() {
    const s = await API.getStatus();
    const el = $("#model-status");
    el.classList.toggle("loading", !s.modelReady);
    $(".t", el).textContent = s.modelReady ? "模型已就緒" : "載入模型中…";
    if (s.version) $("#app-version").textContent = "v" + s.version;
  }

  // ════════════════════════════════════════════════════════
  // 音檔轉字幕
  // ════════════════════════════════════════════════════════
  let picked = null;             // {path?, name, file?} 或 null
  const drop = $("#drop"), hiddenFile = $("#hidden-file");

  function setFile(f) {
    picked = f;
    if (!f) { drop.classList.remove("has-file"); renderDropEmpty(); $("#btn-run").disabled = true; return; }
    drop.classList.add("has-file");
    const meta = f.sizeSec ? `${f.sizeSec} 秒` : (f.size ? (f.size / 1048576).toFixed(1) + " MB" : "");
    drop.innerHTML = `<span class="file-chip">${escapeHtml(f.name)}
      ${meta ? `<span class="meta">${meta}</span>` : ""}
      <button class="x" title="移除">✕</button></span>`;
    drop.querySelector(".x").addEventListener("click", ev => { ev.stopPropagation(); setFile(null); });
    $("#btn-run").disabled = false;
  }
  function renderDropEmpty() {
    drop.innerHTML = `<div class="ico"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v12"/><path d="M7 10l5 5 5-5"/><path d="M5 21h14"/></svg></div>
      <div class="big">拖曳音訊到此，或<b>點擊選擇檔案</b></div>
      <div class="sub">支援 mp3 / wav / m4a / 影片音軌</div>`;
  }

  // 桌面：原生對話框；web/mock：fallback 到 <input type=file>
  drop.addEventListener("click", async () => {
    if (drop.classList.contains("has-file")) return;
    const native = await API.pickFile();
    if (native) setFile(native);
    else hiddenFile.click();
  });
  hiddenFile.addEventListener("change", e => { if (e.target.files[0]) setFile(e.target.files[0]); });
  ["dragover", "dragenter"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add("hot"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("hot"); }));
  drop.addEventListener("drop", e => { const f = e.dataTransfer.files[0]; if (f) setFile(f); });

  $("#btn-load-txt").addEventListener("click", async () => {
    const t = await API.loadHintTxt();
    if (t != null) $("#hint-box").value = t;
    else alert("此模式無法開啟檔案對話框，請直接貼上文字。");
  });

  // 轉錄
  let running = false;
  $("#btn-run").addEventListener("click", async () => {
    if (!picked || running) return;
    running = true;
    const btn = $("#btn-run");
    btn.disabled = true; btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="6" width="12" height="12" rx="2"/></svg>辨識中…`;
    $("#result").hidden = true;
    showProgress(0, "準備中…");
    logLine("▶ 開始轉換：" + picked.name);
    try {
      const res = await API.transcribe({
        path: picked.path, file: picked.file || picked,
        language: null,
        diarize: $("#sw-diar").checked,
        nSpeakers: $("#sel-spk").value,
        align: $("#sw-align").checked,
        hint: $("#hint-box").value.trim(),
      });
      renderResult(res.segments);
      logLine("✓ 完成，共 " + res.segments.length + " 段" + (res.srtPath ? "，已輸出 " + res.srtPath : ""));
      $("#btn-open-dir").disabled = false;
    } catch (err) {
      showProgress(0, "");
      $("#progress").hidden = true;
      logLine("✕ " + (err.message || err));
    } finally {
      running = false;
      btn.disabled = false;
      btn.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>開始轉換`;
    }
  });

  // 進度分派：批次執行中 → 更新當前批次列；否則 → 單檔進度條
  API.on("progress", ({ pct, status }) => {
    if (batchRunning && batchActiveIdx >= 0) {
      batchItems[batchActiveIdx].progress = pct / 100;
      updateBatchRow(batchActiveIdx);
    } else {
      showProgress(pct, status);
    }
  });
  function showProgress(pct, status) {
    $("#progress").hidden = false;
    $("#prog-bar").style.width = pct + "%";
    $("#prog-pct").textContent = Math.round(pct) + "%";
    if (status != null) $("#prog-status").textContent = status;
  }

  // ════ Signature：真實波形 + 播放 + 分段標註（Suno 式）════
  //   用 Web Audio decodeAudioData 解出真實峰值畫波形；<audio> 播放，
  //   播放頭跟走、已播段標色；波形下方對齊時間軸的字幕區塊 + 字幕卡
  //   隨播放高亮，皆可點選 seek —— 肉眼即可看字幕與音訊的對齊準確度。
  const WAVE_N = 200;
  let audioEl = null, audioUrl = null, waveDur = 0, curSegs = [];

  async function renderResult(segments) {
    $("#result").hidden = false;
    curSegs = segments;
    cleanupAudio();

    const file = picked && (picked.file || (picked instanceof Blob ? picked : null));
    let peaks = null;
    waveDur = segments.length ? segments[segments.length - 1].end : 0;

    if (file instanceof Blob) {
      audioUrl = URL.createObjectURL(file);
      audioEl = new Audio(audioUrl);
      try {                                   // 解碼取真實峰值（mp3/wav/m4a/webm…）
        const ab = await file.arrayBuffer();
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const buf = await ctx.decodeAudioData(ab.slice(0));
        waveDur = buf.duration || waveDur;
        peaks = computePeaks(buf, WAVE_N);
        ctx.close();
      } catch (e) { peaks = null; }           // 不支援的格式 → 平波形，但仍可播
    }
    waveDur = waveDur || 60;

    $("#wave-dur").textContent = fmtClock(waveDur);
    drawWave($("#wave"), peaks, segments, waveDur);
    renderSegStrip(segments, waveDur);
    renderSubs(segments);
    wireAudio();
  }

  // 真實峰值：每桶取絕對值最大，整體正規化
  function computePeaks(buf, n) {
    const ch = buf.getChannelData(0);
    const block = Math.max(1, Math.floor(ch.length / n));
    const peaks = [];
    for (let i = 0; i < n; i++) {
      let mx = 0;
      const base = i * block;
      for (let j = 0; j < block; j++) { const v = Math.abs(ch[base + j] || 0); if (v > mx) mx = v; }
      peaks.push(mx);
    }
    const norm = Math.max(...peaks, 1e-4);
    return peaks.map(p => p / norm);
  }

  function drawWave(container, peaks, segments, dur) {
    const n = peaks ? peaks.length : WAVE_N;
    let bars = "";
    for (let i = 0; i < n; i++) {
      const amp = peaks ? peaks[i] : 0.12;
      bars += `<div class="b" style="height:${Math.max(3, amp * 64).toFixed(1)}px"></div>`;
    }
    let marks = "";                            // 分段邊界標記
    segments.forEach(s => {
      marks += `<div class="seg-mark" style="left:${((s.start / dur) * 100).toFixed(2)}%"></div>`;
    });
    container.innerHTML = `<div class="playhead" id="playhead" style="left:0%"></div>${marks}${bars}`;
  }

  // 波形下方：依時間對齊的字幕區塊（Suno 式，可點選 seek）
  function renderSegStrip(segments, dur) {
    let host = $("#seg-strip");
    if (!host) { host = document.createElement("div"); host.id = "seg-strip"; host.className = "seg-strip"; $(".wave-panel").appendChild(host); }
    host.innerHTML = "";
    segments.forEach((s, i) => {
      const left = (s.start / dur) * 100, w = Math.max(0.6, ((s.end - s.start) / dur) * 100);
      const blk = document.createElement("div");
      blk.className = "seg-block" + (s.speaker ? " spk-" + (((s.speaker - 1) % 3) + 1) : "");
      blk.style.left = left.toFixed(2) + "%"; blk.style.width = w.toFixed(2) + "%";
      blk.dataset.seg = i; blk.title = s.text;
      blk.textContent = s.text;
      blk.addEventListener("click", () => seekTo(s.start));
      host.appendChild(blk);
    });
  }

  function renderSubs(segments) {
    const host = $("#subs"); host.innerHTML = "";
    segments.forEach((s, i) => host.appendChild(subCard(s, i)));
  }
  function subCard(s, idx) {
    const el = document.createElement("div");
    el.className = "sub-card"; el.dataset.seg = idx;
    const spk = s.speaker ? `<span class="chip spk-${((s.speaker - 1) % 3) + 1}">說話者 ${s.speaker}</span>` : "";
    el.innerHTML = `<span class="tc">${fmtClock(s.start)} → ${fmtClock(s.end)}</span>
      <div class="body"><div class="spk">${spk}</div><div class="txt">${escapeHtml(s.text)}</div></div>`;
    el.addEventListener("click", () => seekTo(s.start));
    return el;
  }

  // ── 播放接線 ────────────────────────────────────────────
  function wireAudio() {
    const wave = $("#wave"), play = $("#wave-play");
    const bars = $$(".b", wave);
    if (audioEl) {
      audioEl.addEventListener("timeupdate", onTime);
      audioEl.addEventListener("play", () => setPlayIcon(true));
      audioEl.addEventListener("pause", () => setPlayIcon(false));
      audioEl.addEventListener("ended", () => { setPlayIcon(false); });
    }
    // 點波形 seek
    wave.onclick = e => {
      const r = wave.getBoundingClientRect();
      seekTo(Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * waveDur);
    };
    if (play) play.onclick = () => { if (!audioEl) return; audioEl.paused ? audioEl.play() : audioEl.pause(); };

    function onTime() {
      const t = audioEl.currentTime, ratio = waveDur ? t / waveDur : 0;
      const ph = $("#playhead"); if (ph) ph.style.left = (ratio * 100).toFixed(2) + "%";
      const playedIdx = Math.floor(ratio * bars.length);
      bars.forEach((b, i) => b.classList.toggle("played", i <= playedIdx));
      const cur = curSegs.findIndex(s => t >= s.start && t < s.end);
      $$(".sub-card").forEach(c => c.classList.toggle("playing", +c.dataset.seg === cur));
      $$(".seg-block").forEach(c => c.classList.toggle("playing", +c.dataset.seg === cur));
      const tm = $("#wave-cur"); if (tm) tm.textContent = fmtClock(t);
      // 自動捲動到播放中的字幕卡
      if (cur >= 0) { const el = $(`.sub-card[data-seg="${cur}"]`); if (el && playFollow) el.scrollIntoView({ block: "nearest" }); }
    }
  }
  let playFollow = true;
  function setPlayIcon(playing) {
    const b = $("#wave-play"); if (!b) return;
    b.innerHTML = playing
      ? `<svg viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/></svg>`
      : `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>`;
  }
  function seekTo(sec) { if (audioEl) { audioEl.currentTime = sec; audioEl.play().catch(() => {}); } }
  function cleanupAudio() {
    try { if (audioEl) { audioEl.pause(); audioEl.src = ""; } } catch (e) {}
    if (audioUrl) { try { URL.revokeObjectURL(audioUrl); } catch (e) {} audioUrl = null; }
    audioEl = null;
  }

  $("#btn-open-dir").addEventListener("click", () => API.openOutputDir());

  // ── 記錄 ────────────────────────────────────────────────
  function logLine(msg) {
    const el = $("#file-log"); el.hidden = false;
    const d = document.createElement("div"); d.className = "ln"; d.textContent = msg;
    el.appendChild(d); el.scrollTop = el.scrollHeight;
  }

  // ════════════════════════════════════════════════════════
  // 批次（client-side 佇列：循序呼叫既有 /api/transcribe）
  // ════════════════════════════════════════════════════════
  const ST = {
    done: ["完成", "chip-ok"], running: ["辨識中", "chip-accent"],
    pending: ["待處理", "chip-muted"], failed: ["失敗", "chip-live"],
  };
  let batchItems = [];                          // {file, name, status, progress, srtPath?, error?}
  let batchRunning = false, batchActiveIdx = -1;

  // 隱藏的多檔選擇器（瀏覽器原生）
  const batchInput = document.createElement("input");
  batchInput.type = "file"; batchInput.accept = "audio/*,video/*";
  batchInput.multiple = true; batchInput.style.display = "none";
  document.body.appendChild(batchInput);
  batchInput.addEventListener("change", e => {
    for (const f of e.target.files) batchItems.push({ file: f, name: f.name, status: "pending", progress: 0 });
    batchInput.value = "";
    renderBatch();
  });

  function batchSummary() {
    const done = batchItems.filter(i => i.status === "done").length;
    $("#batch-summary").textContent = `${done} / ${batchItems.length} 完成`;
  }
  function renderBatch() {
    batchSummary();
    const host = $("#batch-list"); host.innerHTML = "";
    if (!batchItems.length) {
      host.innerHTML = `<div class="card" style="text-align:center;color:var(--muted);padding:28px">
        尚未加入檔案 — 點「加入檔案」可一次選多個音訊循序處理。</div>`;
      return;
    }
    batchItems.forEach((it, idx) => host.appendChild(batchRow(it, idx)));
  }
  function batchRow(it, idx) {
    const [label, cls] = ST[it.status] || ST.pending;
    const row = document.createElement("div");
    row.className = "b-row"; row.dataset.idx = idx;
    row.innerHTML = `
      <div class="nm"><div class="f">${escapeHtml(it.name)}</div>
        ${it.error ? `<div class="e">${escapeHtml(it.error)}</div>` : ""}</div>
      <div class="pbar"><div class="bar"><i style="width:${Math.round((it.progress || 0) * 100)}%"></i></div></div>
      <span class="chip ${cls}">${label}</span>
      <button class="more" title="${it.status === "done" ? "開啟輸出資料夾" : "移除"}">⋯</button>`;
    row.querySelector(".more").addEventListener("click", () => {
      if (it.status === "done") API.openOutputDir();
      else if (!batchRunning) { batchItems.splice(idx, 1); renderBatch(); }
    });
    return row;
  }
  function updateBatchRow(idx) {
    const it = batchItems[idx];
    const row = $(`.b-row[data-idx="${idx}"]`); if (!row) return;
    const [label, cls] = ST[it.status] || ST.pending;
    row.querySelector(".pbar i").style.width = Math.round((it.progress || 0) * 100) + "%";
    const chip = row.querySelector(".chip"); chip.className = "chip " + cls; chip.textContent = label;
    batchSummary();
  }

  $("#btn-batch-add").addEventListener("click", () => batchInput.click());
  $("#btn-batch-run").addEventListener("click", runBatch);

  async function runBatch() {
    if (batchRunning) return;
    if (!batchItems.some(i => i.status !== "done")) return;
    batchRunning = true;
    $("#btn-batch-run").disabled = true; $("#btn-batch-add").disabled = true;
    for (let i = 0; i < batchItems.length; i++) {
      const it = batchItems[i];
      if (it.status === "done") continue;
      batchActiveIdx = i; it.status = "running"; it.progress = 0; it.error = null;
      renderBatch();
      try {
        const res = await API.transcribe({
          file: it.file, language: null,
          diarize: $("#sw-diar").checked, nSpeakers: $("#sel-spk").value,
          align: $("#sw-align").checked, hint: "",
        });
        it.status = "done"; it.progress = 1; it.srtPath = res.srtPath;
      } catch (err) {
        it.status = "failed"; it.progress = 0;
        it.error = (err.message || String(err)).replace(/^⚠\s*/, "");
      }
      renderBatch();
    }
    batchActiveIdx = -1; batchRunning = false;
    $("#btn-batch-run").disabled = false; $("#btn-batch-add").disabled = false;
  }

  // ════════════════════════════════════════════════════════
  // 模型與裝置
  // ════════════════════════════════════════════════════════
  async function renderModel() {
    const d = await API.listDevices();
    const host = $("#dev-list"); host.innerHTML = "";
    d.devices.forEach(dev => {
      const cpu = dev.kind === "cpu";
      const el = document.createElement("div"); el.className = "dev";
      el.innerHTML = `<span class="ic">${cpu ? ICON_CPU : ICON_GPU}</span>
        <span class="nm">${dev.kind === "cpu" ? "CPU · " : "GPU · "}${escapeHtml(dev.name)}</span>
        <span class="vram">${escapeHtml(dev.note || "")}</span>`;
      host.appendChild(el);
    });
    const diag = $("#gpu-diag");
    if (d.diag && d.diag.text) {
      diag.hidden = false;
      diag.className = "banner " + (d.diag.level === "warn" ? "banner-warn" : "banner-info");
      const txt = diag.querySelector("div"); if (txt) txt.innerHTML = escapeHtml(d.diag.text);
    } else diag.hidden = true;
  }
  $("#backend-cards").addEventListener("click", async e => {
    const card = e.target.closest(".radio-card"); if (!card) return;
    const cards = $$(".radio-card", $("#backend-cards"));
    const idx = cards.indexOf(card);
    const prevIdx = cards.findIndex(c => c.classList.contains("sel"));
    cards.forEach((c, i) => c.classList.toggle("sel", i === idx));
    const res = await API.setBackend(idx);
    // 切換核心 = 持久化選擇 + 請重啟（不熱重載）。選擇保留以反映已記住的核心。
    if (res && res.message) {
      const warn = !!res.restartRequired && res.backend !== "openvino";
      backendMsg(res.message, warn ? "warn" : "info");
    } else {
      backendMsg("", null);
    }
  });
  function backendMsg(text, level) {
    let el = $("#backend-msg");
    if (!text) { if (el) el.hidden = true; return; }
    if (!el) { el = document.createElement("div"); el.id = "backend-msg"; el.style.marginTop = "10px"; $("#backend-cards").after(el); }
    el.hidden = false;
    el.className = "banner " + (level === "warn" ? "banner-warn" : "banner-info");
    el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg><div>${escapeHtml(text)}</div>`;
  }
  const ICON_CPU = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/></svg>`;
  const ICON_GPU = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 18v2M18 18v2"/></svg>`;

  // ════════════════════════════════════════════════════════
  // 端點
  // ════════════════════════════════════════════════════════
  async function renderEndpoint() {
    const e = await API.getEndpoint();
    $("#sw-endpoint").checked = e.running;
    setEpState(e.running);
    $("#ep-port").value = e.port;
    $("#ep-url").textContent = e.running ? maskKeyInUrl(e.url) : "（服務未啟動）";
    $("#ep-url").dataset.full = e.url;
    $("#ep-key").textContent = "••••••••••••";
    $("#ep-key").dataset.full = e.key;
  }
  function setEpState(on) {
    const c = $("#ep-state");
    c.className = "chip " + (on ? "chip-ok" : "chip-muted");
    c.innerHTML = `<span style="font-size:9px">●</span> ${on ? "執行中" : "已停止"}`;
  }
  function maskKeyInUrl(url) { return url.replace(/k=[^&]+/, "k=•••••"); }
  $("#sw-endpoint").addEventListener("change", async e => {
    const r = await API.toggleEndpoint(e.target.checked); setEpState(r.running);
    $("#ep-url").textContent = r.running ? maskKeyInUrl(r.url) : "（服務未啟動）";
  });
  $("#btn-reveal-key").addEventListener("click", () => {
    const k = $("#ep-key"); const showing = k.dataset.shown === "1";
    k.textContent = showing ? "••••••••••••" : (k.dataset.full || "");
    k.dataset.shown = showing ? "" : "1";
    $("#btn-reveal-key").textContent = showing ? "顯示" : "隱藏";
  });
  $("#btn-newkey").addEventListener("click", async () => {
    const r = await API.regenKey(); $("#ep-key").dataset.full = r.key;
    $("#ep-key").dataset.shown = ""; $("#ep-key").textContent = "••••••••••••";
    $("#btn-reveal-key").textContent = "顯示";
    $("#ep-url").dataset.full = r.url; $("#ep-url").textContent = r.running ? maskKeyInUrl(r.url) : "（服務未啟動）";
  });
  // 複製按鈕（資料來自 dataset.full）
  document.addEventListener("click", e => {
    const btn = e.target.closest("[data-copy]"); if (!btn) return;
    const t = $(btn.dataset.copy); const val = t.dataset.full || t.textContent;
    navigator.clipboard?.writeText(val); flash(btn, "已複製");
  });
  function flash(btn, txt) { const o = btn.textContent; btn.textContent = txt; setTimeout(() => btn.textContent = o, 1200); }

  // ════════════════════════════════════════════════════════
  // 設定
  // ════════════════════════════════════════════════════════
  async function loadSettings() {
    const s = await API.getSettings();
    $("#set-scale").value = s.scale; $("#set-scale-val").textContent = s.scale + "%";
    applyScale(s.scale);
    segSet("#set-format", s.format); segSet("#set-vocab", s.vocab); segSet("#set-theme", s.theme);
    $("#set-mirror").value = s.mirror || "";
    if (s.ffmpeg) $("#set-ffmpeg").value = s.ffmpeg;
    if (s.vad != null) { $("#set-vad").value = s.vad; $("#set-vad-val").textContent = (+s.vad).toFixed(2); }
  }
  function segSet(sel, v) { $$(sel + " button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); }
  function applyScale(pct) { document.documentElement.style.fontSize = (14 * pct / 100).toFixed(1) + "px"; }

  $("#set-scale").addEventListener("input", e => {
    const v = +e.target.value; $("#set-scale-val").textContent = v + "%"; applyScale(v); API.setSettings({ scale: v });
  });
  $("#set-vad").addEventListener("input", e => {
    const v = +e.target.value; $("#set-vad-val").textContent = v.toFixed(2); API.setSettings({ vad: v });
  });
  // segmented 控制：點擊切換 + 回寫設定
  [["#set-format", "format"], ["#set-vocab", "vocab"], ["#set-theme", "theme"]].forEach(([sel, key]) => {
    $(sel).addEventListener("click", e => {
      const b = e.target.closest("button"); if (!b) return;
      $$(sel + " button").forEach(x => x.classList.toggle("on", x === b));
      API.setSettings({ [key]: b.dataset.v });
      if (key === "theme") applyTheme(b.dataset.v);
    });
  });
  $("#set-mirror").addEventListener("change", e => API.setSettings({ mirror: e.target.value.trim() }));
  function applyTheme(t) { /* 深色主題後續以 data-theme 切 token；此處先記錄 */ document.documentElement.dataset.theme = t; }

  // ── 共用工具 ────────────────────────────────────────────
  function fmtClock(sec) {
    sec = Math.max(0, Math.round(sec));
    const m = String(Math.floor(sec / 60)).padStart(2, "0");
    const s = String(sec % 60).padStart(2, "0");
    return `${m}:${s}`;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])); }

  // ── 錄製轉換：MediaRecorder + 停頓偵測(VAD)，分段上傳辨識 ──────
  //   127.0.0.1/localhost 屬安全情境，getUserMedia 可用（不需 HTTPS）。
  //   說完停頓 → 切段 → 上傳 /api/transcribe → 逐段附加即時字幕。
  const REC = {
    on: false, sec: 0, timer: null, raf: 0, firstResult: true,
    stream: null, ctx: null, analyser: null, recorder: null, chunks: [],
    speech: false, silentSince: 0, segStart: 0,
  };
  const SILENCE_MS = 2200, MIN_SEG_MS = 500, MAX_SEG_MS = 20000, VAD_THRESH = 0.014;

  $("#rec-btn").addEventListener("click", () => REC.on ? stopRec() : startRec());

  function recSupported() {
    return navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder;
  }
  async function startRec() {
    if (!recSupported()) { recNote("此環境不支援錄音 API。", true); return; }
    try {
      REC.stream = await navigator.mediaDevices.getUserMedia(
        { audio: { echoCancellation: true, noiseSuppression: true } });
    } catch (err) { recNote("無法取得麥克風：" + (err.message || err), true); return; }
    REC.ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = REC.ctx.createMediaStreamSource(REC.stream);
    REC.analyser = REC.ctx.createAnalyser(); REC.analyser.fftSize = 1024;
    src.connect(REC.analyser);
    REC.on = true; REC.sec = 0;
    $("#rec-btn").classList.add("recording");
    const lw = $("#rec-live-wave"); lw.hidden = false;
    lw.innerHTML = Array.from({ length: 28 }, () => `<div class="b" style="height:4px"></div>`).join("");
    $("#rec-timer").textContent = "00:00";
    REC.timer = setInterval(() => { REC.sec++; $("#rec-timer").textContent = fmtClock(REC.sec); }, 1000);
    recNote("聆聽中…說完停頓約 2 秒會自動辨識；再按一次結束並完成最後一段。");
    startSeg(); monitor();
  }
  function startSeg() {
    REC.chunks = []; REC.speech = false; REC.silentSince = 0; REC.segStart = Date.now();
    const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"]
      .find(m => MediaRecorder.isTypeSupported(m)) || "";
    REC.recorder = mime ? new MediaRecorder(REC.stream, { mimeType: mime })
                        : new MediaRecorder(REC.stream);
    REC.recorder.ondataavailable = e => { if (e.data && e.data.size) REC.chunks.push(e.data); };
    REC.recorder.onstop = onSegStop;
    REC.recorder.start();
  }
  function cutSeg() { if (REC.recorder && REC.recorder.state === "recording") REC.recorder.stop(); }
  async function onSegStop() {
    const dur = Date.now() - REC.segStart;
    const blob = new Blob(REC.chunks, { type: REC.chunks[0] ? REC.chunks[0].type : "audio/webm" });
    const ok = REC.speech && dur >= MIN_SEG_MS && blob.size > 1200;
    if (REC.on) startSeg(); else teardownRec();    // 先續錄，避免上傳期間漏話
    if (ok) {
      try {
        const file = new File([blob], "recording.webm", { type: blob.type });
        const res = await API.transcribe({ file, diarize: false, align: false });
        (res.segments || []).forEach(s => { if (s.text && s.text.trim()) appendRecLine(s.text.trim()); });
      } catch (err) { recNote("辨識失敗：" + (err.message || err), true); }
    }
  }
  function monitor() {
    const buf = new Uint8Array(REC.analyser.fftSize);
    const bars = $$(".b", $("#rec-live-wave"));
    const tick = () => {
      if (!REC.on) return;
      REC.analyser.getByteTimeDomainData(buf);
      let sum = 0; for (let i = 0; i < buf.length; i++) { const v = (buf[i] - 128) / 128; sum += v * v; }
      const rms = Math.sqrt(sum / buf.length);
      const base = Math.min(34, rms * 260);
      bars.forEach(b => b.style.height = Math.max(4, base * (0.5 + Math.random() * 0.5)).toFixed(0) + "px");
      const now = Date.now();
      if (rms >= VAD_THRESH) { REC.speech = true; REC.silentSince = 0; }
      else if (REC.speech) {
        if (!REC.silentSince) REC.silentSince = now;
        else if (now - REC.silentSince >= SILENCE_MS) cutSeg();   // 停頓 → 切段
      }
      if (Date.now() - REC.segStart >= MAX_SEG_MS && REC.speech) cutSeg();
      REC.raf = requestAnimationFrame(tick);
    };
    REC.raf = requestAnimationFrame(tick);
  }
  function stopRec() {
    REC.on = false; $("#rec-btn").classList.remove("recording");
    if (REC.raf) cancelAnimationFrame(REC.raf);
    clearInterval(REC.timer);
    $("#rec-live-wave").hidden = true;
    recNote("已停止。");
    cutSeg();    // 收尾段（onSegStop 會上傳並 teardown）
  }
  function teardownRec() {
    try { REC.stream && REC.stream.getTracks().forEach(t => t.stop()); } catch (e) {}
    try { REC.ctx && REC.ctx.close(); } catch (e) {}
    REC.stream = REC.ctx = REC.analyser = REC.recorder = null;
  }
  function recNote(msg, warn) {
    const el = $(".rec-note");
    if (el) { el.textContent = msg; el.style.color = warn ? "var(--live)" : "var(--muted)"; }
  }
  function appendRecLine(text) {
    const host = $("#rec-subs");
    if (REC.firstResult) { host.innerHTML = ""; REC.firstResult = false; }
    const el = document.createElement("div");
    el.className = "sub-card";
    el.innerHTML = `<span class="tc">${fmtClock(REC.sec)}</span>
      <div class="body"><div class="txt">${escapeHtml(text)}</div></div>`;
    host.appendChild(el);
    host.scrollTop = host.scrollHeight;
  }

  // ── 啟動 ────────────────────────────────────────────────
  API.on("status", refreshStatus);          // 桌面背景載入完成 → 更新就緒燈
  (async function init() {
    await API.ready;
    await refreshStatus();
    await loadSettings();
    await renderEndpoint();
    // 預設視圖已是「音檔」；其餘視圖切換時才渲染
    console.info("[app] ready, mode =", API.mode);
  })();
})();
