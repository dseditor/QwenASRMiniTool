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
      $("#btn-verify").disabled = false;
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

  API.on("progress", ({ pct, status }) => showProgress(pct, status));
  function showProgress(pct, status) {
    $("#progress").hidden = false;
    $("#prog-bar").style.width = pct + "%";
    $("#prog-pct").textContent = Math.round(pct) + "%";
    if (status != null) $("#prog-status").textContent = status;
  }

  // Signature：波形 + 字幕卡
  function renderResult(segments) {
    $("#result").hidden = false;
    const dur = segments.length ? segments[segments.length - 1].end : 60;
    $("#wave-dur").textContent = fmtClock(dur);
    drawWaveform($("#wave"), 0.55);            // 預設播放頭在 55%
    const host = $("#subs"); host.innerHTML = "";
    segments.forEach((s, i) => host.appendChild(subCard(s, i === 3)));
  }
  function subCard(s, playing) {
    const el = document.createElement("div");
    el.className = "sub-card" + (playing ? " playing" : "");
    const spk = s.speaker ? `<span class="chip spk-${((s.speaker - 1) % 3) + 1}">說話者 ${s.speaker}</span>` : "";
    el.innerHTML = `<span class="tc">${fmtClock(s.start)} → ${fmtClock(s.end)}</span>
      <div class="body"><div class="spk">${spk}</div><div class="txt">${escapeHtml(s.text)}</div></div>`;
    return el;
  }

  // 波形：產生穩定的擬語音柱狀（非真實音訊，視覺用）
  function drawWaveform(container, playRatio) {
    const N = 130, played = Math.floor(N * playRatio);
    let html = `<div class="playhead" style="left:${(playRatio * 100).toFixed(1)}%"></div>`;
    let seed = 7;
    const rnd = () => (seed = (seed * 9301 + 49297) % 233280) / 233280;
    for (let i = 0; i < N; i++) {
      // 以多個正弦疊加製造語音般的起伏，再加雜訊
      const env = Math.abs(Math.sin(i / 9) * 0.6 + Math.sin(i / 3.3) * 0.3) + rnd() * 0.35;
      const h = Math.max(6, Math.min(64, env * 60));
      html += `<div class="b${i < played ? " played" : ""}" style="height:${h.toFixed(0)}px"></div>`;
    }
    container.innerHTML = html;
  }

  $("#btn-open-dir").addEventListener("click", () => API.openOutputDir());

  // ── 記錄 ────────────────────────────────────────────────
  function logLine(msg) {
    const el = $("#file-log"); el.hidden = false;
    const d = document.createElement("div"); d.className = "ln"; d.textContent = msg;
    el.appendChild(d); el.scrollTop = el.scrollHeight;
  }

  // ════════════════════════════════════════════════════════
  // 批次
  // ════════════════════════════════════════════════════════
  const ST = {
    done: ["完成", "chip-ok"], running: ["辨識中", "chip-accent"],
    pending: ["待處理", "chip-muted"], failed: ["失敗", "chip-live"],
  };
  async function renderBatch() {
    const b = await API.getBatch();
    $("#batch-summary").textContent = `${b.summary.done} / ${b.summary.total} 完成`;
    const host = $("#batch-list"); host.innerHTML = "";
    b.items.forEach(it => {
      const [label, cls] = ST[it.status] || ST.pending;
      const row = document.createElement("div");
      row.className = "b-row";
      row.innerHTML = `
        <div class="nm"><div class="f">${escapeHtml(it.name)}</div>
          ${it.error ? `<div class="e">${escapeHtml(it.error)}</div>` : ""}</div>
        <div class="pbar"><div class="bar"><i style="width:${Math.round((it.progress || 0) * 100)}%"></i></div></div>
        <span class="chip ${cls}">${label}</span>
        <button class="more" title="更多">⋯</button>`;
      host.appendChild(row);
    });
  }
  $("#btn-batch-add").addEventListener("click", async () => { await API.addBatchFiles(); renderBatch(); });
  $("#btn-batch-run").addEventListener("click", async () => { await API.runBatch(); });

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
  $("#backend-cards").addEventListener("click", e => {
    const card = e.target.closest(".radio-card"); if (!card) return;
    $$(".radio-card", $("#backend-cards")).forEach((c, i) => {
      const sel = c === card; c.classList.toggle("sel", sel); if (sel) API.setBackend(i);
    });
  });
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
  }
  function segSet(sel, v) { $$(sel + " button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); }
  function applyScale(pct) { document.documentElement.style.fontSize = (14 * pct / 100).toFixed(1) + "px"; }

  $("#set-scale").addEventListener("input", e => {
    const v = +e.target.value; $("#set-scale-val").textContent = v + "%"; applyScale(v); API.setSettings({ scale: v });
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

  // ── 錄製（展示用動效；實機錄音由桌面/端點各自處理）──────
  let recOn = false, recSec = 0, recTimer = null, waveTimer = null;
  $("#rec-btn").addEventListener("click", () => recOn ? stopRec() : startRec());
  function startRec() {
    recOn = true; recSec = 0;
    $("#rec-btn").classList.add("recording");
    const lw = $("#rec-live-wave"); lw.hidden = false;
    lw.innerHTML = Array.from({ length: 28 }, () => `<div class="b" style="height:6px"></div>`).join("");
    recTimer = setInterval(() => { recSec++; $("#rec-timer").textContent = fmtClock(recSec); }, 1000);
    waveTimer = setInterval(() => {
      $$(".b", lw).forEach(b => b.style.height = (6 + Math.random() * 30).toFixed(0) + "px");
    }, 120);
  }
  function stopRec() {
    recOn = false; $("#rec-btn").classList.remove("recording");
    clearInterval(recTimer); clearInterval(waveTimer);
    $("#rec-live-wave").hidden = true;
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
