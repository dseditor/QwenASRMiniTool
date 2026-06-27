/* ============================================================
   app.js — UI 邏輯（與後端解耦，全程只透過 window.QwenAPI）
   ============================================================ */
(function () {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const API = window.QwenAPI;
  // i18n 取字（含 {n} 等簡單插值）；無字典或無此鍵時回退預設值 def。
  function T(key, def, vars) {
    let s = (window.I18N && I18N.t(key)) || def || key;
    if (vars) for (const k in vars) s = s.replace("{" + k + "}", vars[k]);
    return s;
  }

  const VIEW_TITLES = {
    file: "音檔轉字幕", batch: "批次辨識", record: "錄製轉換",
    endpoint: "端點服務", model: "模型與裝置", settings: "設定",
  };

  // ── 導覽切換 ────────────────────────────────────────────
  let _curView = "file";
  function viewTitle(name) { return (window.I18N && I18N.t("view." + name)) || VIEW_TITLES[name] || ""; }
  function refreshViewTitle() { $("#view-title").textContent = viewTitle(_curView); }
  function switchView(name) {
    _curView = name;
    $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.view === name));
    $$(".view").forEach(v => v.classList.toggle("active", v.dataset.view === name));
    $("#view-title").textContent = viewTitle(name);
    $("#view-ctx").textContent = "";
    if (name === "batch") renderBatch();
    if (name === "model") renderModel();
    if (name === "record") { enumerateMics(); refreshMicPerm(); }   // 列舉裝置 + 麥克風權限狀態
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
    $(".t", el).textContent = s.modelReady ? T("status.ready", "模型已就緒")
      : (s.loading ? T("status.loading", "載入模型中…") : T("status.needModel", "尚未載入模型"));
    // 版本徽章：純數字版本前綴 v（如 v1.0.9）；已含文字者（如 webview 0.1）原樣顯示
    if (s.version) $("#app-version").textContent = /^\d/.test(s.version) ? "v" + s.version : s.version;
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

  // 讀入 TXT：瀏覽器沙箱用隱藏 <input type=file> + FileReader 讀文字進提示框
  const _txtInput = document.createElement("input");
  _txtInput.type = "file"; _txtInput.accept = ".txt,text/plain"; _txtInput.style.display = "none";
  document.body.appendChild(_txtInput);
  _txtInput.addEventListener("change", () => {
    const f = _txtInput.files && _txtInput.files[0]; if (!f) return;
    const r = new FileReader();
    r.onload = () => { $("#hint-box").value = String(r.result || ""); };
    r.readAsText(f, "utf-8");
    _txtInput.value = "";
  });
  $("#btn-load-txt").addEventListener("click", () => _txtInput.click());

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
        language: $("#sel-lang").value || null,
        diarize: $("#sw-diar").checked,
        nSpeakers: $("#sel-spk").value,
        align: $("#sw-align").checked,
        hint: $("#hint-box").value.trim(),
      });
      renderResult(res.segments);
      logLine("✓ 完成，共 " + res.segments.length + " 段" + (res.srtPath ? "，已輸出 " + res.srtPath : ""));
      $("#btn-open-dir").disabled = false;
      $("#btn-save-sub").disabled = false;
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

  // 進度分派：模型就地載入中 → 模型頁進度；批次執行中 → 當前批次列；否則 → 單檔進度條
  API.on("progress", ({ pct, status }) => {
    if (_modelLoading) {
      setModelProg(pct, status);
    } else if (batchRunning && batchActiveIdx >= 0) {
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
    syncStickyOffset();
  }

  // 波形面板為 sticky 釘在頂端 → 捲動容器需保留等高的頂部空間，
  // 否則 scrollIntoView 自動捲到的字幕卡會被面板遮住。
  function syncStickyOffset() {
    const panel = $(".wave-panel"), host = $(".view-host");
    if (!panel || !host) return;
    host.style.scrollPaddingTop = (panel.offsetHeight + 12) + "px";
  }
  window.addEventListener("resize", syncStickyOffset);

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
    // spkc-N：依說話者標示左側外框色（與 chip 同色系，不污染卡片背景）
    el.className = "sub-card" + (s.speaker ? " spkc-" + (((s.speaker - 1) % 3) + 1) : "");
    el.dataset.seg = idx;
    const spk = s.speaker ? `<span class="chip spk-${((s.speaker - 1) % 3) + 1}">說話者 ${s.speaker}</span>` : "";
    el.innerHTML = `<span class="tc">${fmtClock(s.start)} → ${fmtClock(s.end)}</span>
      <div class="body"><div class="spk">${spk}</div><div class="txt">${escapeHtml(s.text)}</div></div>
      <button class="sub-edit" title="編輯此行" aria-label="編輯此行">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>
      </button>`;
    el.addEventListener("click", e => {
      if (e.target.closest(".sub-edit") || el.classList.contains("editing")) return;
      seekTo(s.start);
    });
    el.querySelector(".sub-edit").addEventListener("click", e => { e.stopPropagation(); beginEdit(el, idx); });
    return el;
  }

  // 行內校對：筆 → 該行可編輯，Enter 儲存 / Esc 取消；同步回 curSegs 與波形對齊區塊。
  // curSegs 為下載字幕的資料來源，故編輯後存檔即反映校對結果。
  function beginEdit(card, idx) {
    if (card.classList.contains("editing")) return;
    card.classList.add("editing");
    const txt = card.querySelector(".txt");
    const orig = curSegs[idx].text;
    txt.contentEditable = "true"; txt.spellcheck = false;
    txt.focus();
    const range = document.createRange(); range.selectNodeContents(txt);
    const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    function finish(save) {
      txt.removeEventListener("keydown", onKey);
      txt.removeEventListener("blur", onBlur);
      txt.contentEditable = "false";
      card.classList.remove("editing");
      const val = (txt.textContent || "").trim();
      if (save && val && val !== orig) {
        curSegs[idx].text = val; txt.textContent = val;
        const blk = $(`.seg-block[data-seg="${idx}"]`);
        if (blk) { blk.textContent = val; blk.title = val; }
      } else {
        txt.textContent = orig;               // 取消或清空 → 還原
      }
      txt.blur();
    }
    function onKey(e) {
      if (e.key === "Enter") { e.preventDefault(); finish(true); }
      else if (e.key === "Escape") { e.preventDefault(); finish(false); }
    }
    function onBlur() { finish(true); }
    txt.addEventListener("keydown", onKey);
    txt.addEventListener("blur", onBlur);
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
    // 停止：暫停並回到開頭（像播放器的 stop）
    const stop = $("#wave-stop");
    if (stop) stop.onclick = () => { if (!audioEl) return; audioEl.pause(); audioEl.currentTime = 0; onTime(); };

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

  // ── 字幕存檔：由記憶體中的 curSegs 在前端組檔，下載到使用者指定位置 ──
  //   （瀏覽器沙箱不暴露來源檔真實路徑，無法寫回來源資料夾，故走下載。
  //    SRT/TXT 格式比照 subtitle_lines，含「說話者N：」前綴。）
  function srtTs(s) {
    const ms = Math.round(s * 1000);
    const h = String(Math.floor(ms / 3600000)).padStart(2, "0");
    const m = String(Math.floor((ms % 3600000) / 60000)).padStart(2, "0");
    const sec = String(Math.floor((ms % 60000) / 1000)).padStart(2, "0");
    return `${h}:${m}:${sec},${String(ms % 1000).padStart(3, "0")}`;
  }
  function buildSrt(segs) {
    return segs.map((s, i) => {
      const spk = s.speaker ? `說話者${s.speaker}：` : "";
      return `${i + 1}\n${srtTs(s.start)} --> ${srtTs(s.end)}\n${spk}${s.text}\n`;
    }).join("\n");
  }
  function buildTxt(segs) {
    return segs.some(s => s.speaker)
      ? segs.map(s => (s.speaker ? `說話者${s.speaker}：` : "") + s.text).join("\n")
      : segs.map(s => s.text).join("");
  }
  async function saveSubtitle() {
    if (!curSegs.length) return;
    let fmt = "srt";
    try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
    const text = fmt === "txt" ? buildTxt(curSegs) : buildSrt(curSegs);
    const base = (picked && picked.name) ? picked.name.replace(/\.[^.]+$/, "") : "字幕";
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = base + (fmt === "txt" ? ".txt" : ".srt");
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    logLine("✓ 已存檔字幕：" + a.download);
  }
  $("#btn-save-sub").addEventListener("click", saveSubtitle);

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
          file: it.file, language: $("#sel-lang").value || null,
          diarize: $("#sw-batch-diar").checked, nSpeakers: $("#sel-batch-spk").value,
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
  // 模型與裝置（核心 + 模型下拉 + 架構標籤）
  // ════════════════════════════════════════════════════════
  let _modelOpt = null;     // 最近一次 getModelOptions 結果
  let _curCore = null;      // 目前選中的核心標籤

  // ── 系統自檢：每核心 × 每能力，顯示 Green/Yellow/Red ──────────────
  async function renderHealth() {
    let h;
    try { h = await API.getHealthCheck(); } catch (e) { return; }
    const sum = $("#health-sum");
    if (h.summary) {
      if (h.summary.red > 0) { sum.className = "chip chip-live"; sum.textContent = `${h.summary.red} 項需處理`; }
      else if (h.summary.yellow > 0) { sum.className = "chip chip-accent"; sum.textContent = `核心就緒 · ${h.summary.yellow} 項將於啟用時自動下載`; }
      else { sum.className = "chip chip-ok"; sum.textContent = "全部就緒"; }
    }
    const host = $("#health-panel"); host.innerHTML = "";
    const dot = s => `<span class="hdot hdot-${s}"></span>`;
    const groups = [...(h.cores || [])];
    if (h.shared && h.shared.length) groups.push({ label: "共用元件", items: h.shared });
    groups.forEach(g => {
      const card = document.createElement("div"); card.className = "health-core";
      const active = g.backend && g.backend === h.activeBackend;
      card.innerHTML = `<div class="hc-title">${escapeHtml(g.label)}` +
        (active ? ` <span class="chip chip-ok" style="font-size:10px">使用中</span>` : ``) + `</div>`;
      (g.items || []).forEach(it => {
        const row = document.createElement("div"); row.className = "hc-item";
        row.innerHTML = `${dot(it.status)}<span class="hi-label">${escapeHtml(it.label)}</span>` +
          `<span class="hi-detail">${escapeHtml(it.detail || "")}</span>`;
        card.appendChild(row);
      });
      host.appendChild(card);
    });
  }
  $("#btn-health-recheck") && $("#btn-health-recheck").addEventListener("click", renderHealth);

  async function renderModel() {
    renderHealth();
    // 核心卡 + 模型下拉
    try {
      _modelOpt = await API.getModelOptions();
      _curCore = _modelOpt.current.core;
      renderCoreCards();
      populateModels(_curCore, _modelOpt.current.model);
      // 已記住的選擇 vs 實際載入的架構不同 → 提示重啟
      const curArch = selectedArch();
      if (_modelOpt.activeArch && curArch && _modelOpt.activeArch !== curArch) {
        modelMsg(`已記住「${_curCore} · ${$("#model-select").value}」（${curArch}），`
          + `將於重新啟動後套用（目前以 ${_modelOpt.activeArch} 運行）。`, "info");
      } else modelMsg("", null);
      updateLoadBtn();              // 設定「下載並載入／前往轉錄」按鈕初始狀態
    } catch (e) {}
    // 偵測裝置 + 診斷
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

  function coreObj(label) {
    const cs = _modelOpt && _modelOpt.cores || [];
    return cs.find(c => c.label === label) || cs[0];
  }
  function renderCoreCards() {
    const host = $("#core-cards"); host.innerHTML = "";
    (_modelOpt.cores || []).forEach(core => {
      const card = document.createElement("label");
      card.className = "radio-card" + (core.label === _curCore ? " sel" : "");
      card.dataset.core = core.label;
      card.innerHTML = `<span class="rd"></span><div class="info">
        <div class="t">${escapeHtml(core.label)}</div>
        <div class="d">${escapeHtml(T("model.nModels", `${core.models.length} 種模型可選`, { n: core.models.length }))}</div></div>`;
      host.appendChild(card);
    });
  }
  function populateModels(coreLabel, modelLabel) {
    const core = coreObj(coreLabel); if (!core) return;
    const sel = $("#model-select"); sel.innerHTML = "";
    core.models.forEach(m => {
      const o = document.createElement("option");
      o.value = m.label; o.textContent = m.label; o.dataset.arch = m.arch;
      o.dataset.note = m.note || "";
      if (m.label === modelLabel) o.selected = true;
      sel.appendChild(o);
    });
    if (!core.models.some(m => m.label === modelLabel) && core.models[0]) sel.value = core.models[0].label;
    updateArch();
  }
  function selectedArch() { const o = $("#model-select").selectedOptions[0]; return o ? o.dataset.arch : ""; }
  function updateArch() {
    const o = $("#model-select").selectedOptions[0];
    $("#model-arch").textContent = (o && o.dataset.arch) || "";
    const note = o && o.dataset.note, el = $("#model-note");   // chatllm AMD 等提醒
    if (note) { el.hidden = false; el.textContent = note; } else { el.hidden = true; el.textContent = ""; }
  }

  // 點核心卡 → 切核心、模型回該核心首項並套用
  $("#core-cards").addEventListener("click", async e => {
    const card = e.target.closest(".radio-card"); if (!card) return;
    _curCore = card.dataset.core;
    $$(".radio-card", $("#core-cards")).forEach(c => c.classList.toggle("sel", c === card));
    const core = coreObj(_curCore);
    const first = core && core.models[0] ? core.models[0].label : "";
    populateModels(_curCore, first);
    await applyModel(_curCore, first);
  });
  // 換模型 → 套用
  $("#model-select").addEventListener("change", async e => {
    updateArch();
    await applyModel(_curCore, e.target.value);
  });
  let _modelLoading = false;       // 模型「就地下載並載入」進行中（進度導到模型頁）
  async function applyModel(core, model) {
    try {
      const res = await API.setModel(core, model);
      if (res && res.message) modelMsg(res.message, res.restartRequired ? "warn" : "info");
      else modelMsg("", null);
      updateLoadBtn(res);
    } catch (err) { modelMsg("套用失敗：" + (err.message || err), "warn"); }
  }
  // 依 setModel 結果 + 目前載入狀態，決定「下載並載入」按鈕的文字/可用性
  async function updateLoadBtn(res) {
    const btn = $("#btn-model-load"); if (!btn) return;
    let st = {};
    try { st = await API.getStatus(); } catch (e) {}
    btn.classList.remove("btn-ghost");
    if (_modelLoading || st.loading) {
      btn.disabled = true; btn.textContent = T("model.loading", "下載／載入中…"); return;
    }
    if (st.modelReady && !(res && res.restartRequired)) {
      // 已就緒且未要求重啟 → 引導前往轉錄
      btn.disabled = false; btn.textContent = T("model.goto", "前往語音轉文字"); btn.dataset.act = "goto"; return;
    }
    if (res && res.restartRequired) {
      // 已載入其他核心、切換需重啟 → 按鈕反白（不可就地載入）
      btn.disabled = true; btn.textContent = T("model.needRestart", "切換核心需重新啟動"); btn.dataset.act = ""; return;
    }
    // 尚未載入 → 可就地下載並載入
    btn.disabled = false; btn.textContent = T("model.load", "下載並載入模型"); btn.dataset.act = "load";
  }
  $("#btn-model-load") && $("#btn-model-load").addEventListener("click", async () => {
    const btn = $("#btn-model-load");
    if (btn.dataset.act === "goto") { switchView("file"); return; }
    _modelLoading = true; btn.disabled = true; btn.textContent = "下載／載入中…";
    $("#model-progress").hidden = false;
    setModelProg(0, "準備載入…");
    modelMsg("正在下載並載入模型，請稍候——可留在本頁觀看進度。", "info");
    try { await API.startLoad(); }
    catch (err) { _modelLoading = false; modelMsg("載入失敗：" + (err.message || err), "warn"); updateLoadBtn(); }
  });
  function setModelProg(pct, status) {
    $("#model-prog-bar").style.width = pct + "%";
    $("#model-prog-pct").textContent = Math.round(pct) + "%";
    if (status != null) $("#model-prog-status").textContent = status;
  }
  function modelMsg(text, level) {
    const el = $("#model-msg");
    if (!text) { el.hidden = true; el.innerHTML = ""; return; }
    el.hidden = false; el.style.marginTop = "10px";
    el.className = "banner " + (level === "warn" ? "banner-warn" : "banner-info");
    el.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg><div>${escapeHtml(text)}</div>`;
  }
  const ICON_CPU = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/></svg>`;
  const ICON_GPU = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 18v2M18 18v2"/></svg>`;

  // ── 辨識語言下拉（依已載入引擎；OpenVINO 用 processor 清單）────────
  async function loadLanguages() {
    try {
      const { languages } = await API.getLanguages();
      // 音檔頁與錄製頁各有自己的語言下拉，皆依目前引擎填同一份清單。
      ["#sel-lang", "#rec-lang"].forEach(id => {
        const sel = $(id); if (!sel) return;
        const cur = sel.value;
        sel.innerHTML = "";
        languages.forEach(l => {
          const o = document.createElement("option");
          o.value = l.value; o.textContent = l.value ? l.label : "語言 自動";
          sel.appendChild(o);
        });
        if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
      });
    } catch (e) {}
  }

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
    applyEpQr(e);
    renderTunnel();
  }
  // 上傳網頁 QR（區網）：服務啟動且有網址才顯示
  function applyEpQr(e) {
    const qr = $("#ep-qr"), src = e.running ? API.qrSrc(e.url) : "";
    if (src) { qr.src = src; qr.hidden = false; } else { qr.hidden = true; }
    $("#ep-qr-hint").textContent = e.running ? "同網段裝置可掃此 QR 直接開啟上傳頁" : "（服務未啟動）";
  }

  // ── 對外臨時網址（Cloudflare）─────────────────────────────
  async function renderTunnel() { applyTunnelState(await API.getTunnel()); }
  function applyTunnelState(t) {
    $("#sw-tunnel").checked = !!t.running;
    const ready = !!(t.running && t.url);
    const st = $("#cf-state");
    st.className = "chip " + (ready ? "chip-ok" : (t.running ? "chip-accent" : "chip-muted"));
    st.innerHTML = `<span style="font-size:9px">●</span> ${ready ? "對外中" : (t.running ? "建立中…" : "未啟用")}`;
    $("#cf-body").hidden = !t.running;
    if (t.url) {
      $("#cf-url").textContent = maskKeyInUrl(t.url); $("#cf-url").dataset.full = t.url;
      const src = API.qrSrc(t.url), qr = $("#cf-qr");
      if (src) { qr.src = src; qr.hidden = false; } else qr.hidden = true;
    } else {
      $("#cf-url").textContent = "（建立中…）"; $("#cf-url").dataset.full = "";
      $("#cf-qr").hidden = true;
    }
    const cs = $("#cf-status");
    cs.className = "cf-status" + (t.error ? " err" : (t.status === "ready" ? " ok" : ""));
    cs.textContent = t.error ? t.status
      : (t.status === "ready" ? "✅ 已建立對外網址（含金鑰，等同密碼）— 用完請關閉" : (t.status || ""));
  }
  $("#sw-tunnel").addEventListener("change", async e => {
    const on = e.target.checked;
    if (on && !$("#sw-endpoint").checked) {        // 通道須先有端點服務
      e.target.checked = false;
      const cs = $("#cf-status"); cs.className = "cf-status err"; cs.textContent = "請先啟動端點服務";
      return;
    }
    applyTunnelState(await API.toggleTunnel(on));
  });
  // 後端建立通道為非同步：狀態/網址就緒時由 SSE "tunnel" 事件推回
  API.on("tunnel", t => applyTunnelState(t));
  function setEpState(on) {
    const c = $("#ep-state");
    c.className = "chip " + (on ? "chip-ok" : "chip-muted");
    c.innerHTML = `<span style="font-size:9px">●</span> ${on ? "執行中" : "已停止"}`;
  }
  function maskKeyInUrl(url) { return url.replace(/k=[^&]+/, "k=•••••"); }
  $("#sw-endpoint").addEventListener("change", async e => {
    await API.toggleEndpoint(e.target.checked, $("#ep-port").value.trim());
    renderEndpoint();                       // 重新同步埠/網址/金鑰（換埠會重建服務）
  });
  // 監聽埠變更：服務執行中 → 以新埠重建並刷新；未啟動 → 保留輸入值，啟動時才採用
  $("#ep-port").addEventListener("change", async () => {
    if (!$("#sw-endpoint").checked) return;
    await API.toggleEndpoint(true, $("#ep-port").value.trim());
    renderEndpoint();
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
    segSet("#set-format", s.format); segSet("#set-vocab", s.vocab); segSet("#set-theme", s.theme || "light");
    applyTheme(s.theme || "light");                 // 啟動即套用深/淺色
    $("#set-mirror").value = s.mirror || "";
    $("#set-ffmpeg").value = s.ffmpeg || "";
    if (s.vad != null) { $("#set-vad").value = s.vad; $("#set-vad-val").textContent = (+s.vad).toFixed(2); }
    // 介面語言：選單回填 + 套用 i18n（含目前視圖標題）
    const uiLang = s.uiLang || "繁體中文";
    if ([...$("#set-lang").options].some(o => o.value === uiLang)) $("#set-lang").value = uiLang;
    if (window.I18N) { I18N.setLang(uiLang); refreshViewTitle(); }
  }
  function segSet(sel, v) { $$(sel + " button").forEach(b => b.classList.toggle("on", b.dataset.v === v)); }
  // 介面縮放：用 CSS zoom（Chromium/WebView2 支援）整體縮放，px 版面也能等比生效。
  function applyScale(pct) { document.body.style.zoom = (Math.max(50, Math.min(200, +pct || 100)) / 100); }

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
  // 外觀主題：淺/深/跟隨系統。system 解析成實際 light/dark 掛到 <html>，並監聽 OS 變化。
  // 視窗標題列深淺由後端（app_webview）依同一設定同步，故只需把偏好寫回 setSettings。
  let _mqHandler = null;
  function resolveTheme(t) {
    if (t === "dark" || t === "light") return t;
    return (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches) ? "dark" : "light";
  }
  function applyTheme(t) {
    document.documentElement.dataset.theme = resolveTheme(t);   // 實際 light/dark
    document.documentElement.dataset.themePref = t;             // 原始偏好
    if (window.matchMedia) {
      const mq = window.matchMedia("(prefers-color-scheme: dark)");
      if (_mqHandler) { try { mq.removeEventListener("change", _mqHandler); } catch (e) {} _mqHandler = null; }
      if (t === "system") {
        _mqHandler = () => { document.documentElement.dataset.theme = resolveTheme("system"); };
        try { mq.addEventListener("change", _mqHandler); } catch (e) {}
      }
    }
  }
  $("#set-ffmpeg").addEventListener("change", e => API.setSettings({ ffmpeg: e.target.value.trim() }));
  // 介面語言切換：寫回設定 + 即時套用 i18n（含目前視圖標題、模型頁動態文字）
  $("#set-lang").addEventListener("change", e => {
    const v = e.target.value;
    API.setSettings({ uiLang: v });
    if (window.I18N) { I18N.setLang(v); refreshViewTitle(); }
    if (_curView === "model") renderModel();   // 重繪動態文字（核心卡描述等）
  });
  // 檢查更新：開啟 GitHub Releases 頁（系統瀏覽器）
  $("#btn-check-update") && $("#btn-check-update").addEventListener("click", async () => {
    const btn = $("#btn-check-update"); const o = btn.textContent;
    btn.disabled = true; btn.textContent = "開啟中…";
    try { await API.checkUpdate(); } catch (e) {}
    setTimeout(() => { btn.disabled = false; btn.textContent = o; }, 1500);
  });

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
    segs: [],            // 累積辨識結果（含合成時間軸），供清除／存檔／即時存檔
    fileHandle: null,    // 即時存檔的 File System Access 檔案 handle（若支援）
  };
  const SILENCE_MS = 2200, MIN_SEG_MS = 500, MAX_SEG_MS = 20000, VAD_THRESH = 0.014;
  const REC_LINE_SECS = 5;     // 錄製無精確時間戳 → 每段合成固定 5s（比照 app.py）

  $("#rec-btn").addEventListener("click", () => REC.on ? stopRec() : startRec());

  function recSupported() {
    return navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder;
  }

  // ── 麥克風權限狀態（整合進頁面，而非僅依賴瀏覽器彈窗）──────────────
  //   瀏覽器原生「允許麥克風」提示由 WebView2/Edge 控制、無法被頁面取代；
  //   但我們用 Permissions API 把「已授權／將詢問／被拒」狀態與指引顯示在頁內，
  //   被拒時給明確的頁內排除說明，不再只是一行錯誤。
  async function refreshMicPerm() {
    const chip = $("#rec-perm"), help = $("#rec-perm-help");
    if (!chip) return;
    if (!recSupported()) {
      chip.hidden = false; chip.className = "chip chip-muted"; chip.textContent = "不支援錄音";
      return;
    }
    let state = "prompt";
    try {
      if (navigator.permissions && navigator.permissions.query) {
        const st = await navigator.permissions.query({ name: "microphone" });
        state = st.state;                       // granted / denied / prompt
        st.onchange = () => refreshMicPerm();    // 狀態變更即時更新
      }
    } catch (e) { state = "prompt"; }
    chip.hidden = false;
    if (state === "granted") {
      chip.className = "chip chip-ok"; chip.textContent = T("record.permGranted", "麥克風已授權");
      if (help) help.hidden = true;
    } else if (state === "denied") {
      chip.className = "chip chip-live"; chip.textContent = T("record.permDeniedShort", "麥克風被拒");
      if (help) help.hidden = false;
    } else {
      chip.className = "chip chip-muted"; chip.textContent = T("record.permPrompt", "按麥克風時將請求授權");
      if (help) help.hidden = true;
    }
  }

  // ── 麥克風裝置列舉（瀏覽器原生；對應 app.py 的 sounddevice 裝置選擇）──────
  //   裝置標籤在未授權前為空字串 → 首次 getUserMedia 後再列舉一次才有名稱。
  async function enumerateMics() {
    const sel = $("#rec-mic");
    if (!sel || !navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    try {
      const cur = sel.value;
      const devs = await navigator.mediaDevices.enumerateDevices();
      const mics = devs.filter(d => d.kind === "audioinput");
      sel.innerHTML = "";
      const def = document.createElement("option");
      def.value = ""; def.textContent = "預設麥克風";
      sel.appendChild(def);
      mics.forEach((d, i) => {
        const o = document.createElement("option");
        o.value = d.deviceId;
        o.textContent = d.label || `麥克風 ${i + 1}`;
        sel.appendChild(o);
      });
      if ([...sel.options].some(o => o.value === cur)) sel.value = cur;
    } catch (e) { /* 列舉失敗 → 保留預設選項 */ }
  }
  $("#rec-mic-refresh") && $("#rec-mic-refresh").addEventListener("click", enumerateMics);
  if (navigator.mediaDevices) {
    try { navigator.mediaDevices.addEventListener("devicechange", enumerateMics); } catch (e) {}
  }

  async function startRec() {
    if (!recSupported()) { recNote("此環境不支援錄音 API。", true); return; }
    // 依選定的麥克風建立音訊約束（空值 → 系統預設裝置）。
    const micId = $("#rec-mic") ? $("#rec-mic").value : "";
    const audioConstraint = { echoCancellation: true, noiseSuppression: true };
    if (micId) audioConstraint.deviceId = { exact: micId };
    try {
      REC.stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraint });
    } catch (err) {
      // 被拒 / 無裝置 → 頁內顯示權限指引，不只丟一行錯誤
      const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
      recNote((denied ? "麥克風權限被拒。" : "無法取得麥克風：") + (err.message || err), true);
      refreshMicPerm();
      if (denied && $("#rec-perm-help")) $("#rec-perm-help").hidden = false;
      return;
    }
    enumerateMics();    // 授權後重新列舉 → 取得真實裝置名稱
    refreshMicPerm();
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
        const recLang = $("#rec-lang") ? $("#rec-lang").value : "";
        const res = await API.transcribe({ file, language: recLang || null, diarize: false, align: false });
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
    // 合成時間軸：錄製無精確時間戳，每段固定 5s、段間 0.1s（比照 app.py _on_rt_save）。
    const start = REC.segs.length ? REC.segs[REC.segs.length - 1].end + 0.1 : 0;
    const seg = { start, end: start + REC_LINE_SECS, speaker: null, text };
    REC.segs.push(seg);
    const el = document.createElement("div");
    el.className = "sub-card";
    el.innerHTML = `<span class="tc">${fmtClock(REC.sec)}</span>
      <div class="body"><div class="txt">${escapeHtml(text)}</div></div>`;
    host.appendChild(el);
    host.scrollTop = host.scrollHeight;
    $("#rec-save").disabled = false;
    if (REC.fileHandle) recAutosaveWrite();    // 即時存檔：每段附加後即落盤
  }

  // ── 清除／儲存／即時存檔（對應 app.py 錄製頁的清除、儲存 SRT、即時追加保存）──
  function recClear() {
    REC.segs = []; REC.firstResult = true;
    $("#rec-subs").innerHTML = `<div class="sub-card"><span class="tc">— · —</span>`
      + `<div class="body"><div class="txt" style="color:var(--muted)">開始錄音後，辨識結果會逐段出現在這裡。</div></div></div>`;
    $("#rec-save").disabled = true;
  }
  async function recSave() {
    if (!REC.segs.length) return;
    let fmt = "srt";
    try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
    const text = fmt === "txt" ? buildTxt(REC.segs) : buildSrt(REC.segs);
    const stamp = new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "").slice(0, 14);
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `realtime_${stamp}` + (fmt === "txt" ? ".txt" : ".srt");
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
  $("#rec-clear") && $("#rec-clear").addEventListener("click", recClear);
  $("#rec-save") && $("#rec-save").addEventListener("click", recSave);

  // 即時存檔：用 File System Access API（WebView2/Edge 支援）持續把目前累積的
  // 字幕寫入使用者選定的檔案 —— 重現 app.py「即時追加保存，可隨時中斷不遺失」。
  // 不支援的環境隱藏此開關（仍可用「儲存字幕」一次性匯出）。
  const FS_SAVE_OK = typeof window.showSaveFilePicker === "function";
  if (FS_SAVE_OK) $("#rec-autosave-wrap").hidden = false;
  $("#rec-autosave") && $("#rec-autosave").addEventListener("change", async e => {
    if (!e.target.checked) { REC.fileHandle = null; return; }
    try {
      let fmt = "srt";
      try { fmt = (await API.getSettings()).format || "srt"; } catch (err) {}
      const ext = fmt === "txt" ? "txt" : "srt";
      REC.fileHandle = await window.showSaveFilePicker({
        suggestedName: `realtime.${ext}`,
        types: [{ description: "字幕檔", accept: { "text/plain": ["." + ext] } }],
      });
      await recAutosaveWrite();           // 立即寫一次（含既有片段）
      recNote("即時存檔已啟用：每段辨識完成後會自動寫入所選檔案。");
    } catch (err) {                         // 使用者取消選檔 → 還原開關
      REC.fileHandle = null; e.target.checked = false;
    }
  });
  async function recAutosaveWrite() {
    if (!REC.fileHandle) return;
    try {
      let fmt = "srt";
      try { fmt = (await API.getSettings()).format || "srt"; } catch (e) {}
      const text = fmt === "txt" ? buildTxt(REC.segs) : buildSrt(REC.segs);
      const w = await REC.fileHandle.createWritable();   // 截斷重寫（內容小，安全可靠）
      await w.write(text); await w.close();
    } catch (e) { recNote("即時存檔寫入失敗：" + (e.message || e), true); REC.fileHandle = null; }
  }

  // ── 啟動 ────────────────────────────────────────────────
  // 載入完成 → 更新就緒燈 + 語言清單（依引擎）；若為「就地下載並載入」流程 → 收尾
  API.on("status", async (s) => {
    refreshStatus(); loadLanguages();
    let ready = s && s.modelReady;
    if (ready == null) { try { ready = (await API.getStatus()).modelReady; } catch (e) {} }
    if (_modelLoading && ready) {
      _modelLoading = false;
      setModelProg(100, "完成");
      setTimeout(() => { $("#model-progress").hidden = true; }, 800);
      modelMsg("模型已就緒。已為你切換到「語音轉文字」。", "info");
      renderHealth();                 // 自檢色點刷新
      switchView("file");             // 完成 → 進入轉錄頁
    } else if (s && s.error && _modelLoading) {
      _modelLoading = false;
      $("#model-progress").hidden = true;
      modelMsg("載入失敗：" + s.error, "warn");
      updateLoadBtn();
    }
  });
  (async function init() {
    await API.ready;
    await refreshStatus();
    await loadLanguages();
    await loadSettings();
    await renderEndpoint();
    // 開始頁決策：目前選擇的模型已就緒 → 直接進「語音轉文字」；否則停在「模型」頁，
    // 讓使用者先確認硬體＋選擇模型（可改 Whisper 等），按「下載並載入」才下載。
    let ready = false;
    try { ready = !!(await API.getStatus()).selectedReady; } catch (e) {}
    switchView(ready ? "file" : "model");
    console.info("[app] ready, mode =", API.mode, "selectedReady =", ready);
  })();
})();
