/* ============================================================
   i18n.js — 介面語言（繁體中文 / 简体中文 / English）
   用法：在 HTML 元素掛 data-i18n="key"（textContent）、
        data-i18n-html（innerHTML）、data-i18n-ph（placeholder）、
        data-i18n-title（title）。呼叫 window.I18N.setLang(uiLang) 切換。
   字典僅涵蓋靜態介面字串；後端動態訊息（進度/錯誤）維持原語言。
   ============================================================ */
(function () {
  "use strict";

  // uiLang 設定值（與 #set-lang 選項一致）→ locale code
  const LANG_MAP = { "繁體中文": "hant", "简体中文": "hans", "簡體中文": "hans", "English": "en" };
  let LOCALE = "hant";

  // key: [繁體, 简体, English]
  const D = {
    "nav.file": ["音檔", "音频", "Audio"],
    "nav.batch": ["批次", "批量", "Batch"],
    "nav.record": ["錄製", "录制", "Record"],
    "nav.endpoint": ["端點", "端点", "Endpoint"],
    "nav.model": ["模型", "模型", "Model"],
    "nav.settings": ["設定", "设置", "Settings"],

    "view.file": ["音檔轉字幕", "音频转字幕", "Audio to Subtitles"],
    "view.batch": ["批次辨識", "批量识别", "Batch Transcribe"],
    "view.record": ["錄製轉換", "录制转换", "Record & Transcribe"],
    "view.endpoint": ["端點服務", "端点服务", "Endpoint Service"],
    "view.model": ["模型與裝置", "模型与设备", "Model & Devices"],
    "view.settings": ["設定", "设置", "Settings"],

    "file.dropBig": ["拖曳音訊到此，或<b>點擊選擇檔案</b>", "拖拽音频到此，或<b>点击选择文件</b>", "Drop audio here, or <b>click to choose a file</b>"],
    "file.dropSub": ["支援 mp3 / wav / m4a / 影片音軌", "支持 mp3 / wav / m4a / 视频音轨", "Supports mp3 / wav / m4a / video audio"],
    "file.run": ["開始轉換", "开始转换", "Start"],
    "file.openDir": ["開啟輸出資料夾", "打开输出文件夹", "Open output folder"],
    "file.saveSub": ["字幕存檔", "保存字幕", "Save subtitles"],
    "file.verify": ["字幕驗證", "字幕校验", "Verify"],
    "file.diar": ["說話者分離", "说话者分离", "Speaker diarization"],
    "file.align": ["時間軸對齊", "时间轴对齐", "Timestamp align"],
    "file.hintTitle": ["辨識提示（可選）", "识别提示（可选）", "Recognition hint (optional)"],
    "file.hintDesc": ["貼入歌詞、關鍵字或背景說明，可提升辨識準確度", "粘贴歌词、关键字或背景说明，可提升识别准确度", "Paste lyrics, keywords or context to improve accuracy"],
    "file.hintPh": ["例如：本段為產品行銷 Podcast，會出現「轉換率」「漏斗」等行銷術語…", "例如：本段为产品营销 Podcast，会出现“转化率”“漏斗”等营销术语…", "e.g. This is a marketing podcast with terms like 'conversion rate', 'funnel'…"],
    "file.loadTxt": ["讀入 TXT…", "读入 TXT…", "Load TXT…"],
    "file.resultTitle": ["辨識結果", "识别结果", "Result"],
    "common.langAuto": ["語言 自動", "语言 自动", "Lang Auto"],
    "common.spkAuto": ["人數 自動", "人数 自动", "Count Auto"],

    "batch.add": ["加入檔案", "添加文件", "Add files"],
    "batch.run": ["全部開始", "全部开始", "Start all"],

    "record.mic": ["麥克風", "麦克风", "Microphone"],
    "record.refresh": ["重新整理", "刷新", "Refresh"],
    "record.liveTitle": ["即時字幕", "实时字幕", "Live captions"],
    "record.liveDesc": ["說完停頓自動辨識；可清除或依輸出格式存檔", "说完停顿自动识别；可清除或按输出格式保存", "Auto-transcribes on pause; clear or save by output format"],
    "record.autosave": ["即時存檔", "实时保存", "Auto-save"],
    "record.clear": ["清除", "清除", "Clear"],
    "record.save": ["儲存字幕", "保存字幕", "Save subtitles"],
    "record.note": ["錄製轉換：偵測到說話停頓時才辨識，句中短暫停頓不會中斷。按麥克風開始，再按一次結束並完成最後一段。", "录制转换：检测到说话停顿时才识别，句中短暂停顿不会中断。按麦克风开始，再按一次结束并完成最后一段。", "Transcribes on speech pauses; brief mid-sentence pauses won't cut. Tap the mic to start, tap again to finish the last segment."],
    "record.placeholder": ["開始錄音後，辨識結果會逐段出現在這裡。", "开始录音后，识别结果会逐段出现在这里。", "After recording starts, results appear here segment by segment."],
    "record.permGranted": ["麥克風已授權", "麦克风已授权", "Microphone allowed"],
    "record.permPrompt": ["按麥克風時將請求授權", "按麦克风时将请求授权", "Will ask permission on record"],
    "record.permDenied": ["麥克風權限被拒。請在視窗網址列的權限圖示中允許麥克風，或於系統設定開放本程式的麥克風存取後再試。", "麦克风权限被拒。请在窗口地址栏的权限图标中允许麦克风，或在系统设置开放本程序的麦克风访问后再试。", "Microphone blocked. Allow it via the permission icon in the address bar, or enable mic access for this app in system settings, then retry."],
    "record.permDeniedShort": ["麥克風被拒", "麦克风被拒", "Mic blocked"],

    "ep.svcTitle": ["OpenAI 相容轉錄服務", "OpenAI 兼容转录服务", "OpenAI-compatible transcription"],
    "ep.svcDesc": ["讓手機或其他程式透過區網上傳音檔辨識", "让手机或其他程序通过局域网上传音频识别", "Let phones or apps upload audio over LAN"],
    "ep.start": ["啟動服務", "启动服务", "Start service"],
    "ep.connInfo": ["連線資訊", "连接信息", "Connection"],
    "ep.cfTitle": ["對外臨時網址（Cloudflare）", "对外临时网址（Cloudflare）", "Public temporary URL (Cloudflare)"],
    "ep.cfMake": ["建立對外網址", "创建对外网址", "Create public URL"],
    "ep.cfOut": ["對外", "对外", "Public"],
    "ep.settings": ["設定", "设置", "Settings"],
    "ep.port": ["監聽埠", "监听端口", "Listen port"],
    "ep.path": ["端點路徑", "端点路径", "Endpoint path"],
    "ep.pathDesc": ["OpenAI Whisper 相容，可直接替換 base URL 使用", "OpenAI Whisper 兼容，可直接替换 base URL 使用", "OpenAI Whisper-compatible; swap your base URL directly"],

    "model.health": ["系統自檢", "系统自检", "System check"],
    "model.recheck": ["重新檢查", "重新检查", "Re-check"],
    "model.core": ["推理核心", "推理核心", "Inference core"],
    "model.model": ["模型", "模型", "Model"],
    "model.load": ["下載並載入模型", "下载并载入模型", "Download & load model"],
    "model.loading": ["下載／載入中…", "下载／载入中…", "Downloading / loading…"],
    "model.goto": ["前往語音轉文字", "前往语音转文字", "Go to transcription"],
    "model.needRestart": ["切換核心需重新啟動", "切换核心需重启", "Switching core needs restart"],
    "model.nModels": ["{n} 種模型可選", "{n} 种模型可选", "{n} models"],
    "model.devices": ["偵測到的裝置", "检测到的设备", "Detected devices"],
    "status.ready": ["模型已就緒", "模型已就绪", "Model ready"],
    "status.loading": ["載入模型中…", "载入模型中…", "Loading model…"],
    "status.needModel": ["尚未載入模型", "尚未载入模型", "No model loaded"],
    "model.diag": ["診斷", "诊断", "Diagnostics"],

    "set.scale": ["介面縮放", "界面缩放", "UI scale"],
    "set.scaleDesc": ["調整文字與控制項大小", "调整文字与控件大小", "Adjust text and control size"],
    "set.format": ["輸出格式", "输出格式", "Output format"],
    "set.formatDesc": ["全域預設，影響單檔／批次／端點", "全局默认，影响单文件／批量／端点", "Global default for single / batch / endpoint"],
    "set.fmtSrt": ["SRT 字幕", "SRT 字幕", "SRT subtitles"],
    "set.fmtTxt": ["純文字", "纯文本", "Plain text"],
    "set.vad": ["語音偵測靈敏度（VAD）", "语音检测灵敏度（VAD）", "Voice detection (VAD)"],
    "set.vadDesc": ["降低閾值可減少漏識（被當成空白的片段可能有聲音）；提高則減少假陽性。預設 0.50", "降低阈值可减少漏识（被当成空白的片段可能有声音）；提高则减少假阳性。默认 0.50", "Lower threshold reduces misses; higher reduces false positives. Default 0.50"],
    "set.vocab": ["簡繁詞彙轉換", "简繁词汇转换", "Chinese variant"],
    "set.vocabDesc": ["把模型輸出轉成在地用語", "把模型输出转成本地用语", "Localize model output wording"],
    "set.vocabOff": ["關閉", "关闭", "Off"],
    "set.vocabTW": ["台灣用語", "台湾用语", "Taiwan"],
    "set.vocabStd": ["標準", "标准", "Standard"],
    "set.mirror": ["HuggingFace 鏡像站", "HuggingFace 镜像站", "HuggingFace mirror"],
    "set.mirrorDesc": ["下載模型較慢時可改用鏡像", "下载模型较慢时可改用镜像", "Use a mirror if downloads are slow"],
    "set.ffmpeg": ["FFmpeg 路徑", "FFmpeg 路径", "FFmpeg path"],
    "set.ffmpegDesc": ["處理影片音軌所需（留空則需要時自動下載）", "处理视频音轨所需（留空则需要时自动下载）", "Needed for video audio (auto-downloads if blank)"],
    "set.theme": ["外觀主題", "外观主题", "Appearance"],
    "set.themeLight": ["淺色", "浅色", "Light"],
    "set.themeDark": ["深色", "深色", "Dark"],
    "set.themeSystem": ["跟隨系統", "跟随系统", "System"],
    "set.lang": ["介面語言", "界面语言", "Language"],
    "set.appDesc": ["本地推理 · 資料不離開你的電腦", "本地推理 · 数据不离开你的电脑", "Local inference · data stays on your PC"],
    "set.checkUpdate": ["檢查更新", "检查更新", "Check update"],
  };

  function idx() { return LOCALE === "hans" ? 1 : (LOCALE === "en" ? 2 : 0); }
  function t(key) { const e = D[key]; return e ? (e[idx()] || e[0]) : null; }

  function apply(root) {
    root = root || document;
    root.querySelectorAll("[data-i18n]").forEach(el => { const v = t(el.dataset.i18n); if (v != null) el.textContent = v; });
    root.querySelectorAll("[data-i18n-html]").forEach(el => { const v = t(el.dataset.i18nHtml); if (v != null) el.innerHTML = v; });
    root.querySelectorAll("[data-i18n-ph]").forEach(el => { const v = t(el.dataset.i18nPh); if (v != null) el.setAttribute("placeholder", v); });
    root.querySelectorAll("[data-i18n-title]").forEach(el => { const v = t(el.dataset.i18nTitle); if (v != null) el.title = v; });
  }

  function setLang(uiLang) {
    LOCALE = LANG_MAP[uiLang] || "hant";
    document.documentElement.lang = LOCALE === "en" ? "en" : (LOCALE === "hans" ? "zh-Hans" : "zh-Hant");
    apply(document);
  }

  window.I18N = { setLang, t, apply, get locale() { return LOCALE; } };
})();
