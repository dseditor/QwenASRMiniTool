"""webview_backend.py — WebView 共用後端邏輯（與視窗/傳輸無關）

把「狀態 / 設定 / 裝置 / 端點 / 轉錄」等業務邏輯集中於此，重用既有
ASREngine。不依賴 pywebview，也不依賴 HTTP —— 可被 webview_server
（桌面）直接呼叫，亦可被測試直接 import 驗證。

設計原則：
  • 不碰 GUI、不碰 socket；純函式式後端。
  • transcribe() 以 progress_cb(pct:int, status:str) 回報進度，由上層
    （HTTP 層）轉成 SSE 或其他通道推給前端。
  • 模型較重 → load() 設計成可在背景執行緒呼叫。
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path

import app as core          # 重用 ASREngine / 常數 / 診斷函式（__main__ guard 保護）

BASE_DIR = Path(__file__).resolve().parent

# ── 模型目錄：核心 → 模型 → (backend, settings patch)。──────────────────
#   webview 自有目錄（不直接用 app._resolve_backend），因為它多了一個
#   app.py 尚無的「Qwen on CRISPASR」項（同顆 crispasr.exe 跑 Qwen3-ASR GGUF，
#   見 crispasr-nemotron-eval）。settings 鍵與 app.py 相容：
#     openvino → cpu_model_size("0.6B"/"1.7B")
#     chatllm  → 無額外鍵（單一 .bin）
#     crispasr → crisp_model("breeze"/"qwen3") + crisp_quant(breeze 才用)
#   每個 entry：(核心標籤, 模型標籤, backend, settings_patch)
# chatllm（Vulkan）核心：因 AMD 內顯已知問題 + 打包時二進位常被防毒隔離，
# 核心二進位「不放進安裝包」（build_webview.bat 不再複製 chatllm/）。但引擎程式碼
# (_load_chatllm/chatllm_engine)保留，且為「向下相容」——曾使用 chatllm 的舊用戶
# （或自 CTk 桌面版升級、機器上已備有 chatllm/ 的人）仍能選用並確認其載入狀況：
#   • 模型清單：chatllm 項目只在「機器上實際有 chatllm/libchatllm.dll」或「目前
#     已記住 backend=chatllm」時才顯示（_chatllm_available）→ 新安裝的乾淨機器看不到，
#     不污染主推 CASR-Qwen 的 UI；舊用戶仍可見、可切。
#   • 系統自檢：同條件下 health_check 會列出 chatllm 核心（DLL／模型／FA）就緒狀況。
_CHATLLM_LABEL = "Qwen3-ASR-1.7B Q8（Vulkan · 相容）"
_CHATLLM_AMD_NOTE = ("⚠️ chatllm 核心在部分 AMD 內顯／APU 有已知相容問題，且核心二進位"
                     "未隨安裝包提供（保留供既有使用者向下相容）；若遇當機或無輸出，"
                     "建議改用「Qwen · CRISPASR/Vulkan」核心。")
_MODEL_CATALOG = [
    ("Qwen", "Qwen3-ASR-0.6B",              "openvino", {"cpu_model_size": "0.6B"}),
    ("Qwen", "Qwen3-ASR-1.7B INT8",         "openvino", {"cpu_model_size": "1.7B"}),
    ("Qwen", "Qwen3-ASR-1.7B Q4 (CRISPASR)", "crispasr", {"crisp_model": "qwen3", "crisp_qwen_quant": "q4"}),
    ("Qwen", "Qwen3-ASR-1.7B Q8 (CRISPASR)", "crispasr", {"crisp_model": "qwen3", "crisp_qwen_quant": "q8"}),
    # 日語動漫特化（cstr/qwen3-asr-1.7b-ja-anime）：同架構、針對日語微調，日文歌詞/台詞辨識較佳。
    ("Qwen", "Qwen3-ASR-1.7B 日語動漫 Q4 (CRISPASR)", "crispasr", {"crisp_model": "qwen3-ja", "crisp_qwen_quant": "q4"}),
    ("Qwen", "Qwen3-ASR-1.7B 日語動漫 Q8 (CRISPASR)", "crispasr", {"crisp_model": "qwen3-ja", "crisp_qwen_quant": "q8"}),
    # 台灣國語特化（JacobLinCool/TEA-ASR-1.1，Qwen3-ASR-1.7B 微調）：繁中／台灣用語／
    # 中英夾雜。HF 上現成的 mradermacher GGUF 是 llama.cpp mmproj 拆檔版、CrispASR
    # 讀不了，故本專案自行以 CrispASR 官方轉換器重轉為單檔融合式 GGUF（見 downloader）。
    ("Qwen", "TEA-ASR-1.1 台灣國語 Q8 (CRISPASR)", "crispasr", {"crisp_model": "tea", "crisp_qwen_quant": "q8"}),
    # chatllm（相容項）：永遠在目錄中（供 set_model 查得），但 get_model_options
    # 會依 _chatllm_available 決定是否實際呈現給前端。
    ("Qwen", _CHATLLM_LABEL,                "chatllm",  {}),
    ("Whisper (Breeze)", "Breeze Q4 (輕量)", "crispasr", {"crisp_model": "breeze", "crisp_quant": "q4"}),
    ("Whisper (Breeze)", "Breeze Q5 (標準)", "crispasr", {"crisp_model": "breeze", "crisp_quant": "q5"}),
    ("Whisper (Breeze)", "Breeze Q8 (精確)", "crispasr", {"crisp_model": "breeze", "crisp_quant": "q8"}),
]
_BREEZE_QUANT_LABEL = {"q4": "Breeze Q4 (輕量)", "q5": "Breeze Q5 (標準)", "q8": "Breeze Q8 (精確)"}
_QWEN_CASR_QUANT_LABEL = {"q4": "Qwen3-ASR-1.7B Q4 (CRISPASR)",
                          "q8": "Qwen3-ASR-1.7B Q8 (CRISPASR)"}
_QWEN_JA_CASR_QUANT_LABEL = {"q4": "Qwen3-ASR-1.7B 日語動漫 Q4 (CRISPASR)",
                             "q8": "Qwen3-ASR-1.7B 日語動漫 Q8 (CRISPASR)"}
_TEA_CASR_QUANT_LABEL = {"q8": "TEA-ASR-1.1 台灣國語 Q8 (CRISPASR)"}
# TEA-ASR 是「凍結音訊編碼器 ＋ 解碼器小 LoRA、不到 10 小時公開語料」的輕量微調
# （見模型卡）。實測（crispasr / q8_0 與未量化 f16 結果一致，故非量化所致）：
#   • 乾淨人聲：辨識最佳，且**原生輸出繁體**（免簡繁轉換）
#   • 配樂較重的段落／歌曲：該段可能整段無輸出（模型直接吐 EOS）
# 這是模型的域外行為，不是包裝層可修的，故如實標示讓使用者自行選材。
_TEA_DOMAIN_NOTE = ("⚠️ TEA-ASR 為「台灣國語日常人聲」特化模型：乾淨人聲最準且原生輸出繁體，"
                    "但背景配樂較重的段落或歌曲可能整段無輸出。音樂類素材請改用 "
                    "Qwen3-ASR-1.7B（通用）或 Whisper Breeze。")
# 模型標籤 → 提醒文字（目前只有 TEA 有；chatllm 的提醒仍依 backend 給）
_MODEL_NOTES = {_TEA_CASR_QUANT_LABEL["q8"]: _TEA_DOMAIN_NOTE}

# ══════════════════════════════════════════════════════════════════════
# 基礎模式：用「用途」而非「模型名」讓使用者選
# ══════════════════════════════════════════════════════════════════════
#   模型頁分兩種模式（預設基礎）：
#     基礎(basic)   → 選用途 → 依偵測到的硬體給出建議模型 → 按同一顆載入鈕
#     進階(advanced)→ 維持原本的「核心 + 模型下拉」全手動介面
#   兩者共用同一個 set_model / request_load 管道，故不會有狀態不一致。
#
# 硬體分級（tier）決定同一用途要給哪個量化：
#   gpu  = 有獨立顯卡（NVIDIA／AMD 獨顯）→ 給最準的 Q8
#   igpu = 只有內顯（Intel／AMD APU）    → 給較輕的 Q4／Q5
#   cpu  = 完全沒偵測到顯示卡            → 中文用途改推 OpenVINO 0.6B（純 CPU 主場）
_BASIC_PROFILES = [
    {
        "key":   "zh",
        "title": "中文（標準）",
        "desc":  "一般用途首選：中文為主、斷句與標點最完整，長影片／會議／訪談都適用。",
        "picks": {
            "gpu":  ("Qwen", _QWEN_CASR_QUANT_LABEL["q8"]),
            "igpu": ("Qwen", _QWEN_CASR_QUANT_LABEL["q4"]),
            "cpu":  ("Qwen", "Qwen3-ASR-0.6B"),
        },
    },
    {
        "key":   "tea",
        "title": "繁體中文 TEA（乾淨聲音）",
        "desc":  "台灣國語特化：原生輸出繁體與台灣用語，中英夾雜也保得住。"
                 "限乾淨人聲（訪談、質詢、講座）。",
        "note":  _TEA_DOMAIN_NOTE,
        "picks": {
            "gpu":  ("Qwen", _TEA_CASR_QUANT_LABEL["q8"]),
            "igpu": ("Qwen", _TEA_CASR_QUANT_LABEL["q8"]),
            "cpu":  ("Qwen", _TEA_CASR_QUANT_LABEL["q8"]),
        },
    },
    {
        "key":   "ja",
        "title": "日本語",
        "desc":  "日語／動漫特化：日文歌詞與台詞辨識明顯較佳，且輸出保留日文原生漢字"
                 "（不會被誤轉成繁體字形）。",
        "picks": {
            "gpu":  ("Qwen", _QWEN_JA_CASR_QUANT_LABEL["q8"]),
            "igpu": ("Qwen", _QWEN_JA_CASR_QUANT_LABEL["q4"]),
            "cpu":  ("Qwen", _QWEN_JA_CASR_QUANT_LABEL["q4"]),
        },
    },
    {
        "key":   "twmix",
        "title": "國台語混合（高速）",
        "desc":  "Whisper Breeze-ASR-26：繁中／台語特化且速度極快，適合大量批次。"
                 "模型不輸出標點，字幕改以長度切行（斷句較平）。",
        "picks": {
            "gpu":  ("Whisper (Breeze)", _BREEZE_QUANT_LABEL["q8"]),
            "igpu": ("Whisper (Breeze)", _BREEZE_QUANT_LABEL["q5"]),
            "cpu":  ("Whisper (Breeze)", _BREEZE_QUANT_LABEL["q5"]),
        },
    },
]
# 沒有顯示卡時，非中文用途只剩 GGUF 模型可用 → 據實提醒會比較慢
_BASIC_CPU_SLOW_NOTE = ("此用途沒有純 CPU 的專用模型，將以 CrispASR 在處理器上執行，"
                        "速度會明顯較慢；若可接受請直接下載，或改選「中文（標準）」。")
# crispasr-Qwen 系列的 crisp_model 值（皆走 --backend qwen3-1.7b、同置 ov_models/、
# 共用 crisp_qwen_quant，差別只在權重與下載來源）。"breeze" 不在此列。
_QWEN_CASR_MODELS = ("qwen3", "qwen3-ja", "tea")
# 各 crispasr-Qwen 模型在下載訊息中的中文註記
_QWEN_CASR_NOTE = {"qwen3-ja": "（日語動漫）", "tea": "（台灣國語）"}


def _qwen_casr_dl(crisp_model: str):
    """依 crisp_model 回傳對應的 (檔名, 存在檢查, 下載) 三個下載器函式。

    qwen3     → 標準 Qwen3-ASR-1.7B（cstr/qwen3-asr-1.7b-GGUF）
    qwen3-ja  → 日語動漫特化（cstr/qwen3-asr-1.7b-ja-anime-GGUF）
    tea       → 台灣國語特化（TEA-ASR-1.1，本專案自轉的融合式 GGUF）
    讓所有「crispasr-Qwen」路徑共用一處分派，避免各模型分支散落各方法。
    """
    from downloader import (qwen3_asr_gguf_filename, quick_check_qwen3_asr_gguf,
                            download_qwen3_asr_gguf, qwen3_asr_ja_gguf_filename,
                            quick_check_qwen3_asr_ja_gguf, download_qwen3_asr_ja_gguf,
                            tea_asr_gguf_filename, quick_check_tea_asr_gguf,
                            download_tea_asr_gguf)
    if crisp_model == "qwen3-ja":
        return (qwen3_asr_ja_gguf_filename, quick_check_qwen3_asr_ja_gguf,
                download_qwen3_asr_ja_gguf)
    if crisp_model == "tea":
        return (tea_asr_gguf_filename, quick_check_tea_asr_gguf,
                download_tea_asr_gguf)
    return (qwen3_asr_gguf_filename, quick_check_qwen3_asr_gguf,
            download_qwen3_asr_gguf)

# 說話者分離是「與後端無關的外部 ONNX」(diarize.py / DiarizationEngine)：
# OpenVINO 與 CRISPASR(whisper/qwen) 皆支援——前者 process_file 內建 use_diar 分支，
# 後者由 crisp_engine._apply_diarization 依時間指派。diar_engine 由 _ensure_diarization 掛上。
_DIARIZE_BACKENDS = {"openvino", "chatllm", "crispasr"}

# 辨識語言：crispasr/chatllm 共用的常用語系清單（OpenVINO 改用 processor 的）
_COMMON_LANGS = [
    "Chinese", "English", "Japanese", "Korean", "Cantonese", "French", "German",
    "Spanish", "Portuguese", "Russian", "Arabic", "Thai", "Vietnamese",
    "Indonesian", "Malay",
]


def parse_srt_to_segments(srt_path) -> list[dict]:
    """把 process_file 產出的 SRT 解析成前端要的 segments。

    說話者分離時，SRT 文字以「說話者 N：…」開頭 → 拆出 speaker 與 text。
    """
    try:
        from api_server import _parse_srt
    except Exception:
        return []
    if not srt_path or not Path(srt_path).exists():
        return []
    raw = _parse_srt(Path(srt_path).read_text(encoding="utf-8"))
    out = []
    for s in raw:
        text, speaker = s["text"], None
        if "：" in text[:8]:
            head, _, rest = text.partition("：")
            if head.startswith("說話者"):
                digits = "".join(c for c in head if c.isdigit())
                speaker = int(digits) if digits else None
                text = rest
        out.append({"start": s["start"], "end": s["end"], "speaker": speaker, "text": text})
    return out


def _rich_to_segments(rich) -> list[dict]:
    """引擎側通道 _last_segments_rich → 前端 segments（含字級 words，卡拉OK用）。

    引擎存的 speaker 是原始標籤（"說話者N" 或 None）；此處正規化成整數，
    與 parse_srt_to_segments 的前端契約一致。words 原樣帶出（[]=無對齊）。
    """
    out = []
    for seg in (rich or []):
        spk = seg.get("speaker")
        speaker = None
        if isinstance(spk, int):
            speaker = spk
        elif isinstance(spk, str):
            digits = "".join(c for c in spk if c.isdigit())
            speaker = int(digits) if digits else None
        out.append({
            "start": seg["start"], "end": seg["end"],
            "speaker": speaker, "text": seg["text"],
            "words": seg.get("words") or [],
        })
    return out


class WebBackend:
    """桌面 WebView 的後端。HTTP 層（webview_server）持有一個實例。"""

    def __init__(self, on_event=None):
        """on_event(event:str, payload:dict)：狀態/進度推送回呼（可選）。"""
        self.engine = core.ASREngine()
        self._loaded = False
        self._load_err = None
        self._loading = False
        self._cancel = False
        self._server = None              # api_server.TranscribeServer（LAN 端點）
        self._tunnel = None              # cf_tunnel.CloudflareTunnel（對外臨時網址，延遲建立）
        self._on_event = on_event
        self._theme_cb = None            # 主題變更回呼（app_webview 用來同步視窗標題列深淺）
        self._lock = threading.Lock()
        self._seed_defaults()            # 首次啟動（無 backend）→ 種子預設模型
        self._apply_runtime_prefs()      # 啟動即套用持久化偏好（VAD/簡繁/鏡像/格式）

    # ── 首次啟動預設：1.7B Qwen on CRISPASR Q4（多數機器跑得動）─────────────
    def _seed_defaults(self):
        """settings.json 尚無 backend 時，種下預設模型選擇。

        多數狀況下「Qwen3-ASR-1.7B Q4（CRISPASR/Vulkan）」可順利運行（GPU Vulkan，
        繁中佳）。寫入 settings 後，後續 _persisted_backend/_load_worker 等皆讀到它。
        既有使用者（已有 backend）不受影響。
        """
        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        try:
            cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            cur = {}
        if cur.get("backend"):
            return                       # 既有設定 → 不覆寫
        cur.setdefault("backend", "crispasr")
        cur.setdefault("crisp_model", "qwen3")
        cur.setdefault("crisp_qwen_quant", "q4")
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── 事件推送 ────────────────────────────────────────────
    def _emit(self, event: str, payload: dict):
        if self._on_event:
            try:
                self._on_event(event, payload)
            except Exception:
                pass

    # ── 模型載入（背景執行緒呼叫）──────────────────────────
    def start_load(self):
        if self._loading or self._loaded:
            return
        self._loading = True
        threading.Thread(target=self._load_worker, name="model-loader", daemon=True).start()

    def _load_worker(self):
        # 依 settings.backend 全新載入對應引擎。因「切換=重啟」，每次啟動只載入
        # 一個核心、且為全新行程，天然避開 chatllm Vulkan 雙 context 切換當機。
        backend = self._persisted_backend()           # openvino / chatllm / crispasr
        try:
            if backend == "chatllm":
                self._load_chatllm()
            elif backend == "crispasr":
                self._load_crispasr()
            else:
                backend = "openvino"
                self._load_openvino()
            self._loaded = True
            self._active_backend = backend
            self._remember_active(backend)
            self._emit("status", {"modelReady": True})
        except Exception as e:
            traceback.print_exc()
            head = str(e).splitlines()[0][:110] if str(e) else type(e).__name__
            # GPU 核心載入失敗 → 退回 CPU(OpenVINO)，但明確告知(非靜默回退)
            if backend != "openvino":
                self._emit("progress", {"pct": 0, "status": f"{backend} 載入失敗，改用 CPU 核心…"})
                try:
                    self._load_openvino()
                    self._loaded = True
                    self._active_backend = "openvino"
                    self._remember_active("openvino")
                    self._load_err = f"{backend} 核心載入失敗，已退回 CPU(OpenVINO)：{head}"
                    self._emit("status", {"modelReady": True, "error": self._load_err})
                    return
                except Exception:
                    traceback.print_exc()
            self._load_err = str(e)
            self._emit("status", {"modelReady": False, "error": str(e)})
        finally:
            self._loading = False

    # ── 各核心載入（重用既有引擎類別與 downloader）─────────────
    def _settings_raw(self) -> dict:
        try:
            f = getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json")
            return json.loads(Path(f).read_text(encoding="utf-8")) if Path(f).exists() else {}
        except Exception:
            return {}

    def _vk_device_id(self, s: dict) -> int:
        import re
        m = re.search(r"GPU:(\d+)", s.get("device", "") or "")
        return int(m.group(1)) if m else 0

    # 每段最長秒數：使用者設定 → 套到當前引擎，但永遠夾到「模型天花板」。
    # 天花板＝引擎類別的 max_chunk_secs（0.6B/chatllm 30、1.7B 10），由音訊
    # 編碼器匯出長度寫死，超過會被靜默截斷掉字，故只允許往短調，絕不超過。
    # crispasr（無 max_chunk_secs 類別屬性）走自身視窗，不受此設定影響。
    _CHUNK_FLOOR = 5
    def _apply_chunk_secs(self) -> None:
        eng = getattr(self, "engine", None)
        ceiling = getattr(type(eng), "max_chunk_secs", None) if eng is not None else None
        if not isinstance(ceiling, int):
            return                                  # crispasr 等：不以 max_chunk_secs 分段
        try:
            want = int(self._settings_raw().get("chunk_secs", 0) or 0)
        except Exception:
            want = 0
        eff = ceiling if want <= 0 else max(self._CHUNK_FLOOR, min(want, ceiling))
        try:
            eng.max_chunk_secs = eff                # 實例屬性遮蓋類別預設，僅影響本實例
        except Exception:
            pass

    def _dl_progress(self, *a):
        """下載進度回呼（容忍 progress_cb(frac) 或 progress_cb(frac, msg)）。"""
        frac = a[0] if a else 0
        msg = a[1] if len(a) > 1 else "下載中…"
        try:
            self._emit("progress", {"pct": int(float(frac) * 100), "status": str(msg)})
        except Exception:
            pass

    def _st(self, m):
        self._emit("progress", {"pct": 0, "status": m})

    def _load_openvino(self):
        s = self._settings_raw()
        model_dir = Path(s.get("model_dir", str(core._DEFAULT_MODEL_DIR)))
        use_17b = "1.7B" in s.get("cpu_model_size", "0.6B")
        cpu_threads = int(s.get("cpu_threads", 0) or 0)
        if use_17b:
            from downloader import quick_check_1p7b, download_1p7b
            if not quick_check_1p7b(model_dir):
                self._st("下載 1.7B 模型（約 4.3 GB）…")
                download_1p7b(model_dir, progress_cb=self._dl_progress)
        eng = core.ASREngine1p7B() if use_17b else core.ASREngine()
        eng.load(device="CPU", model_dir=model_dir, cb=self._st, cpu_threads=cpu_threads)
        self.engine = eng

    def _load_chatllm(self):
        from chatllm_engine import ChatLLMASREngine
        s = self._settings_raw()
        chatllm_dir = Path(s.get("chatllm_dir", str(getattr(core, "_CHATLLM_DIR", BASE_DIR / "chatllm"))))
        default_bin = getattr(core, "_BIN_PATH", BASE_DIR / "ov_models" / "qwen3-asr-1.7b.bin")
        model_path = Path(s.get("model_path") or s.get("gguf_path") or str(default_bin))
        if not model_path.exists():
            for c in (Path(s.get("model_dir", "")) / "qwen3-asr-1.7b.bin" if s.get("model_dir") else None,
                      default_bin):
                if c and Path(c).exists():
                    model_path = Path(c)
                    break
        if not model_path.exists():
            self._download_chatllm_bin(model_path)
        eng = ChatLLMASREngine()
        eng.load(model_path=model_path, chatllm_dir=chatllm_dir, n_gpu_layers=99,
                 device_id=self._vk_device_id(s), cb=self._st)
        self.engine = eng

    def _download_chatllm_bin(self, model_path: Path):
        import urllib.request
        from downloader import _ssl_ctx
        self._st("下載 chatllm 模型（~2.3 GB）…")
        url = "https://huggingface.co/dseditor/Collection/resolve/main/qwen3-asr-1.7b.bin"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; QwenASR)"})
        with urllib.request.urlopen(req, context=_ssl_ctx()) as resp, \
                open(str(model_path) + ".tmp", "wb") as out:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            while True:
                block = resp.read(65536)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total > 0:
                    self._dl_progress(done / total, f"模型 {done/1048576:.0f}/{total/1048576:.0f} MB")
        import os as _os
        _os.replace(str(model_path) + ".tmp", str(model_path))

    def _load_crispasr(self):
        from crisp_engine import CrispWhisperEngine
        from downloader import (quick_check_crispasr, download_crispasr_core,
                                quick_check_breeze, download_breeze, breeze_filename,
                                quick_check_aligner_gguf, download_aligner_gguf,
                                aligner_gguf_filename, crispasr_gpu_backend)
        s = self._settings_raw()
        crispasr_dir = self._crispasr_dir()
        crisp_model = s.get("crisp_model", "breeze")     # breeze / qwen3 / qwen3-ja / tea
        quant = s.get("crisp_quant", "q5")
        fa_enabled = bool(s.get("crisp_fa", True))
        fa_quant = s.get("crisp_fa_quant", "q5")
        variant = self._crisp_variant()                  # 加速版本（vulkan/cuda…）

        # crispasr.exe 核心一律需要。帶 variant 檢查 → 版本升級或使用者改選
        # 加速版本時都會判定為「不就緒」而重新下載對應的 zip。
        if not quick_check_crispasr(crispasr_dir, variant):
            self._st("下載 CrispASR 核心…")
            download_crispasr_core(crispasr_dir, progress_cb=self._dl_progress,
                                   variant=variant)

        if crisp_model in _QWEN_CASR_MODELS:
            # Qwen3-ASR GGUF（與 OV 模型同置 ov_models/；crisp_engine 依檔名含 "1.7b"
            # 自動推斷 --backend qwen3-1.7b）。標準版與日語動漫版共用此路徑，僅來源/
            # 檔名不同（由 _qwen_casr_dl 分派）。Q4/Q8 兩量化，缺檔自動下載（cstr repo）。
            fname_fn, check_fn, download_fn = _qwen_casr_dl(crisp_model)
            qquant = s.get("crisp_qwen_quant", "q8")
            model_dir = Path(s.get("model_dir", str(getattr(core, "_DEFAULT_MODEL_DIR",
                                                            BASE_DIR / "ov_models"))))
            model_path = model_dir / fname_fn(qquant)
            if not check_fn(model_dir, qquant):
                _note = _QWEN_CASR_NOTE.get(crisp_model, "")
                _name = "TEA-ASR-1.1" if crisp_model == "tea" else "Qwen3-ASR-1.7B"
                self._st(f"下載 {_name}{_note} {qquant.upper()} GGUF…")
                download_fn(model_dir, qquant, progress_cb=self._dl_progress)
        else:
            model_path = crispasr_dir / breeze_filename(quant)
            if not quick_check_breeze(crispasr_dir, quant):
                self._st(f"下載 Breeze-ASR-26 {quant.upper()} 模型…")
                download_breeze(crispasr_dir, quant, progress_cb=self._dl_progress)
        aligner_path = None
        if fa_enabled:
            if quick_check_aligner_gguf(crispasr_dir, fa_quant):
                aligner_path = crispasr_dir / aligner_gguf_filename(fa_quant)
            else:
                try:
                    self._st("下載時間軸對齊器…")
                    download_aligner_gguf(crispasr_dir, fa_quant, progress_cb=self._dl_progress)
                    aligner_path = crispasr_dir / aligner_gguf_filename(fa_quant)
                except Exception:
                    aligner_path = None       # 退回 Whisper 自帶時間軸
        eng = CrispWhisperEngine()
        eng.load(model_path=model_path, crispasr_dir=crispasr_dir,
                 device_id=self._vk_device_id(s), cb=self._st, aligner_path=aligner_path,
                 gpu_backend=crispasr_gpu_backend(variant))
        self.engine = eng

    # ── 狀態 ────────────────────────────────────────────────
    def get_status(self) -> dict:
        active = getattr(self, "_active_backend", "openvino")
        return {
            "modelReady": bool(getattr(self.engine, "ready", False)),
            "loading": self._loading,
            "error": self._load_err,
            "backend": self._backend_label(active, "CPU · OpenVINO INT8"),
            "backendKey": self._persisted_backend(),       # 已記住的核心選擇
            "activeBackend": active,                        # 實際載入中的核心
            "device": "GPU" if active in ("chatllm", "crispasr") else "CPU",
            "version": self._app_version(),
            "appName": "聲音辨識小工具",
            # 是否已有「任一組」模型下載完成（資訊用）
            "hasAnyModel": self._has_any_model(),
            # 目前選擇的模型是否已就緒 → 前端決定開始頁（就緒=語音轉文字／否=模型頁等下載）
            "selectedReady": self.selected_model_present(),
        }

    def selected_model_present(self) -> bool:
        """目前持久化選擇的『那一個』模型檔是否已下載。

        決定兩件事：① 啟動時是否自動載入（present→直接載；缺→不自動下載，等
        使用者在模型頁選好、按「下載並載入」才觸發，讓老手能先改選 Whisper 等）；
        ② 前端開始頁（present→語音轉文字；缺→模型頁）。
        """
        try:
            from downloader import (quick_check, quick_check_1p7b, quick_check_breeze,
                                    quick_check_crispasr)
            s = self._settings_raw()
            be = s.get("backend", "openvino")
            model_dir = self._model_dir()
            if be == "openvino":
                return (quick_check_1p7b(model_dir)
                        if "1.7B" in s.get("cpu_model_size", "0.6B") else quick_check(model_dir))
            if be == "crispasr":
                cd = self._crispasr_dir()
                if not quick_check_crispasr(cd):
                    return False                  # 核心 exe 缺（理論上隨包附帶）→ 當作未就緒
                cm = s.get("crisp_model", "breeze")
                if cm in _QWEN_CASR_MODELS:
                    _, check_fn, _ = _qwen_casr_dl(cm)
                    return check_fn(model_dir, s.get("crisp_qwen_quant", "q8"))
                return quick_check_breeze(cd, s.get("crisp_quant", "q5"))
            if be == "chatllm":
                return ((self._chatllm_dir() / "qwen3-asr-1.7b.bin").exists()
                        or (model_dir / "qwen3-asr-1.7b.bin").exists())
        except Exception:
            traceback.print_exc()
        return False

    def _has_any_model(self) -> bool:
        """機器上是否已有任一可用模型（任一核心任一量化）。供首啟頁面決策。"""
        try:
            from downloader import (quick_check, quick_check_1p7b, quick_check_breeze,
                                    quick_check_qwen3_asr_gguf, quick_check_qwen3_asr_ja_gguf,
                                    quick_check_tea_asr_gguf)
            model_dir = self._model_dir()
            crispasr_dir = self._crispasr_dir()
            if quick_check(model_dir) or quick_check_1p7b(model_dir):
                return True
            for q in ("q4", "q5", "q8"):
                if quick_check_breeze(crispasr_dir, q):
                    return True
            for q in ("q4", "q8"):
                if quick_check_qwen3_asr_gguf(model_dir, q) or quick_check_qwen3_asr_ja_gguf(model_dir, q):
                    return True
            if quick_check_tea_asr_gguf(model_dir, "q8"):
                return True
            # chatllm .bin（向下相容）
            if (self._chatllm_dir() / "qwen3-asr-1.7b.bin").exists():
                return True
            if (model_dir / "qwen3-asr-1.7b.bin").exists():
                return True
        except Exception:
            traceback.print_exc()
        return False

    def _persisted_backend(self) -> str:
        try:
            f = getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json")
            if Path(f).exists():
                return json.loads(Path(f).read_text(encoding="utf-8")).get("backend", "openvino")
        except Exception:
            pass
        return "openvino"

    def _app_version(self) -> str:
        try:
            import version
            # WebView 版優先用獨立版本字串（webview 0.1）；缺則退回語意化版本。
            return getattr(version, "WEBVIEW_VERSION", None) or version.__version__
        except Exception:
            return ""

    # ── 轉錄 ────────────────────────────────────────────────
    # opts: {path, language, diarize, nSpeakers, align, hint}
    # progress_cb(pct:int, status:str)
    def transcribe(self, opts: dict, progress_cb=None) -> dict:
        if not getattr(self.engine, "ready", False):
            raise RuntimeError("模型尚未載入完成，請稍候再試。")
        path = opts.get("path")
        if not path or not Path(path).exists():
            raise RuntimeError("找不到音訊檔。")
        self._cancel = False

        def _cb(i, total, msg):
            if progress_cb:
                pct = min(99, int((i / max(total, 1)) * 100))
                progress_cb(pct, msg)

        # 影片 → 先抽音軌（與 api_server 一致）
        audio_path = Path(path)
        tmp_extra = None
        try:
            ext = audio_path.suffix.lower()
            from api_server import _VIDEO_HINT
            if ext in _VIDEO_HINT:
                from ffmpeg_utils import extract_audio_to_wav
                ff = self._ensure_ffmpeg()
                if not ff:
                    raise RuntimeError("上傳為影片但找不到 ffmpeg，且自動下載失敗，無法抽音軌。")
                wav = audio_path.with_suffix(".extracted.wav")
                extract_audio_to_wav(audio_path, wav, ff)
                audio_path, tmp_extra = wav, wav

            want_align = bool(opts.get("align", True))
            want_diar = bool(opts.get("diarize"))

            # ── 時間軸對齊（FA）：缺模型則按需下載（移植自 app.py 的協調層）──
            if want_align:
                self._ensure_fa()
            if hasattr(self.engine, "use_aligner") and getattr(self.engine, "_fa_bin", None):
                self.engine.use_aligner = want_align

            # ── 說話者分離：外部 ONNX，與後端無關。缺模型則下載 + 掛上 diar_engine ──
            if want_diar:
                if not (getattr(self.engine, "diar_engine", None)
                        and getattr(self.engine.diar_engine, "ready", False)):
                    try:
                        self._ensure_diarization()
                    except Exception as e:
                        head = str(e).splitlines()[0][:100] if str(e) else type(e).__name__
                        self._emit("progress", {"pct": 0, "status": f"說話者分離模型下載失敗：{head}"})

            n_spk = opts.get("nSpeakers")
            n_speakers = int(n_spk) if str(n_spk).isdigit() else None

            # 語言：空字串 / 自動偵測 / auto → None（讓引擎自動偵測）
            _lang = (opts.get("language") or "").strip()
            if _lang in ("", "自動偵測", "auto"):
                _lang = None

            # 輸出落點：subtitles/ 下、用「原始上傳檔名」命名（可在「開啟輸出資料夾」找到）。
            # 上傳檔名為使用者可控（含 LAN/外網端點）→ 必須只取基名，杜絕路徑穿越。
            srt_dir = Path(getattr(core, "SRT_DIR", BASE_DIR / "subtitles"))
            srt_dir.mkdir(parents=True, exist_ok=True)
            raw_name = opts.get("name") or Path(path).name
            out_name = Path(str(raw_name).replace("\\", "/")).name   # 去除任何目錄成分
            if not out_name or out_name.startswith("."):
                out_name = "transcript"
            out_ref = srt_dir / out_name
            # 防呆：確認解析後仍在 subtitles/ 內
            try:
                if srt_dir.resolve() not in out_ref.resolve().parents:
                    out_ref = srt_dir / "transcript"
            except Exception:
                out_ref = srt_dir / "transcript"

            with self._lock:
                self._apply_chunk_secs()      # 依設定夾定本次 max_chunk_secs（夾到模型上限）
                # 清除上一輪字級殘留，避免本輪若提前 return 時讀到舊資料
                try:
                    self.engine._last_segments_rich = None
                except Exception:
                    pass
                srt = self.engine.process_file(
                    audio_path,
                    progress_cb=_cb,
                    language=_lang,
                    context=(opts.get("hint") or "").strip() or None,
                    diarize=want_diar,
                    n_speakers=n_speakers,
                    original_path=out_ref,
                    out_format="srt",
                )
        finally:
            if tmp_extra:
                try:
                    Path(tmp_extra).unlink(missing_ok=True)
                except Exception:
                    pass

        if not srt:
            diag = getattr(self.engine, "_last_vad_diag", None)
            raise RuntimeError(diag or "未產生字幕（未偵測到人聲）。")

        # UI 永遠以記憶體中的 segments 渲染波形/字幕卡（需時間軸），故引擎固定產 SRT。
        # 但「輸出格式」設定決定使用者實際取得的存檔：選純文字 → 另寫 .txt、移除 .srt。
        # 優先用引擎的字級結構（含 words，驅動卡拉OK）；無則退回解析 SRT（行級）。
        rich = getattr(self.engine, "_last_segments_rich", None)
        segments = _rich_to_segments(rich) if rich else parse_srt_to_segments(srt)
        saved = str(srt)
        out_fmt = (self._settings_raw().get("output_format", "srt") or "srt").lower()
        if out_fmt == "txt":
            try:
                from subtitle_lines import write_transcript
                lines = [(seg["start"], seg["end"], seg["text"],
                          (f'說話者{seg["speaker"]}' if seg.get("speaker") else None))
                         for seg in segments]
                txt = write_transcript(Path(srt), lines, out_format="txt")
                try:
                    Path(srt).unlink(missing_ok=True)
                except Exception:
                    pass
                saved = str(txt)
            except Exception:
                traceback.print_exc()          # 退回 SRT，不讓轉錄整體失敗

        if progress_cb:
            progress_cb(100, "完成")
        return {"segments": segments, "srtPath": saved}

    def _resolve_ffmpeg(self):
        """解析 ffmpeg：優先採用設定中手動指定的 ffmpeg_path，否則自動偵測。

        與 setting.py 的偵測順序一致（使用者手動指定 → 系統 PATH / App 目錄）。
        """
        from ffmpeg_utils import find_ffmpeg
        p = (self._settings_raw().get("ffmpeg_path", "") or "").strip()
        if p and Path(p).exists():
            return Path(p)
        return find_ffmpeg()

    def _ensure_ffmpeg(self):
        """回傳可用的 ffmpeg；找不到時自 HF 下載精簡 zip 解壓後再找。

        ffmpeg 不隨安裝包附帶 → 首次需要（轉錄影片）時自動下載到 <app>/ffmpeg/。
        下載進度經 self._dl_progress → SSE 推給前端。失敗回 None（由上層報錯）。
        """
        ff = self._resolve_ffmpeg()
        if ff:
            return ff
        try:
            from ffmpeg_utils import get_default_ffmpeg_dest
            from downloader import quick_check_ffmpeg, download_ffmpeg
            dest = get_default_ffmpeg_dest()
            if not quick_check_ffmpeg(dest):
                self._st("下載 ffmpeg（影片音軌提取）…")
                download_ffmpeg(dest, progress_cb=self._dl_progress)
            return self._resolve_ffmpeg()
        except Exception:
            traceback.print_exc()
            return None

    def _model_dir(self) -> Path:
        s = self._settings_raw()
        return Path(s.get("model_dir", str(getattr(core, "_DEFAULT_MODEL_DIR",
                                                   BASE_DIR / "ov_models"))))

    # ── 按需下載：說話者分離 / FA（移植自 app.py 的 CTk 下載協調層）──────────
    def _ensure_diarization(self) -> bool:
        """確保說話者分離 ONNX 模型存在並把 diar_engine 掛到當前引擎。

        diar 是與後端無關的外部 ONNX，OpenVINO/CRISPASR 共用。缺模型則自動
        下載（約 32 MB），完成後建立 DiarizationEngine 掛上 self.engine.diar_engine。
        回傳是否就緒。
        """
        from downloader import quick_check_diarization, download_diarization
        from diarize import DiarizationEngine
        model_dir = self._model_dir()
        diar_dir = model_dir / "diarization"
        if not quick_check_diarization(model_dir):
            self._st("下載說話者分離模型（約 32 MB）…")
            download_diarization(diar_dir, progress_cb=self._dl_progress)
        eng = DiarizationEngine(diar_dir)
        if getattr(eng, "ready", False):
            try:
                self.engine.diar_engine = eng
            except Exception:
                pass
            return True
        return False

    def _ensure_fa(self) -> bool:
        """確保「時間軸對齊」模型存在（移植自 app.py _check_aligner_model）。

        CRISPASR：FA aligner gguf 已於 _load_crispasr 處理，這裡略過。
        OpenVINO/chatllm：需 chatllm ForcedAligner .bin（約 939 MB），缺則下載並
        重新載入引擎內的對齊器（eng._load_aligner）。回傳是否就緒。
        """
        backend = getattr(self, "_active_backend", "openvino")
        if backend == "crispasr":
            return bool(getattr(self.engine, "_fa_bin", None))
        try:
            from downloader import quick_check_aligner, download_aligner
            model_dir = self._model_dir()
            if not quick_check_aligner(model_dir):
                self._st("下載時間軸對齊模型（約 939 MB）…")
                download_aligner(model_dir, progress_cb=self._dl_progress)
            if hasattr(self.engine, "_load_aligner"):
                self.engine._load_aligner(cb=self._st)
            return bool(getattr(self.engine, "_fa_bin", None)
                        or getattr(self.engine, "use_aligner", False))
        except Exception:
            traceback.print_exc()
            return False

    def cancel(self):
        self._cancel = True
        return True

    def open_output_dir(self) -> bool:
        import os
        try:
            d = getattr(core, "SRT_DIR", BASE_DIR / "subtitles")
            Path(d).mkdir(parents=True, exist_ok=True)
            os.startfile(str(d))
            return True
        except Exception:
            return False

    def open_releases(self) -> dict:
        """「檢查更新」→ 以系統瀏覽器開啟 GitHub Releases 頁。"""
        import webbrowser
        try:
            import version
            url = getattr(version, "GITHUB_RELEASES_PAGE",
                          "https://github.com/dseditor/QwenASRMiniTool/releases/latest")
        except Exception:
            url = "https://github.com/dseditor/QwenASRMiniTool/releases/latest"
        try:
            webbrowser.open(url)
            return {"ok": True, "url": url}
        except Exception as e:
            return {"ok": False, "url": url, "error": str(e)}

    # ── 裝置 ────────────────────────────────────────────────
    #   偵測以 **CrispASR 為主**（crispasr.exe --diagnostics 自帶 GPU 列舉，不需
    #   chatllm）。chatllm 的 main.exe --show_devices 未隨安裝包提供 → 改為後備：
    #   僅在 crispasr 不在、但機器上有 chatllm main.exe 時才用（向下相容）。
    def _crispasr_dir(self) -> Path:
        # 用 core.BASE_DIR（frozen 時 = sys.executable 旁；非 webview_backend 的
        # __file__，後者在 onefile 凍結後指向 _MEIPASS 暫存夾）→ 才找得到隨包
        # 附帶／執行時下載到 exe 旁的 crispasr/ 核心。
        s = self._settings_raw()
        app_dir = getattr(core, "BASE_DIR", BASE_DIR)
        return Path(s.get("crispasr_dir", str(app_dir / "crispasr")))

    # ── 推理加速版本（CrispASR 的 Vulkan / CUDA / CPU 三種 build）────────
    #   偵測→推薦→下載對應 zip。注意這裡的硬體偵測**不能**用既有的
    #   probe_crispasr_devices()（那要先有 crispasr.exe），而走 downloader 的
    #   detect_hardware()（WMI + nvidia-smi），才能在下載核心前就決定抓哪一包。
    def _crisp_variant(self) -> str:
        """目前選定的加速版本；settings 沒有就依硬體推薦一個並記住。"""
        from downloader import (crispasr_variants, recommend_crispasr_variant)
        s = self._settings_raw()
        v = s.get("crisp_variant")
        if v in crispasr_variants():
            return v
        try:
            v = recommend_crispasr_variant()["recommended"]
        except Exception:
            v = "vulkan"
        self._persist_setting("crisp_variant", v)
        return v

    def get_accel(self) -> dict:
        """回傳硬體偵測結果 + 加速版本選單（供前端「推理加速版本」區塊渲染）。"""
        from downloader import (recommend_crispasr_variant, installed_crispasr_info,
                                _CRISPASR_VERSION)
        try:
            info = recommend_crispasr_variant()
        except Exception as e:
            traceback.print_exc()
            return {"ok": False, "error": f"硬體偵測失敗：{e}", "options": []}
        installed = installed_crispasr_info(self._crispasr_dir())
        selected  = self._crisp_variant()
        return {
            "ok": True,
            "selected": selected,
            "recommended": info["recommended"],
            "reason": info["reason"],
            "options": info["options"],
            "hardware": info["hardware"],
            "installed": installed,          # {"version","variant"}；None＝尚未安裝
            "latest": _CRISPASR_VERSION,
            # 已裝但版本／版本別不符 → 前端提示「載入時會自動重新下載」
            "needsDownload": (installed.get("version") != _CRISPASR_VERSION
                              or installed.get("variant") != selected),
        }

    def set_accel(self, variant: str) -> dict:
        """記住使用者選的加速版本。實際下載發生在下次載入模型時。"""
        from downloader import crispasr_variants, installed_crispasr_info, _CRISPASR_VERSION
        variants = crispasr_variants()
        if variant not in variants:
            return {"ok": False, "message": f"未知的加速版本：{variant}"}
        self._persist_setting("crisp_variant", variant)
        meta = variants[variant]
        installed = installed_crispasr_info(self._crispasr_dir())
        fresh = (installed.get("version") == _CRISPASR_VERSION
                 and installed.get("variant") == variant)
        if fresh:
            return {"ok": True, "variant": variant, "restartRequired": False,
                    "message": f"已選用「{meta['label']}」（核心已就緒）。"}
        return {"ok": True, "variant": variant, "restartRequired": True,
                "message": (f"已記住「{meta['label']}」。下次載入模型時會下載該版本"
                            f"核心（約 {meta['size_mb']} MB），請重新啟動程式以套用。")}

    def list_devices(self) -> dict:
        import platform
        devices = [{"kind": "cpu", "name": platform.processor() or "CPU", "note": "使用中"}]
        diag = {"level": None, "text": ""}

        vk, source = self._probe_gpu_devices()
        try:
            for d in (vk.get("devices") if vk else []) or []:
                gb = d.get("vram_free", 0) / (1024 ** 3)
                devices.append({"kind": "gpu", "name": d.get("name", "GPU"),
                                "note": f"{gb:.1f} GB 可用" if gb else ""})
            if source is None:
                # 沒有任何可用的偵測器（crispasr 與 chatllm 皆未就位）
                diag = {"level": "info",
                        "text": "GPU 偵測需要 CrispASR 核心；啟用 GPU 核心時會自動下載，"
                                "之後即可在此列出可用的獨立顯示卡。目前僅 CPU 推理可用。"}
            elif vk and vk.get("error"):
                diag = {"level": "warn",
                        "text": f"GPU 偵測未完成（{source}）：{vk['error']}　已自動改用 CPU 推理。"}
            elif not (vk and vk.get("devices")):
                diag = {"level": "info", "text": "未偵測到可用的獨立 GPU，僅 CPU 推理可用。"}
        except Exception as e:
            diag = {"level": "warn", "text": f"GPU 偵測例外：{e}"}
        return {"devices": devices, "diag": diag}

    def _probe_gpu_devices(self):
        """回傳 (探測結果 dict, 來源標籤)；找不到任何偵測器時回 (None, None)。

        優先 CrispASR（crispasr.exe --diagnostics）；其次後備 chatllm（main.exe
        --show_devices，僅供仍備有 chatllm 的既有使用者）。
        """
        # ① CrispASR 為主
        try:
            from crisp_engine import probe_crispasr_devices, _find_exe
            cd = self._crispasr_dir()
            if _find_exe(cd):                       # 直接或子資料夾找到 crispasr.exe
                return probe_crispasr_devices(cd), "CrispASR"
        except Exception:
            traceback.print_exc()
        # ② chatllm 後備（向下相容）
        try:
            chatllm_dir = self._chatllm_dir()
            if (chatllm_dir / "main.exe").exists():
                return core.probe_vulkan_devices(str(chatllm_dir)), "chatllm"
        except Exception:
            traceback.print_exc()
        return None, None

    # ── 核心切換：持久化選擇 + 請使用者重啟（不就地熱重載）─────
    #   理由：① chatllm DLL 在核心切換時 Vulkan context 未釋放會整機當機
    #   (見記憶 vulkan-dual-context-crash)；② 統一所有核心的切換行為，最安全。
    #   chatllm 保留原桌面實作以向下相容；未來全面改 casr，但切換一律需重啟。
    _BACKENDS = {0: "openvino", 1: "chatllm", 2: "crispasr"}
    _BACKEND_LABELS = {
        "openvino": "CPU · OpenVINO INT8",
        "chatllm":  "GPU · chatllm Vulkan",
        # CRISPASR 的加速版本可切（Vulkan/CUDA/CPU）→ 實際標籤由 _backend_label()
        # 依 crisp_variant 補上；這裡只留字典的預設值供舊呼叫端使用。
        "crispasr": "GPU · CRISPASR",
    }

    @staticmethod
    def _note_for(backend: str, model_label: str = "") -> str:
        """該（核心, 模型）要顯示的提醒文字；沒有則空字串。

        chatllm 依 backend 給（AMD 內顯問題），TEA 等依模型標籤給（模型域限制）。
        """
        if backend == "chatllm":
            return _CHATLLM_AMD_NOTE
        return _MODEL_NOTES.get(model_label, "")

    def _backend_label(self, backend: str, fallback: str = "") -> str:
        """核心顯示名。CRISPASR 會補上目前選定的加速版本（Vulkan／CUDA／CPU）。"""
        label = self._BACKEND_LABELS.get(backend)
        if label is None:
            return fallback or backend
        if backend == "crispasr":
            try:
                from downloader import crispasr_variants
                v = self._crisp_variant()
                meta = crispasr_variants().get(v)
                if meta:
                    kind = "CPU" if meta["gpu_backend"] is None else "GPU"
                    return f"{kind} · CRISPASR（{meta.get('short') or meta['label']}）"
            except Exception:
                pass
        return label

    def set_backend(self, idx) -> dict:
        backend = self._BACKENDS.get(int(idx) if str(idx).isdigit() else 0, "openvino")
        label = self._backend_label(backend)
        self._persist_backend(backend)
        active = getattr(self, "_active_backend", "openvino")
        if backend == active:
            return {"ok": True, "backend": backend, "restartRequired": False,
                    "message": f"「{label}」已是目前使用的核心。"}
        if backend == "openvino":
            return {"ok": True, "backend": backend, "restartRequired": True,
                    "message": f"已記住「{label}」。請重新啟動程式以套用新核心。"}
        # GPU 核心：重啟後會全新載入（首次啟用會自動下載對應模型）
        return {"ok": True, "backend": backend, "restartRequired": True,
                "message": (f"已記住「{label}」。請重新啟動程式以套用 —— "
                            f"首次啟用該核心會在啟動時自動下載對應模型。")}

    def _persist_backend(self, backend: str):
        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        try:
            cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            cur = {}
        cur["backend"] = backend
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    # ── 模型下拉（核心 + 模型）＋ 辨識語言 ─────────────────────────────
    #   持久化後走「切換=重啟」(同 set_backend 的理由：避免就地切核心當機)。
    def _remember_active(self, backend: str):
        """記住「實際載入」的選擇識別碼，供 set_model 判斷是否需重啟。"""
        self._active_backend = backend
        self._active_identity = self._identity_for(backend, self._settings_raw())

    def _identity_for(self, backend: str, s: dict):
        """把 (backend + 相關 settings) 化為可比較的識別碼（決定要不要重啟）。"""
        if backend == "openvino":
            return ("openvino", s.get("cpu_model_size", "0.6B"))
        if backend == "crispasr":
            cm = s.get("crisp_model", "breeze")
            quant = (s.get("crisp_quant", "q5") if cm == "breeze"
                     else s.get("crisp_qwen_quant", "q8"))   # qwen3 量化也納入識別
            # 加速版本也納入識別：換 Vulkan↔CUDA 等同換核心，需重啟才生效
            return ("crispasr", cm, quant, self._crisp_variant())
        return ("chatllm",)

    def _current_selection(self):
        """settings → (核心標籤, 模型標籤)，供下拉預選。"""
        s = self._settings_raw()
        be = s.get("backend", "openvino")
        if be == "crispasr":
            cm = s.get("crisp_model", "breeze")
            if cm == "tea":
                return ("Qwen", _TEA_CASR_QUANT_LABEL["q8"])
            if cm == "qwen3-ja":
                qq = s.get("crisp_qwen_quant", "q8")
                return ("Qwen", _QWEN_JA_CASR_QUANT_LABEL.get(qq, _QWEN_JA_CASR_QUANT_LABEL["q8"]))
            if cm == "qwen3":
                qq = s.get("crisp_qwen_quant", "q8")
                return ("Qwen", _QWEN_CASR_QUANT_LABEL.get(qq, _QWEN_CASR_QUANT_LABEL["q8"]))
            q = s.get("crisp_quant", "q5")
            return ("Whisper (Breeze)", _BREEZE_QUANT_LABEL.get(q, _BREEZE_QUANT_LABEL["q5"]))
        if be == "chatllm":
            return ("Qwen", _CHATLLM_LABEL)
        sz = s.get("cpu_model_size", "0.6B")
        return ("Qwen", "Qwen3-ASR-1.7B INT8" if "1.7B" in sz else "Qwen3-ASR-0.6B")

    def _chatllm_dir(self) -> Path:
        s = self._settings_raw()
        return Path(s.get("chatllm_dir",
                          str(getattr(core, "_CHATLLM_DIR", BASE_DIR / "chatllm"))))

    def _chatllm_available(self) -> bool:
        """chatllm 是否該對使用者「現身」（向下相容判準）。

        條件（任一成立）：① 機器上實際備有 chatllm 核心二進位（libchatllm.dll）；
        ② 目前已記住的 backend 就是 chatllm（曾選用過）。新安裝的乾淨機器兩者皆否
        → chatllm 不出現在模型清單／自檢，UI 維持以 CASR-Qwen 為主。
        """
        try:
            if self._persisted_backend() == "chatllm":
                return True
            if (self._chatllm_dir() / "libchatllm.dll").exists():
                return True
        except Exception:
            pass
        return False

    # ── 基礎模式：用途 → 依硬體建議模型 ────────────────────────────────
    def _hw_tier(self) -> tuple[str, str]:
        """回傳 (tier, 硬體摘要文字)。tier ∈ gpu / igpu / cpu。

        tier 決定基礎模式要推薦哪個量化：有獨顯給最準的 Q8，只有內顯給較輕的
        Q4／Q5，完全沒顯示卡則中文用途改推純 CPU 的 OpenVINO 0.6B。
        """
        from downloader import detect_hardware
        try:
            hw = detect_hardware()
        except Exception:
            return "igpu", "無法偵測顯示卡，以一般配備為準給建議。"
        gpus = hw.get("gpus") or []
        if not gpus:
            return "cpu", "未偵測到顯示卡 —— 以純處理器（CPU）推理為準給建議。"
        names = "、".join(g["name"] for g in gpus)
        if hw.get("has_discrete"):
            return "gpu", f"偵測到獨立顯示卡：{names} —— 可用最準的模型。"
        return "igpu", f"偵測到內建顯示晶片：{names} —— 建議用較輕量的模型。"

    def _present_for(self, backend: str, patch: dict) -> bool:
        """某個目錄項（backend + settings patch）的模型檔是否已下載。

        與 selected_model_present() 同樣的判斷，但改吃「任意一項」而非目前選定的，
        供基礎模式在卡片上標示「已下載／需下載」。
        """
        try:
            from downloader import (quick_check, quick_check_1p7b, quick_check_breeze)
            model_dir = self._model_dir()
            if backend == "openvino":
                return (quick_check_1p7b(model_dir)
                        if "1.7B" in patch.get("cpu_model_size", "0.6B")
                        else quick_check(model_dir))
            if backend == "crispasr":
                cm = patch.get("crisp_model", "breeze")
                if cm in _QWEN_CASR_MODELS:
                    _, check_fn, _ = _qwen_casr_dl(cm)
                    return check_fn(model_dir, patch.get("crisp_qwen_quant", "q8"))
                return quick_check_breeze(self._crispasr_dir(),
                                          patch.get("crisp_quant", "q5"))
            if backend == "chatllm":
                return ((self._chatllm_dir() / "qwen3-asr-1.7b.bin").exists()
                        or (model_dir / "qwen3-asr-1.7b.bin").exists())
        except Exception:
            traceback.print_exc()
        return False

    def get_basic_profiles(self) -> dict:
        """基礎模式的用途清單（已套用硬體建議）。

        回傳 {"tier","hardware","accel","profiles":[…],"current"}；每個 profile 帶
        它在本機硬體下建議的 (core, model) 與該模型是否已下載。前端點選後仍走
        既有的 set_model()／request_load()，不另開載入路徑。
        """
        from downloader import crispasr_variants
        tier, hw_text = self._hw_tier()
        cur_core, cur_model = self._current_selection()
        variant = self._crisp_variant()
        vmeta = crispasr_variants().get(variant, {})

        profiles = []
        for p in _BASIC_PROFILES:
            core_label, model_label = p["picks"].get(tier, p["picks"]["igpu"])
            entry = next((e for e in _MODEL_CATALOG
                          if e[0] == core_label and e[1] == model_label), None)
            if entry is None:            # 目錄改名時不要整頁壞掉，跳過該用途
                continue
            _, _, be, patch = entry
            note = p.get("note", "")
            if tier == "cpu" and be != "openvino":
                note = (note + "　" if note else "") + _BASIC_CPU_SLOW_NOTE
            profiles.append({
                "key": p["key"], "title": p["title"], "desc": p["desc"],
                "note": note,
                "core": core_label, "model": model_label,
                "arch": self._backend_label(be, be),
                "present": self._present_for(be, patch),
                "selected": (core_label == cur_core and model_label == cur_model),
            })
        return {
            "tier": tier, "hardware": hw_text,
            "accel": vmeta.get("short") or vmeta.get("label") or variant,
            "profiles": profiles,
            "current": {"core": cur_core, "model": cur_model},
        }

    def get_model_options(self) -> dict:
        """核心/模型階層 + 目前選擇 + 每模型對應架構標籤（前端渲染下拉用）。

        chatllm 為「向下相容」項：僅在 _chatllm_available 時納入清單（見其註解）。
        """
        cur_core, cur_model = self._current_selection()
        show_chatllm = self._chatllm_available()
        order, by_core = [], {}
        for core_label, model_label, be, _patch in _MODEL_CATALOG:
            if be == "chatllm" and not show_chatllm:
                continue
            if core_label not in by_core:
                by_core[core_label] = []
                order.append(core_label)
            by_core[core_label].append({
                "label": model_label, "backend": be,
                "arch": self._backend_label(be, be),
                "note": self._note_for(be, model_label),
            })
        active = getattr(self, "_active_backend", None)
        return {
            "cores": [{"label": c, "models": by_core[c]} for c in order],
            "current": {"core": cur_core, "model": cur_model},
            "activeArch": self._backend_label(active, "") if active else "",
        }

    def set_model(self, core_label: str, model_label: str) -> dict:
        """選定 (核心,模型) → 寫對應 settings 鍵、回是否需重啟。"""
        entry = next((e for e in _MODEL_CATALOG
                      if e[0] == core_label and e[1] == model_label), None)
        if entry is None:   # 防呆：退回該核心首項 / 目錄首項
            entry = next((e for e in _MODEL_CATALOG if e[0] == core_label),
                         _MODEL_CATALOG[0])
        core_label, model_label, backend, patch = entry

        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        try:
            cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            cur = {}
        cur["backend"] = backend
        cur.update(patch)                 # cpu_model_size / crisp_model / crisp_quant
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        arch = self._backend_label(backend, backend)
        # 尚未載入任何核心（首次啟動）→ 可「就地下載並載入」，免重啟（此時沒有
        # 既有 Vulkan context 會衝突，安全）。已載入核心後再換 → 仍需重啟（safety）。
        loaded_or_loading = self._loaded or self._loading
        identity_changed = (self._identity_for(backend, cur)
                            != getattr(self, "_active_identity", None))
        can_load_now = not loaded_or_loading
        restart = loaded_or_loading and identity_changed
        if can_load_now:
            msg = (f"已選定「{core_label} · {model_label}」（{arch}）。"
                   f"點「下載並載入模型」即可開始（不需重啟）。")
        elif restart:
            msg = (f"已記住「{core_label} · {model_label}」（{arch}）。"
                   f"切換核心需重新啟動程式以套用。")
        else:
            msg = f"「{core_label} · {model_label}」已是目前使用的模型。"
        note = self._note_for(backend, model_label)
        if note:                          # chatllm AMD 問題／TEA 域限制 → 一併提醒
            msg += "\n" + note
        return {
            "ok": True, "core": core_label, "model": model_label,
            "backend": backend, "arch": arch, "restartRequired": restart,
            "canLoadNow": can_load_now,
            "note": note,
            "message": msg,
        }

    def request_load(self) -> dict:
        """前端「下載並載入模型」→ 觸發背景載入（讀已持久化的 backend 選擇）。

        僅在尚未載入時有效（start_load 內含 guard）。下載/載入進度經 SSE
        "progress" 事件推送，完成時 "status" 事件 modelReady=True。
        """
        already = self._loaded or self._loading
        self.start_load()
        return {"ok": True, "loading": True, "alreadyLoaded": bool(self._loaded),
                "wasBusy": bool(already)}

    def get_languages(self) -> dict:
        """辨識語言清單：OpenVINO 用 processor.supported_languages，其餘用常用清單。

        value="" 代表自動偵測（transcribe 會轉成 None）；其餘 value 即引擎接受的
        語系字串（如 "Chinese"）。
        """
        langs = None
        try:
            proc = getattr(self.engine, "processor", None)
            if proc is not None and getattr(proc, "supported_languages", None):
                langs = list(proc.supported_languages)
        except Exception:
            langs = None
        if not langs:
            langs = _COMMON_LANGS
        return {"languages": [{"label": "自動偵測", "value": ""}]
                + [{"label": l, "value": l} for l in langs]}

    # ── 啟動自檢：每核心 × 每能力，實際探測檔案/鍵路是否就緒 ─────────────
    #   status：green=已就緒 / yellow=缺但會自動下載 / red=缺且需處理 / na=不適用
    def health_check(self) -> dict:
        from downloader import (quick_check, quick_check_1p7b, quick_check_aligner,
                                quick_check_diarization, quick_check_crispasr,
                                quick_check_breeze, quick_check_qwen3_asr_gguf,
                                quick_check_qwen3_asr_ja_gguf, quick_check_aligner_gguf,
                                quick_check_tea_asr_gguf, crispasr_variants,
                                _CRISPASR_VERSION)
        s = self._settings_raw()
        model_dir = self._model_dir()
        crispasr_dir = self._crispasr_dir()

        def item(key, label, ok, ok_d, miss_d, downloadable=True):
            return {"key": key, "label": label,
                    "status": "green" if ok else ("yellow" if downloadable else "red"),
                    "detail": ok_d if ok else miss_d}

        def _vad_ok():
            try:
                if getattr(self.engine, "vad_sess", None) is not None:
                    return True
                base = Path(getattr(core, "_MEIPASS", BASE_DIR)) if getattr(core, "_MEIPASS", None) else BASE_DIR
                for p in (model_dir / "silero_vad_v4.onnx",
                          BASE_DIR / "ov_models" / "silero_vad_v4.onnx"):
                    if p.exists():
                        return True
            except Exception:
                pass
            return False

        # 共用：說話者分離（外部 ONNX，所有核心共用）＋ ffmpeg
        diar_ok = quick_check_diarization(model_dir)
        ff = self._resolve_ffmpeg()

        # OpenVINO（CPU）路徑
        ov_size = s.get("cpu_model_size", "0.6B")
        ov_model_ok = quick_check_1p7b(model_dir) if "1.7B" in ov_size else quick_check(model_dir)
        ov = {
            "label": "Qwen · OpenVINO（CPU）", "backend": "openvino",
            "items": [
                item("model", f"ASR 模型（{ov_size}）", ov_model_ok,
                     "已下載", "未下載（啟用時自動下載）"),
                item("vad", "語音分段 VAD（silero）", _vad_ok(), "已內建", "缺 VAD onnx", downloadable=False),
                item("fa", "時間軸對齊 FA（ForcedAligner .bin）", quick_check_aligner(model_dir),
                     "已下載", "未下載（約 939MB，啟用對齊時下載）"),
                item("diar", "說話者分離（外部 ONNX）", diar_ok,
                     "已下載", "未下載（約 32MB，啟用分離時下載）"),
            ],
        }

        # CRISPASR 路徑：核心 exe + Breeze/Qwen/TEA 模型 + FA aligner gguf + 共用 diar
        #   核心的「就緒」條件含版本與加速版本 → 舊版或選了別的加速版本都算未就緒，
        #   會在載入時自動重抓（訊息裡標明目前選的是哪一個 build）。
        variant = self._crisp_variant()
        variants_meta = crispasr_variants()
        vmeta = variants_meta.get(variant, variants_meta["vulkan"])
        core_ok = quick_check_crispasr(crispasr_dir, variant)
        crisp_quant = s.get("crisp_quant", "q5")
        qwen_quant = s.get("crisp_qwen_quant", "q8")
        fa_quant = s.get("crisp_fa_quant", "q5")
        crisp = {
            "label": (f"CRISPASR（{vmeta.get('short') or vmeta['label']}："
                      f"Whisper/Breeze + Qwen + TEA）"),
            "backend": "crispasr",
            "items": [
                item("core", (f"CrispASR {_CRISPASR_VERSION} 核心"
                              f"（{vmeta.get('short') or vmeta['label']}）"),
                     core_ok, "已下載",
                     f"未下載或版本不符（約 {vmeta['size_mb']}MB，啟用時下載）"),
                item("breeze", f"Whisper/Breeze 模型（{crisp_quant.upper()}）",
                     quick_check_breeze(crispasr_dir, crisp_quant),
                     "已下載", "未下載（啟用時下載）"),
                item("qwen", f"Qwen3-ASR-1.7B 模型（{qwen_quant.upper()}）",
                     quick_check_qwen3_asr_gguf(model_dir, qwen_quant),
                     "已下載", "未下載（啟用時下載）"),
                item("qwen_ja", f"Qwen3-ASR-1.7B 日語動漫模型（{qwen_quant.upper()}）",
                     quick_check_qwen3_asr_ja_gguf(model_dir, qwen_quant),
                     "已下載", "未下載（日語增強，啟用時下載）"),
                item("tea", "TEA-ASR-1.1 台灣國語模型（Q8）",
                     quick_check_tea_asr_gguf(model_dir, "q8"),
                     "已下載", "未下載（繁中／台灣用語增強，啟用時下載）"),
                item("fa", f"時間軸對齊 FA（aligner gguf {fa_quant.upper()}）",
                     quick_check_aligner_gguf(crispasr_dir, fa_quant),
                     "已下載", "未下載（約 643MB，啟用時下載）"),
                item("diar", "說話者分離（外部 ONNX，與 OpenVINO 共用）", diar_ok,
                     "已下載", "未下載（約 32MB，啟用分離時下載）"),
            ],
        }

        shared = [
            item("ffmpeg", "FFmpeg（影片抽音軌用）", bool(ff),
                 f"已偵測：{ff}" if ff else "", "未下載（轉錄影片時自動下載解壓）", downloadable=True),
            item("diar_shared", "說話者分離模型（OpenVINO/CRISPASR 共用）", diar_ok,
                 "已下載", "未下載（啟用分離時自動下載）"),
        ]

        cores = [ov, crisp]

        # chatllm（向下相容）：僅在 _chatllm_available 時列出，讓既有使用者能確認
        # 其載入狀況。核心 DLL「不隨安裝包提供」→ 缺則紅燈（非可自動下載）；
        # .bin 模型缺則黃燈（啟用時自動下載）。FA／diar 與 OpenVINO 共用。
        if self._chatllm_available():
            chatllm_dir = self._chatllm_dir()
            dll_ok = (chatllm_dir / "libchatllm.dll").exists()
            bin_candidates = [
                Path(s.get("model_path") or "") if s.get("model_path") else None,
                Path(s.get("gguf_path") or "") if s.get("gguf_path") else None,
                getattr(core, "_BIN_PATH", None),
                model_dir / "qwen3-asr-1.7b.bin",
            ]
            bin_ok = any(p and Path(p).exists() for p in bin_candidates)
            chatllm = {
                "label": "Qwen · chatllm（Vulkan · 相容）", "backend": "chatllm",
                "items": [
                    item("core", "chatllm 核心（libchatllm.dll）", dll_ok,
                         "已就緒", "未提供（核心未隨安裝包附帶，僅供既有使用者）",
                         downloadable=False),
                    item("model", "Qwen3-ASR 模型（.bin）", bin_ok,
                         "已下載", "未下載（約 2.3GB，啟用時自動下載）"),
                    item("fa", "時間軸對齊 FA（ForcedAligner .bin）",
                         quick_check_aligner(model_dir),
                         "已下載", "未下載（約 939MB，啟用對齊時下載）"),
                    item("diar", "說話者分離（外部 ONNX，與 OpenVINO 共用）", diar_ok,
                         "已下載", "未下載（約 32MB，啟用分離時下載）"),
                ],
            }
            cores.append(chatllm)

        # 紅燈：缺且不可自動補（目前僅 VAD/ffmpeg 屬此類）
        reds = sum(1 for c in cores for it in c["items"] if it["status"] == "red")
        yellows = sum(1 for c in cores for it in c["items"] if it["status"] == "yellow")
        return {
            "cores": cores, "shared": shared,
            "summary": {"red": reds, "yellow": yellows,
                        "ok": reds == 0},
            "activeBackend": getattr(self, "_active_backend", None),
        }

    # ── 設定（讀寫既有 settings.json）───────────────────────
    def get_settings(self) -> dict:
        s = {}
        try:
            f = getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json")
            if Path(f).exists():
                s = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            s = {}
        # 簡繁/詞彙：app.py 以 output_simplified + vocab_convert 兩 bool 表達；
        # webview 前端用單一字串（off/s2twp/s2t）。此處還原成字串供前端預選。
        vc = s.get("vocab_convert", True)
        if isinstance(vc, str):                       # 舊版 webview 曾誤存字串 → 直接沿用
            vocab = vc if vc in ("off", "s2twp", "s2t") else "s2twp"
        else:
            vocab = self._flags_to_vocab(bool(s.get("output_simplified", False)), bool(vc))
        return {
            "scale": int(s.get("ui_scale", 100)),
            "format": s.get("output_format", "srt"),
            "vocab": vocab,
            "mirror": s.get("hf_mirror", ""),
            "ffmpeg": s.get("ffmpeg_path", ""),
            "theme": s.get("appearance", "light"),
            "uiLang": s.get("ui_lang", "繁體中文"),
            "vad": float(s.get("vad_threshold", 0.5)),
            "chunkSecs": int(s.get("chunk_secs", 0) or 0),   # 0=自動（依模型上限）
            # 模型頁模式：basic（用途導向，預設）／advanced（核心+模型全手動）
            "modelMode": (s.get("model_mode") if s.get("model_mode") in ("basic", "advanced")
                          else "basic"),
        }

    def set_settings(self, patch: dict) -> dict:
        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        cur = {}
        try:
            if f.exists():
                cur = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
        patch = patch or {}
        # 簡繁/詞彙：webview 單一字串 → 寫成 app.py 相容的兩個 bool。
        if "vocab" in patch:
            simp, conv = self._vocab_to_flags(patch["vocab"])
            cur["output_simplified"] = simp
            cur["vocab_convert"] = conv
        key_map = {"scale": "ui_scale", "format": "output_format",
                   "mirror": "hf_mirror", "ffmpeg": "ffmpeg_path", "theme": "appearance",
                   "uiLang": "ui_lang", "vad": "vad_threshold", "chunkSecs": "chunk_secs",
                   "modelMode": "model_mode"}
        for k, v in patch.items():
            if k in key_map:
                cur[key_map[k]] = v
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        # ── 即時套用副作用（與 app.py 的 _on_*_change 對齊）──────────────
        if "vad" in patch:
            self._apply_vad(patch["vad"])      # _detect_speech_groups 讀全域，立即生效
        if "vocab" in patch:
            self._apply_vocab(*self._vocab_to_flags(patch["vocab"]))
        if "mirror" in patch:
            self._apply_mirror(patch["mirror"])
        if "format" in patch:
            self._apply_output_format(patch["format"])
        if "theme" in patch and self._theme_cb:    # 通知視窗層同步標題列深淺
            try:
                self._theme_cb(patch["theme"])
            except Exception:
                pass
        return self.get_settings()

    def set_theme_callback(self, cb):
        """由 app_webview 註冊：theme 變更時呼叫 cb(theme_str) 同步視窗標題列深淺。"""
        self._theme_cb = cb

    def persisted_theme(self) -> str:
        """目前持久化的外觀設定（light/dark/system），供視窗層決定初始標題列。"""
        return self._settings_raw().get("appearance", "light") or "light"

    # ── 偏好即時生效（重用 app.py 的全域旗標機制）─────────────────────────
    @staticmethod
    def _vocab_to_flags(vocab: str):
        """webview vocab 字串 → (output_simplified, vocab_convert)。

        off   → 簡體輸出（不經 OpenCC 繁化）
        s2t   → 繁體・標準字形（僅字形轉換，保留原始用詞）
        s2twp → 繁體・台灣用語（含詞彙在地化，預設）
        """
        if vocab == "off":
            return True, True
        if vocab == "s2t":
            return False, False
        return False, True

    @staticmethod
    def _flags_to_vocab(output_simplified: bool, vocab_convert: bool):
        if output_simplified:
            return "off"
        return "s2twp" if vocab_convert else "s2t"

    def _apply_vocab(self, output_simplified: bool, vocab_convert: bool):
        """同步三引擎各自模組的簡繁/詞彙旗標，並即時重建當前引擎的 OpenCC。

        OpenVINO 讀 app(core)._g_*；chatllm/crisp 各讀自己模組級 _output_simplified/
        _vocab_convert。三者都有 rebuild_cc()，故先設旗標、再重建即可免重載生效。
        """
        import importlib
        try:
            core._g_output_simplified = bool(output_simplified)
            core._g_vocab_convert = bool(vocab_convert)
        except Exception:
            pass
        for name in ("chatllm_engine", "crisp_engine"):
            try:
                m = importlib.import_module(name)
                m._output_simplified = bool(output_simplified)
                m._vocab_convert = bool(vocab_convert)
            except Exception:
                pass
        eng = getattr(self, "engine", None)
        if eng is not None and hasattr(eng, "rebuild_cc"):
            try:
                eng.rebuild_cc()
            except Exception:
                pass

    def _apply_mirror(self, base):
        try:
            import downloader as _dl
            _dl.set_mirror((base or "").strip())
        except Exception:
            pass

    def _apply_output_format(self, fmt):
        try:
            import subtitle_lines as _subs
            _subs.OUTPUT_FORMAT = "txt" if str(fmt).lower() == "txt" else "srt"
        except Exception:
            pass

    def _apply_vad(self, value):
        """把 VAD 閾值即時套到 app 模組全域 —— _detect_speech_groups 於呼叫時讀取。"""
        try:
            core.VAD_THRESHOLD = float(value)
        except Exception:
            pass

    def _apply_runtime_prefs(self):
        """啟動時把持久化偏好一次套用到各引擎模組全域（對齊 app._apply_ui_prefs）。

        在 start_load() 之前呼叫 → 模組旗標先就位，引擎載入時即依正確 OpenCC
        設定建構 cc，且 VAD/鏡像/輸出格式皆於首次轉錄前生效。
        """
        s = self._settings_raw()
        try:
            self._apply_vad(s.get("vad_threshold", 0.5))
        except Exception:
            pass
        vc = s.get("vocab_convert", True)
        if isinstance(vc, str):
            simp, conv = self._vocab_to_flags(vc if vc in ("off", "s2twp", "s2t") else "s2twp")
        else:
            simp, conv = bool(s.get("output_simplified", False)), bool(vc)
        self._apply_vocab(simp, conv)
        self._apply_mirror(s.get("hf_mirror", ""))
        self._apply_output_format(s.get("output_format", "srt"))

    # ── LAN 端點服務（重用 api_server.TranscribeServer）─────
    def _endpoint_port(self) -> int:
        """目前要使用的監聽埠：已啟動的服務 > 設定值 > 預設 11435。"""
        if self._server:
            return self._server._port
        try:
            p = int(self._settings_raw().get("endpoint_port", 11435) or 11435)
            return p if 1 <= p <= 65535 else 11435
        except Exception:
            return 11435

    def get_endpoint(self) -> dict:
        from api_server import get_local_ip
        running = bool(self._server and self._server.running)
        host = get_local_ip()
        port = self._endpoint_port()
        key = self._server.token if self._server else ""
        url = f"http://{host}:{port}/?k={key}" if running else ""
        return {"running": running, "host": host, "port": port, "key": key, "url": url}

    def toggle_endpoint(self, on_: bool, port=None) -> dict:
        from api_server import TranscribeServer
        if on_:
            # 解析監聽埠：前端傳入 > 設定 > 預設；變更時持久化並重建服務。
            new_port = None
            if port is not None and str(port).strip():
                try:
                    cand = int(port)
                    if 1 <= cand <= 65535:
                        new_port = cand
                except Exception:
                    new_port = None
            target = new_port if new_port is not None else self._endpoint_port()
            if new_port is not None:
                self._persist_setting("endpoint_port", new_port)
                if self._server and self._server._port != new_port:
                    self._server.stop()                 # 換埠 → 丟棄舊服務重建
                    self._server = None
            if not self._server:
                self._server = TranscribeServer(get_engine=lambda: self.engine, port=target)
            self._server.start()
        elif self._server:
            self._stop_tunnel()              # 端點停 → 外網通道一併停（金鑰/埠失效）
            self._server.stop()
        return self.get_endpoint()

    def _persist_setting(self, key: str, value):
        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        try:
            cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        except Exception:
            cur = {}
        cur[key] = value
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def regen_key(self) -> dict:
        was = bool(self._server and self._server.running)
        self._stop_tunnel()                  # 換金鑰 → 舊外網網址失效，先收掉通道
        if self._server:
            self._server.stop()
        self._server = None
        if was:
            self.toggle_endpoint(True)
        return self.get_endpoint()

    # ── 對外臨時網址（Cloudflare quick tunnel）＋ QR ──────────────────────
    #   通道指向 LAN 端點（api_server）的埠＋金鑰，故須先啟動端點服務。
    #   建立耗時數秒 → 背景執行緒跑，狀態/網址經 SSE "tunnel" 事件推給前端。
    def get_tunnel(self) -> dict:
        t = self._tunnel
        return {
            "running": bool(t and t.running),
            "url": (t.url if t else "") or "",
            "status": (t.status if t else "") or "",
        }

    def toggle_tunnel(self, on_: bool) -> dict:
        import cf_tunnel
        if on_:
            if not (self._server and self._server.running):
                return {"running": False, "url": "", "status": "請先啟動端點服務", "error": True}
            if self._tunnel and self._tunnel.running:
                return self.get_tunnel()
            self._tunnel = cf_tunnel.CloudflareTunnel()
            port, token = self._server._port, self._server.token

            def worker():
                try:
                    self._tunnel.start(
                        port, token,
                        status_cb=lambda m: self._emit("tunnel", {
                            "status": m, "running": bool(self._tunnel and self._tunnel.running),
                            "url": (self._tunnel.url if self._tunnel else "") or "",
                        }))
                except Exception as e:
                    head = str(e).splitlines()[0][:120] if str(e) else type(e).__name__
                    self._emit("tunnel", {"status": f"建立失敗：{head}",
                                          "running": False, "url": "", "error": True})

            threading.Thread(target=worker, name="cf-tunnel", daemon=True).start()
            return {"running": True, "url": "", "status": "建立中…"}
        self._stop_tunnel()
        return self.get_tunnel()

    def _stop_tunnel(self):
        if self._tunnel:
            try:
                self._tunnel.stop()
            except Exception:
                pass
        self._emit("tunnel", {"status": "", "running": False, "url": ""})

    def make_qr(self, data: str) -> bytes | None:
        """回傳 QR 的 PNG 位元組（供 /api/qr 路由）。"""
        import cf_tunnel
        return cf_tunnel.make_qr_png(data) if data else None

    # ── 批次（桌面實機後續搬入）────────────────────────────
    def get_batch(self) -> dict:
        return {"summary": {"done": 0, "total": 0}, "items": []}
