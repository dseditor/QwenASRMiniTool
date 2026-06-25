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
        self._on_event = on_event
        self._lock = threading.Lock()
        self._apply_vad_from_settings()  # 啟動即套用持久化的 VAD 閾值

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
                                aligner_gguf_filename)
        s = self._settings_raw()
        crispasr_dir = Path(s.get("crispasr_dir", str(BASE_DIR / "crispasr")))
        quant = s.get("crisp_quant", "q5")
        model_path = crispasr_dir / breeze_filename(quant)
        fa_enabled = bool(s.get("crisp_fa", True))
        fa_quant = s.get("crisp_fa_quant", "q5")

        if not quick_check_crispasr(crispasr_dir):
            self._st("下載 CrispASR 核心（約 27 MB）…")
            download_crispasr_core(crispasr_dir, progress_cb=self._dl_progress)
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
                 device_id=self._vk_device_id(s), cb=self._st, aligner_path=aligner_path)
        self.engine = eng

    # ── 狀態 ────────────────────────────────────────────────
    def get_status(self) -> dict:
        active = getattr(self, "_active_backend", "openvino")
        return {
            "modelReady": bool(getattr(self.engine, "ready", False)),
            "loading": self._loading,
            "error": self._load_err,
            "backend": self._BACKEND_LABELS.get(active, "CPU · OpenVINO INT8"),
            "backendKey": self._persisted_backend(),       # 已記住的核心選擇
            "activeBackend": active,                        # 實際載入中的核心
            "device": "GPU" if active in ("chatllm", "crispasr") else "CPU",
            "version": self._app_version(),
            "appName": "聲音辨識",
        }

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
            return version.__version__
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
                from ffmpeg_utils import find_ffmpeg, extract_audio_to_wav
                ff = find_ffmpeg()
                if not ff:
                    raise RuntimeError("上傳為影片但找不到 ffmpeg，無法抽音軌。")
                wav = audio_path.with_suffix(".extracted.wav")
                extract_audio_to_wav(audio_path, wav, ff)
                audio_path, tmp_extra = wav, wav

            if hasattr(self.engine, "use_aligner") and getattr(self.engine, "_fa_bin", None):
                self.engine.use_aligner = bool(opts.get("align", True))

            n_spk = opts.get("nSpeakers")
            n_speakers = int(n_spk) if str(n_spk).isdigit() else None

            with self._lock:
                srt = self.engine.process_file(
                    audio_path,
                    progress_cb=_cb,
                    language=opts.get("language") or None,
                    context=(opts.get("hint") or "").strip() or None,
                    diarize=bool(opts.get("diarize")),
                    n_speakers=n_speakers,
                    original_path=Path(path),
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
        if progress_cb:
            progress_cb(100, "完成")
        return {"segments": parse_srt_to_segments(srt), "srtPath": str(srt)}

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

    # ── 裝置 ────────────────────────────────────────────────
    def list_devices(self) -> dict:
        import platform
        devices = [{"kind": "cpu", "name": platform.processor() or "CPU", "note": "使用中"}]
        diag = {"level": None, "text": ""}
        try:
            chatllm_dir = str(getattr(core, "_CHATLLM_DIR", BASE_DIR))
            vk = core.probe_vulkan_devices(chatllm_dir)
            for d in vk.get("devices", []):
                gb = d.get("vram_free", 0) / (1024 ** 3)
                devices.append({"kind": "gpu", "name": d.get("name", "GPU"),
                                "note": f"{gb:.1f} GB 可用" if gb else ""})
            if vk.get("error"):
                diag = {"level": "warn", "text": f"GPU 偵測未完成：{vk['error']}　已自動改用 CPU 推理。"}
            elif not vk.get("devices"):
                diag = {"level": "info", "text": "未偵測到可用的獨立 GPU，僅 CPU 推理可用。"}
        except Exception as e:
            diag = {"level": "warn", "text": f"GPU 偵測例外：{e}"}
        return {"devices": devices, "diag": diag}

    # ── 核心切換：持久化選擇 + 請使用者重啟（不就地熱重載）─────
    #   理由：① chatllm DLL 在核心切換時 Vulkan context 未釋放會整機當機
    #   (見記憶 vulkan-dual-context-crash)；② 統一所有核心的切換行為，最安全。
    #   chatllm 保留原桌面實作以向下相容；未來全面改 casr，但切換一律需重啟。
    _BACKENDS = {0: "openvino", 1: "chatllm", 2: "crispasr"}
    _BACKEND_LABELS = {
        "openvino": "CPU · OpenVINO INT8",
        "chatllm":  "GPU · chatllm Vulkan",
        "crispasr": "GPU · Whisper / Breeze-ASR",
    }

    def set_backend(self, idx) -> dict:
        backend = self._BACKENDS.get(int(idx) if str(idx).isdigit() else 0, "openvino")
        label = self._BACKEND_LABELS.get(backend, backend)
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

    # ── 設定（讀寫既有 settings.json）───────────────────────
    def get_settings(self) -> dict:
        s = {}
        try:
            f = getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json")
            if Path(f).exists():
                s = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            s = {}
        return {
            "scale": int(s.get("ui_scale", 100)),
            "format": s.get("output_format", "srt"),
            "vocab": s.get("vocab_convert", "s2twp"),
            "mirror": s.get("hf_mirror", ""),
            "ffmpeg": s.get("ffmpeg_path", ""),
            "theme": s.get("appearance", "light"),
            "uiLang": s.get("ui_lang", "繁體中文"),
            "vad": float(s.get("vad_threshold", 0.5)),
        }

    def set_settings(self, patch: dict) -> dict:
        f = Path(getattr(core, "SETTINGS_FILE", BASE_DIR / "settings.json"))
        cur = {}
        try:
            if f.exists():
                cur = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
        key_map = {"scale": "ui_scale", "format": "output_format", "vocab": "vocab_convert",
                   "mirror": "hf_mirror", "ffmpeg": "ffmpeg_path", "theme": "appearance",
                   "uiLang": "ui_lang", "vad": "vad_threshold"}
        for k, v in (patch or {}).items():
            if k in key_map:
                cur[key_map[k]] = v
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        if "vad" in (patch or {}):
            self._apply_vad(patch["vad"])      # 即時生效（_detect_speech_groups 讀全域）
        return self.get_settings()

    def _apply_vad(self, value):
        """把 VAD 閾值即時套到 app 模組全域 —— _detect_speech_groups 於呼叫時讀取。"""
        try:
            core.VAD_THRESHOLD = float(value)
        except Exception:
            pass

    def _apply_vad_from_settings(self):
        try:
            self._apply_vad(self.get_settings().get("vad", 0.5))
        except Exception:
            pass

    # ── LAN 端點服務（重用 api_server.TranscribeServer）─────
    def get_endpoint(self) -> dict:
        from api_server import get_local_ip
        running = bool(self._server and self._server.running)
        host = get_local_ip()
        port = self._server._port if self._server else 11435
        key = self._server.token if self._server else ""
        url = f"http://{host}:{port}/?k={key}" if running else ""
        return {"running": running, "host": host, "port": port, "key": key, "url": url}

    def toggle_endpoint(self, on_: bool) -> dict:
        from api_server import TranscribeServer
        if on_:
            if not self._server:
                self._server = TranscribeServer(get_engine=lambda: self.engine, port=11435)
            self._server.start()
        elif self._server:
            self._server.stop()
        return self.get_endpoint()

    def regen_key(self) -> dict:
        was = bool(self._server and self._server.running)
        if self._server:
            self._server.stop()
        self._server = None
        if was:
            self.toggle_endpoint(True)
        return self.get_endpoint()

    # ── 批次（桌面實機後續搬入）────────────────────────────
    def get_batch(self) -> dict:
        return {"summary": {"done": 0, "total": 0}, "items": []}
