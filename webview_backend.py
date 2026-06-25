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
        try:
            self.engine.load(device="CPU", model_dir=core._DEFAULT_MODEL_DIR,
                             cb=lambda m: self._emit("progress", {"pct": 0, "status": m}))
            self._loaded = True
            self._emit("status", {"modelReady": True})
        except Exception as e:
            self._load_err = str(e)
            traceback.print_exc()
            self._emit("status", {"modelReady": False, "error": str(e)})
        finally:
            self._loading = False

    # ── 狀態 ────────────────────────────────────────────────
    def get_status(self) -> dict:
        return {
            "modelReady": bool(getattr(self.engine, "ready", False)),
            "loading": self._loading,
            "error": self._load_err,
            "backend": "CPU · OpenVINO INT8",
            "device": "CPU",
            "version": self._app_version(),
            "appName": "聲音辨識",
        }

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

    # ── 核心切換 ────────────────────────────────────────────
    #   OpenVINO：就地重載(安全)。
    #   chatllm / Whisper-Breeze(GPU)：不在本行程就地載入 —— chatllm DLL 在核心
    #   切換時 Vulkan context 未釋放會整機當機(見記憶 vulkan-dual-context-crash)，
    #   需沿用桌面版的裝置選擇與子程序隔離流程。先持久化選擇，回明確訊息。
    _BACKENDS = {0: "openvino", 1: "chatllm", 2: "crisp"}

    def set_backend(self, idx) -> dict:
        backend = self._BACKENDS.get(int(idx) if str(idx).isdigit() else 0, "openvino")
        self._persist_backend(backend)
        if backend == "openvino":
            if self._loading:
                return {"ok": False, "backend": backend, "message": "模型載入中，請稍候再切換。"}
            threading.Thread(target=self._reload_openvino, name="reload-ov", daemon=True).start()
            return {"ok": True, "backend": backend, "reloading": True}
        return {
            "ok": False, "backend": backend,
            "message": ("GPU 核心（chatllm / Whisper-Breeze）的切換尚未在 WebView 版接通。"
                        "為避免核心切換時的顯卡當機風險，請暫用桌面版切換，或維持 "
                        "CPU(OpenVINO) 核心。已記住你的選擇。"),
        }

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

    def _reload_openvino(self):
        import gc
        self._loading = True
        self._loaded = False
        self._load_err = None
        self._emit("status", {"modelReady": False, "loading": True})
        try:
            old = self.engine
            for attr in ("audio_enc", "embedder", "dec_req", "vad_sess"):
                if hasattr(old, attr):
                    setattr(old, attr, None)
            gc.collect()
            eng = core.ASREngine()
            eng.load(device="CPU", model_dir=core._DEFAULT_MODEL_DIR,
                     cb=lambda m: self._emit("progress", {"pct": 0, "status": m}))
            self.engine = eng
            self._loaded = True
            self._emit("status", {"modelReady": True})
        except Exception as e:
            self._load_err = str(e)
            traceback.print_exc()
            self._emit("status", {"modelReady": False, "error": str(e)})
        finally:
            self._loading = False

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
                   "uiLang": "ui_lang"}
        for k, v in (patch or {}).items():
            if k in key_map:
                cur[key_map[k]] = v
        try:
            f.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        return self.get_settings()

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
