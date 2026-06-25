"""app_webview.py — 桌面版 WebView 啟動器（pywebview）

「三版共用同一前端」的桌面側：載入 webview/index.html，並把前端
window.QwenAPI 呼叫橋接到既有 ASREngine。長時間進度以
window.evaluate_js("QwenAPI._emit('progress', {...})") 主動推回前端。

重用策略：
  • 直接 import app（其 __main__ 有 guard，不會開 CTk 視窗），
    沿用同一份 ASREngine / 設定檔 / 端點服務，避免重造輪子。
  • 引擎較重的模型在背景執行緒載入；載入完成後 emit 'status'。

此檔尚屬第一階段：核心轉錄、選檔、設定、裝置、端點皆已接上；
批次／錄製的桌面實機流程沿用既有分頁邏輯，後續再逐步搬入。
"""
from __future__ import annotations

import json
import os
import threading
import traceback
from pathlib import Path

import webview

import app as core          # 重用 ASREngine / 常數 / 診斷函式（__main__ guard 保護）

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "webview"


def _seg_from_srt(srt_path: Path) -> list[dict]:
    """把 process_file 產出的 SRT 解析成前端要的 segments。

    說話者分離時，SRT 文字行以「說話者 N：…」開頭 → 拆出 speaker 與 text。
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


class JsApi:
    """前端 window.pywebview.api.* 的後端實作（對齊 bridge.js 契約）。"""

    def __init__(self):
        self.window: webview.Window | None = None
        self.engine = core.ASREngine()
        self._loaded = False
        self._cancel = False
        self._server = None              # TranscribeServer（端點）

    # ── 事件推送（→ 前端 event bus）─────────────────────────
    def _emit(self, event: str, payload: dict):
        if not self.window:
            return
        try:
            self.window.evaluate_js(
                f"window.QwenAPI && window.QwenAPI._emit({json.dumps(event)}, {json.dumps(payload)})")
        except Exception:
            pass

    # ── 啟動：背景載入模型 ──────────────────────────────────
    def start_load(self):
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            self.engine.load(device="CPU", model_dir=core._DEFAULT_MODEL_DIR,
                             cb=lambda m: self._emit("progress", {"pct": 0, "status": m}))
            self._loaded = True
            self._emit("status", {"modelReady": True})
        except Exception as e:
            traceback.print_exc()
            self._emit("status", {"modelReady": False, "error": str(e)})

    # ── 狀態 ────────────────────────────────────────────────
    def get_status(self):
        return {
            "modelReady": bool(getattr(self.engine, "ready", False)),
            "backend": "CPU · OpenVINO INT8",
            "device": "CPU",
            "version": getattr(core, "__version__", "") or self._app_version(),
            "appName": "聲音辨識",
        }

    def _app_version(self):
        try:
            import version
            return version.__version__
        except Exception:
            return ""

    # ── 原生選檔對話框 ──────────────────────────────────────
    def pick_file(self):
        types = ("音訊／影片 (*.mp3;*.wav;*.m4a;*.flac;*.mp4;*.mkv;*.mov)",
                 "所有檔案 (*.*)")
        res = self.window.create_file_dialog(webview.OPEN_DIALOG, allow_multiple=False,
                                             file_types=types)
        if not res:
            return None
        p = Path(res[0])
        return {"path": str(p), "name": p.name}

    def load_hint_txt(self):
        res = self.window.create_file_dialog(
            webview.OPEN_DIALOG, allow_multiple=False, file_types=("文字檔 (*.txt)",))
        if not res:
            return None
        try:
            return Path(res[0]).read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    # ── 轉錄 ────────────────────────────────────────────────
    def transcribe(self, opts: dict):
        if not self._loaded or not getattr(self.engine, "ready", False):
            raise RuntimeError("模型尚未載入完成，請稍候再試。")
        path = opts.get("path")
        if not path or not Path(path).exists():
            raise RuntimeError("找不到音訊檔，請重新選擇。")
        self._cancel = False

        total_holder = {"n": 1}

        def progress_cb(i, total, msg):
            total_holder["n"] = max(total, 1)
            pct = min(99, int((i / max(total, 1)) * 100))
            self._emit("progress", {"pct": pct, "status": msg})

        if hasattr(self.engine, "use_aligner") and getattr(self.engine, "_fa_bin", None):
            self.engine.use_aligner = bool(opts.get("align", True))

        n_spk = opts.get("nSpeakers")
        n_speakers = int(n_spk) if str(n_spk).isdigit() else None

        srt = self.engine.process_file(
            Path(path),
            progress_cb=progress_cb,
            language=opts.get("language") or None,
            context=(opts.get("hint") or "").strip() or None,
            diarize=bool(opts.get("diarize")),
            n_speakers=n_speakers,
            original_path=Path(path),
            out_format="srt",
        )
        if not srt:
            diag = getattr(self.engine, "_last_vad_diag", None)
            raise RuntimeError(diag or "未產生字幕（未偵測到人聲）。")
        self._emit("progress", {"pct": 100, "status": "完成"})
        return {"segments": _seg_from_srt(srt), "srtPath": str(srt)}

    def cancel(self):
        self._cancel = True
        return True

    def open_output_dir(self):
        try:
            d = getattr(core, "SRT_DIR", BASE_DIR / "subtitles")
            Path(d).mkdir(parents=True, exist_ok=True)
            os.startfile(str(d))
            return True
        except Exception:
            return False

    # ── 裝置 ────────────────────────────────────────────────
    def list_devices(self):
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

    def set_backend(self, idx):
        # 後端切換需重載模型，桌面整合進行中；先記錄選擇。
        return True

    # ── 設定（讀寫既有 settings.json）───────────────────────
    def get_settings(self):
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

    def set_settings(self, patch: dict):
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

    # ── 端點服務（重用 api_server.TranscribeServer）─────────
    def get_endpoint(self):
        from api_server import get_local_ip
        running = bool(self._server and self._server.running)
        host = get_local_ip()
        port = self._server._port if self._server else 11435
        key = self._server.token if self._server else ""
        url = f"http://{host}:{port}/?k={key}" if running else ""
        return {"running": running, "host": host, "port": port, "key": key, "url": url}

    def toggle_endpoint(self, on_: bool):
        from api_server import TranscribeServer
        if on_:
            if not self._server:
                self._server = TranscribeServer(get_engine=lambda: self.engine, port=11435)
            self._server.start()
        elif self._server:
            self._server.stop()
        return self.get_endpoint()

    def regen_key(self):
        import secrets
        was_running = bool(self._server and self._server.running)
        if self._server:
            self._server.stop()
        self._server = None
        if was_running:
            self.toggle_endpoint(True)
            self._server.token = secrets.token_urlsafe(12)
        return self.get_endpoint()

    # ── 批次（桌面實機流程後續搬入；先回空集）──────────────
    def get_batch(self):
        return {"summary": {"done": 0, "total": 0}, "items": []}

    def add_batch_files(self):
        return self.get_batch()

    def run_batch(self):
        return True


def main():
    api = JsApi()
    window = webview.create_window(
        "聲音辨識",
        url=str(WEB_DIR / "index.html"),
        js_api=api,
        width=1180, height=820, min_size=(960, 680),
    )
    api.window = window
    # 視窗就緒後才開始載入模型（確保 evaluate_js 可用）
    webview.start(api.start_load, debug=False)


if __name__ == "__main__":
    main()
