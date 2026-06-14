"""api_server.py — OpenAI 相容音訊轉錄端點 + 簡易上傳網頁（純標準庫）

設計：
  • 與既有引擎共用：透過 get_engine() callable 取得「目前載入的」引擎，
    使用者切換後端重載模型後自動跟著換。
  • 零第三方依賴：http.server / 手寫 multipart 解析 / 內嵌 HTML。
  • 後端通用：engine 只要有 process_file()（OpenVINO 與 chatllm 皆有）即可。

路由：
  GET  /                          → 上傳網頁（self-contained）
  GET  /health                    → {"status":"ok", ...}
  POST /v1/audio/transcriptions   → OpenAI 相容轉錄（multipart 檔案上傳）
"""
from __future__ import annotations

import json
import secrets
import socket
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse, parse_qs

# 影片副檔名（需 ffmpeg 抽音軌）——與 ffmpeg_utils 一致即可，這裡延遲 import
_VIDEO_HINT = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
               ".ts", ".m2ts", ".mpg", ".mpeg", ".m4v", ".vob", ".3gp",
               ".f4v", ".mxf"}


# ── multipart/form-data 解析（手寫，避開已棄用的 cgi）──────────────────
def _parse_multipart(body: bytes, boundary: bytes):
    """回傳 (fields: dict[str,str], files: dict[str,(filename, bytes)])。"""
    fields: dict[str, str] = {}
    files: dict[str, tuple[str, bytes]] = {}
    sep = b"--" + boundary
    for part in body.split(sep):
        part = part.strip(b"\r\n")
        if not part or part == b"--":
            continue
        if b"\r\n\r\n" not in part:
            continue
        head, data = part.split(b"\r\n\r\n", 1)
        data = data[:-2] if data.endswith(b"\r\n") else data
        name = filename = None
        for line in head.decode("utf-8", "replace").split("\r\n"):
            if line.lower().startswith("content-disposition"):
                for seg in line.split(";"):
                    seg = seg.strip()
                    if seg.startswith("name="):
                        name = seg[5:].strip('"')
                    elif seg.startswith("filename="):
                        filename = seg[9:].strip('"')
        if name is None:
            continue
        if filename is not None:
            files[name] = (filename, data)
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


# ── SRT 解析（把 process_file 產出的 SRT 轉成 segments / 純文字）─────────
def _parse_srt(srt_text: str):
    """回傳 [{"id","start","end","text"}, ...]（start/end 為秒）。"""
    def _ts(t: str) -> float:
        t = t.strip().replace(",", ".")
        hh, mm, ss = t.split(":")
        return int(hh) * 3600 + int(mm) * 60 + float(ss)

    segs = []
    for block in srt_text.strip().split("\n\n"):
        lines = [l for l in block.splitlines() if l.strip()]
        if len(lines) < 2:
            continue
        # 找含 --> 的時間行
        ti = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if ti is None:
            continue
        try:
            a, b = lines[ti].split("-->")
            start, end = _ts(a), _ts(b)
        except Exception:
            continue
        text = " ".join(lines[ti + 1:]).strip()
        segs.append({"id": len(segs), "start": start, "end": end, "text": text})
    return segs


def get_local_ip() -> str:
    """取得本機區網 IP（供顯示連線網址）。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


class TranscribeServer:
    """背景 HTTP 轉錄服務。與 GUI 同行程，共用既有引擎。

    參數
    ----
    get_engine     : callable() -> engine | None（取得目前載入的引擎）
    port           : 監聽埠（預設 11435）
    host           : 預設 0.0.0.0（區網可連）
    token          : 存取金鑰；None 時自動產生。所有請求（網頁與端點）都需
                     攜帶（Authorization: Bearer <token> 或 ?k=<token>）。
                     金鑰隨「端點分頁」顯示的網址/QR 流動，等同密碼。
    on_log         : callable(str)，記錄訊息（可選）
    """

    def __init__(
        self,
        get_engine: Callable[[], object],
        port: int = 11435,
        host: str = "0.0.0.0",
        token: str | None = None,
        on_log: Callable[[str], None] | None = None,
    ):
        self._get_engine = get_engine
        self._port = port
        self._host = host
        self.token = token or secrets.token_urlsafe(12)
        self._on_log = on_log
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ── 生命週期 ──────────────────────────────────────────────────────
    @property
    def running(self) -> bool:
        return self._httpd is not None

    def _authorized(self, handler) -> bool:
        """檢查請求是否攜帶正確金鑰（Bearer 標頭或 ?k= 查詢參數）。"""
        if not self.token:
            return True
        auth = handler.headers.get("Authorization", "")
        if auth.startswith("Bearer ") and secrets.compare_digest(
                auth[7:].strip(), self.token):
            return True
        q = parse_qs(urlparse(handler.path).query)
        got = q.get("k", [""])[0]
        return bool(got) and secrets.compare_digest(got, self.token)

    def start(self):
        if self._httpd is not None:
            return
        server = self  # 閉包引用

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                if server._on_log:
                    server._on_log(fmt % args)

            def _send(self, code, ctype, body: bytes):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/health":
                    # 健康檢查不含敏感資訊，免金鑰（供探活）
                    eng = server._get_engine()
                    ready = bool(eng and getattr(eng, "ready", False))
                    self._send(200, "application/json",
                               json.dumps({"status": "ok", "model_ready": ready}).encode())
                elif path in ("", "/"):
                    if not server._authorized(self):
                        self._send(401, "text/html; charset=utf-8",
                                   _UNAUTH_HTML.encode("utf-8"))
                        return
                    self._send(200, "text/html; charset=utf-8", _INDEX_HTML.encode("utf-8"))
                else:
                    self._send(404, "application/json", b'{"error":"not found"}')

            def do_POST(self):
                if not urlparse(self.path).path.endswith("/audio/transcriptions"):
                    self._send(404, "application/json", b'{"error":"not found"}')
                    return
                if not server._authorized(self):
                    self._send(401, "application/json",
                               b'{"error":{"message":"unauthorized: missing or invalid key"}}')
                    return
                try:
                    server._handle_transcribe(self)
                except Exception as e:
                    msg = json.dumps({"error": {"message": str(e), "type": "server_error"}})
                    self._send(500, "application/json", msg.encode("utf-8"))

        self._httpd = ThreadingHTTPServer((self._host, self._port), _Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        if self._on_log:
            self._on_log(f"API 服務已啟動：http://{get_local_ip()}:{self._port}/")

    def stop(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
            self._thread = None
            if self._on_log:
                self._on_log("API 服務已停止")

    # ── 轉錄處理 ──────────────────────────────────────────────────────
    def _handle_transcribe(self, req: BaseHTTPRequestHandler):
        ctype = req.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype or "boundary=" not in ctype:
            self._reply_err(req, 400, "需以 multipart/form-data 上傳 file")
            return
        boundary = ctype.split("boundary=", 1)[1].strip().strip('"').encode()
        length = int(req.headers.get("Content-Length", 0))
        body = req.rfile.read(length)
        fields, files = _parse_multipart(body, boundary)

        if "file" not in files:
            self._reply_err(req, 400, "缺少 file 欄位")
            return
        filename, data = files["file"]
        if not data:
            self._reply_err(req, 400, "file 內容為空")
            return

        eng = self._get_engine()
        if not (eng and getattr(eng, "ready", False)):
            self._reply_err(req, 503, "模型尚未載入完成，請稍候")
            return

        # 參數
        resp_fmt = (fields.get("response_format") or "json").lower()
        language = fields.get("language") or None
        if language in ("", "auto", "自動偵測"):
            language = None
        diarize = (fields.get("diarize", "") or "").lower() in ("1", "true", "yes", "on")
        n_spk_raw = fields.get("n_speakers", "")
        n_speakers = int(n_spk_raw) if n_spk_raw.isdigit() else None
        align_raw = fields.get("align", "")
        align = (align_raw.lower() in ("1", "true", "yes", "on")) if align_raw else None

        # 存暫存檔（保留原副檔名讓 librosa/ffmpeg 判斷格式）
        ext = Path(filename).suffix or ".wav"
        tmp_dir = Path(tempfile.mkdtemp(prefix="asr_api_"))
        in_path = tmp_dir / ("upload" + ext)
        in_path.write_bytes(data)

        srt_path = None
        prev_align = getattr(eng, "use_aligner", None)
        try:
            audio_path = in_path
            original = in_path
            # 影片 → 先抽音軌
            if ext.lower() in _VIDEO_HINT:
                from ffmpeg_utils import find_ffmpeg, extract_audio_to_wav
                ff = find_ffmpeg()
                if not ff:
                    self._reply_err(req, 400, "上傳為影片但找不到 ffmpeg，無法抽音軌")
                    return
                wav_path = tmp_dir / "audio.wav"
                extract_audio_to_wav(in_path, wav_path, ff)
                audio_path = wav_path

            # 時間軸對齊（best-effort：僅在 FA 就緒時可切換）
            if align is not None and hasattr(eng, "use_aligner") and getattr(eng, "_fa_bin", None):
                eng.use_aligner = align

            srt_path = eng.process_file(
                audio_path,
                language=language,
                diarize=diarize,
                n_speakers=n_speakers,
                original_path=original,
            )
        finally:
            if prev_align is not None and hasattr(eng, "use_aligner"):
                eng.use_aligner = prev_align

        srt_text = ""
        if srt_path and Path(srt_path).exists():
            srt_text = Path(srt_path).read_text(encoding="utf-8")

        segs = _parse_srt(srt_text)
        plain = "".join(s["text"] for s in segs) if segs else ""
        # 含說話者前綴時用換行較可讀
        if any("：" in s["text"] for s in segs):
            plain = "\n".join(s["text"] for s in segs)

        # 清理暫存
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        # 依 response_format 回應
        if resp_fmt == "srt":
            self._send_ok(req, "text/plain; charset=utf-8", srt_text.encode("utf-8"))
        elif resp_fmt == "text":
            self._send_ok(req, "text/plain; charset=utf-8", plain.encode("utf-8"))
        elif resp_fmt == "verbose_json":
            out = {"task": "transcribe", "language": language or "auto",
                   "text": plain, "segments": segs}
            self._send_ok(req, "application/json",
                          json.dumps(out, ensure_ascii=False).encode("utf-8"))
        else:  # json（OpenAI 預設）
            self._send_ok(req, "application/json",
                          json.dumps({"text": plain}, ensure_ascii=False).encode("utf-8"))

    # ── 回應輔助 ──────────────────────────────────────────────────────
    def _send_ok(self, req, ctype, body: bytes):
        req.send_response(200)
        req.send_header("Content-Type", ctype)
        req.send_header("Content-Length", str(len(body)))
        req.end_headers()
        req.wfile.write(body)

    def _reply_err(self, req, code, msg):
        body = json.dumps({"error": {"message": msg}}, ensure_ascii=False).encode("utf-8")
        req.send_response(code)
        req.send_header("Content-Type", "application/json")
        req.send_header("Content-Length", str(len(body)))
        req.end_headers()
        req.wfile.write(body)


# ── 未授權頁（缺金鑰）──────────────────────────────────────────────────
_UNAUTH_HTML = """<!DOCTYPE html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>401 未授權</title></head>
<body style="background:#1b1d23;color:#e6e8ee;font-family:'Microsoft JhengHei',sans-serif;text-align:center;padding-top:18vh">
<h2>&#128274; 需要存取金鑰</h2>
<p style="color:#8b94a6">請使用 QwenASR「端點」分頁顯示的<b>完整網址</b>（含金鑰）開啟，<br>
或掃描分頁提供的 QR code。直接連線網域是無效的。</p>
</body></html>"""


# ── 內嵌上傳網頁（self-contained，無 CDN，可離線）─────────────────────
#    設計：玻璃質感深色介面、漸層強調、行動優先（響應式 grid + 44px 觸控目標）。
#    功能：① 檔案上傳轉錄 ② 即時錄音（停頓自動切段 / 按停止 → 上傳辨識）。
#    注意：此為純標準庫伺服器吐出的「單一自包含 HTML 字串」，刻意不引入任何
#    CDN / 建置步驟，以維持零依賴並可內嵌 EXE、離線可用。
_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0f1117">
<title>QwenASR 轉錄服務</title>
<style>
  :root{
    --bg0:#0f1117; --fg:#eef1f7; --mut:#9aa3b2; --line:rgba(255,255,255,.09);
    --card:rgba(255,255,255,.045); --acc:#4a90d9; --acc2:#5ec8c0; --danger:#ef5a6f;
  }
  *{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  html,body{ margin:0; }
  body{
    color:var(--fg); font-size:15px; line-height:1.5;
    font-family:"Microsoft JhengHei","Segoe UI",system-ui,-apple-system,sans-serif;
    background:
      radial-gradient(1100px 560px at 78% -12%, rgba(74,144,217,.20), transparent 58%),
      radial-gradient(900px 500px at -10% 8%, rgba(94,200,192,.12), transparent 55%),
      linear-gradient(180deg,#0f1117 0%, #0b0d13 100%);
    background-attachment:fixed; min-height:100vh;
  }
  .wrap{ max-width:740px; margin:0 auto; padding:max(22px,env(safe-area-inset-top)) 16px 64px; }

  /* 頁首 */
  .head{ display:flex; align-items:center; gap:12px; margin-bottom:18px; }
  .mark{ width:44px; height:44px; border-radius:13px; display:grid; place-items:center;
    font-size:22px; background:linear-gradient(135deg,var(--acc),#3a78c0);
    box-shadow:0 8px 22px rgba(74,144,217,.40); flex:0 0 auto; }
  .head h1{ font-size:19px; font-weight:800; margin:0; letter-spacing:.2px; }
  .head .sub{ color:var(--mut); font-size:12.5px; margin-top:2px; }

  /* 卡片（玻璃質感） */
  .card{ background:var(--card); border:1px solid var(--line); border-radius:18px;
    padding:18px; margin-bottom:16px; backdrop-filter:blur(14px);
    -webkit-backdrop-filter:blur(14px); box-shadow:0 10px 30px rgba(0,0,0,.28); }

  /* 模式切換（segmented） */
  .seg{ display:flex; gap:5px; padding:5px; margin-bottom:16px; width:100%;
    background:rgba(255,255,255,.05); border:1px solid var(--line); border-radius:14px; }
  .seg button{ flex:1; background:transparent; color:var(--mut); border:0; cursor:pointer;
    padding:11px 10px; border-radius:10px; font-size:14px; font-weight:600; transition:.18s; }
  .seg button.on{ color:#fff; background:linear-gradient(135deg,var(--acc),#3a78c0);
    box-shadow:0 6px 16px rgba(74,144,217,.38); }

  /* 拖放區 */
  .drop{ border:1.5px dashed var(--line); border-radius:14px; padding:30px 16px;
    text-align:center; color:var(--mut); cursor:pointer; transition:.18s; }
  .drop:hover{ border-color:rgba(74,144,217,.6); }
  .drop.hot{ border-color:var(--acc); color:var(--fg); background:rgba(74,144,217,.10); }
  .drop b{ color:var(--acc); }
  .fname{ margin-top:10px; color:var(--acc2); font-size:13px; word-break:break-all; }

  /* 選項 grid */
  .opts{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px; margin-top:16px; }
  .fld{ display:flex; flex-direction:column; gap:5px; font-size:12.5px; color:var(--mut); }
  select{ width:100%; background:rgba(0,0,0,.28); color:var(--fg);
    border:1px solid var(--line); border-radius:10px; padding:10px 10px; font-size:14px;
    appearance:none; }
  .checks{ display:flex; gap:18px; flex-wrap:wrap; margin-top:14px; }
  .chk{ display:flex; align-items:center; gap:7px; font-size:13.5px; color:var(--fg);
    cursor:pointer; }
  .chk input{ width:18px; height:18px; accent-color:var(--acc); }

  /* 按鈕 */
  .btn{ width:100%; margin-top:16px; border:0; border-radius:13px; cursor:pointer;
    padding:14px; font-size:15.5px; font-weight:700; color:#fff;
    background:linear-gradient(135deg,var(--acc),#3a78c0);
    box-shadow:0 8px 20px rgba(74,144,217,.34); transition:.15s; }
  .btn:active{ transform:translateY(1px); }
  .btn:disabled{ opacity:.45; box-shadow:none; cursor:default; }
  .status{ color:var(--mut); font-size:13px; margin-top:12px; min-height:18px; text-align:center; }

  /* 錄音區 */
  .rec-wrap{ display:flex; flex-direction:column; align-items:center; gap:14px; padding:8px 0; }
  .mic{ width:96px; height:96px; border-radius:50%; border:0; cursor:pointer; font-size:38px;
    color:#fff; display:grid; place-items:center; transition:.15s;
    background:linear-gradient(135deg,var(--acc),#3a78c0);
    box-shadow:0 10px 26px rgba(74,144,217,.40); }
  .mic:active{ transform:scale(.96); }
  .mic.rec{ background:linear-gradient(135deg,var(--danger),#c8344a);
    animation:pulse 1.5s infinite; }
  @keyframes pulse{
    0%{ box-shadow:0 0 0 0 rgba(239,90,111,.55); }
    70%{ box-shadow:0 0 0 24px rgba(239,90,111,0); }
    100%{ box-shadow:0 0 0 0 rgba(239,90,111,0); } }
  .mic:disabled{ opacity:.45; cursor:default; box-shadow:none; }
  .meter{ width:200px; max-width:70%; height:8px; border-radius:6px;
    background:rgba(255,255,255,.10); overflow:hidden; }
  .meter > i{ display:block; height:100%; width:0%;
    background:linear-gradient(90deg,var(--acc2),var(--acc)); transition:width .08s linear; }
  .rec-hint{ color:var(--mut); font-size:12.5px; text-align:center; }
  .warn{ color:#f4b860; font-size:12.5px; text-align:center; line-height:1.55; }

  /* 結果 */
  .res-head{ display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
  .res-head .t{ font-size:14px; font-weight:700; }
  .tools{ display:flex; gap:8px; }
  .tool{ background:rgba(255,255,255,.07); color:var(--fg); border:1px solid var(--line);
    border-radius:9px; padding:7px 14px; font-size:13px; cursor:pointer; transition:.15s; }
  .tool:active{ transform:translateY(1px); }
  pre{ white-space:pre-wrap; word-break:break-word; margin:0; font-size:13.5px;
    background:rgba(0,0,0,.30); border:1px solid var(--line); border-radius:12px;
    padding:14px; max-height:44vh; overflow:auto; line-height:1.6;
    font-family:"SFMono-Regular",Consolas,"Microsoft JhengHei",monospace; }
  pre::-webkit-scrollbar{ width:9px; } pre::-webkit-scrollbar-thumb{
    background:rgba(255,255,255,.14); border-radius:8px; }
  .foot{ text-align:center; color:var(--mut); font-size:11.5px; margin-top:8px; }
  .hidden{ display:none !important; }
</style>
</head>
<body>
<div class="wrap">
  <div class="head">
    <div class="mark">&#127908;</div>
    <div>
      <h1>QwenASR 轉錄服務</h1>
      <div class="sub">上傳或錄音 → 取得字幕 · OpenAI 相容端點</div>
    </div>
  </div>

  <div class="seg">
    <button id="m-file" class="on">&#128193; 上傳檔案</button>
    <button id="m-rec">&#127908; 即時錄音</button>
  </div>

  <!-- 上傳模式 -->
  <div id="pane-file" class="card">
    <div id="drop" class="drop">拖放音檔／影片到這裡，或 <b>點此選擇</b>
      <div id="fname" class="fname"></div>
    </div>
    <input id="file" type="file" accept="audio/*,video/*" class="hidden">
    <button id="go" class="btn" disabled>開始轉錄</button>
    <div id="status" class="status"></div>
  </div>

  <!-- 錄音模式 -->
  <div id="pane-rec" class="card hidden">
    <div class="rec-wrap">
      <button id="mic" class="mic">&#127908;</button>
      <div class="meter"><i id="lvl"></i></div>
      <div id="rec-status" class="rec-hint">按麥克風開始錄音。說完停頓約 2 秒自動上傳；句中短暫停頓不會中斷，不必趕著講。</div>
      <div id="rec-warn" class="warn hidden"></div>
    </div>
  </div>

  <!-- 共用選項 -->
  <div class="card">
    <div class="opts">
      <label class="fld">語言
        <select id="lang">
          <option value="">自動偵測</option>
          <option>Chinese</option><option>English</option><option>Japanese</option>
          <option>Korean</option><option>Cantonese</option><option>French</option>
          <option>German</option><option>Spanish</option><option>Russian</option>
        </select>
      </label>
      <label class="fld">輸出格式
        <select id="fmt">
          <option value="srt">SRT 字幕</option>
          <option value="text">純文字</option>
          <option value="verbose_json">verbose_json</option>
        </select>
      </label>
    </div>
    <div class="checks">
      <label class="chk"><input id="align" type="checkbox" checked> 時間軸對齊</label>
      <label class="chk"><input id="diar" type="checkbox"> 說話者分離</label>
    </div>
  </div>

  <!-- 結果 -->
  <div class="card">
    <div class="res-head">
      <span class="t">辨識結果</span>
      <div class="tools">
        <button id="clear" class="tool">清除</button>
        <button id="copy" class="tool">複製</button>
        <button id="dl" class="tool">下載</button>
      </div>
    </div>
    <pre id="out">（結果會顯示在這裡）</pre>
  </div>
  <div class="foot">QwenASR · 本地推理 · 資料不離開你的伺服器</div>
</div>
<script>
const $ = s => document.querySelector(s);
const KEY = new URLSearchParams(location.search).get('k') || '';
const fmt = () => $("#fmt").value;
let picked = null, lastName = "transcript";

/* ── 共用上傳 ─────────────────────────────────────────────── */
async function upload(blob, filename, statusEl, append){
  const fd = new FormData();
  fd.append("file", blob, filename);
  fd.append("language", $("#lang").value);
  fd.append("response_format", fmt());
  fd.append("align", $("#align").checked ? "1":"0");
  fd.append("diarize", $("#diar").checked ? "1":"0");
  const t0 = Date.now();
  if(statusEl) statusEl.textContent = "辨識中…";
  const r = await fetch("/v1/audio/transcriptions?k="+encodeURIComponent(KEY),
        {method:"POST", body:fd, headers: KEY ? {"Authorization":"Bearer "+KEY} : {}});
  const ctype = r.headers.get("Content-Type")||"";
  let text;
  if(ctype.includes("application/json")){
    const j = await r.json();
    if(j.error){ throw new Error(j.error.message||"server error"); }
    text = j.segments ? JSON.stringify(j, null, 2) : (j.text||"");
  } else { text = await r.text(); }
  const out = $("#out");
  if(append){
    const t = (text||"").trim();
    if(t){ out.textContent = (out.dataset.has ? out.textContent + "\\n" : "") + t; out.dataset.has = "1"; }
  } else {
    out.textContent = text || "（無內容）"; out.dataset.has = text ? "1" : "";
  }
  out.scrollTop = out.scrollHeight;
  if(statusEl) statusEl.textContent = "完成（"+((Date.now()-t0)/1000).toFixed(1)+" 秒）";
  return text;
}

/* ── 模式切換 ─────────────────────────────────────────────── */
function mode(m){
  const f = m==="file";
  $("#m-file").classList.toggle("on", f);
  $("#m-rec").classList.toggle("on", !f);
  $("#pane-file").classList.toggle("hidden", !f);
  $("#pane-rec").classList.toggle("hidden", f);
  if(!f) initRec();
}
$("#m-file").onclick = ()=>mode("file");
$("#m-rec").onclick  = ()=>mode("rec");

/* ── 上傳模式：拖放 / 選檔 ──────────────────────────────────── */
const drop = $("#drop"), fileEl = $("#file");
drop.onclick = () => fileEl.click();
fileEl.onchange = e => setFile(e.target.files[0]);
["dragover","dragenter"].forEach(ev => drop.addEventListener(ev, e=>{e.preventDefault();drop.classList.add("hot");}));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, e=>{e.preventDefault();drop.classList.remove("hot");}));
drop.addEventListener("drop", e => { if(e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
function setFile(f){ picked=f; $("#fname").textContent=f?f.name:""; $("#go").disabled=!f; }

$("#go").onclick = async () => {
  if(!picked) return;
  $("#go").disabled=true;
  try{
    lastName = picked.name.replace(/\\.[^.]+$/,"") || "transcript";
    await upload(picked, picked.name, $("#status"), false);
  }catch(err){ $("#status").textContent = "\\u274c "+err.message; }
  finally{ $("#go").disabled=false; }
};

/* ── 結果工具 ─────────────────────────────────────────────── */
$("#copy").onclick = () => navigator.clipboard.writeText($("#out").textContent);
$("#clear").onclick = () => { const o=$("#out"); o.textContent="（結果會顯示在這裡）"; o.dataset.has=""; };
$("#dl").onclick = () => {
  const ext = fmt()==="srt" ? ".srt" : ".txt";
  const blob = new Blob([$("#out").textContent], {type:"text/plain;charset=utf-8"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = lastName + ext; a.click();
};

/* ── 錄音模式：MediaRecorder + 停頓偵測（VAD）──────────────── */
let recInited=false, stream=null, recorder=null, chunks=[],
    audioCtx=null, analyser=null, rafId=0,
    sessionOn=false, speechStarted=false, silentSince=0, segStart=0;
// SILENCE_MS 拉長到 2.2s：自然講話的句中停頓（思考/換氣，常 1～2s）不會被切碎；
// 真正的「說完」通常停更久，或直接按麥克風結束。降低雙重 VAD 對「暫停」認知的落差。
const SILENCE_MS = 2200, MIN_SEG_MS = 500, MAX_SEG_MS = 20000, VAD_THRESH = 0.014;

function recSupported(){
  return window.isSecureContext && navigator.mediaDevices &&
         navigator.mediaDevices.getUserMedia && window.MediaRecorder;
}
function initRec(){
  if(recInited) return; recInited = true;
  if(!recSupported()){
    $("#mic").disabled = true;
    const w = $("#rec-warn"); w.classList.remove("hidden");
    w.innerHTML = window.isSecureContext
      ? "此瀏覽器不支援錄音 API。"
      : "&#128274; 即時錄音需在 <b>HTTPS</b> 或 localhost 開啟。<br>請改用「端點」分頁的<b>對外臨時網址 / QR</b>（自帶 https）連線。";
    $("#rec-status").textContent = "目前連線無法使用麥克風。";
  }
}
$("#mic").onclick = () => { sessionOn ? stopSession(true) : startSession(); };

async function startSession(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true}});
  }catch(err){ $("#rec-status").textContent = "\\u274c 無法取得麥克風：" + err.message; return; }
  audioCtx = new (window.AudioContext||window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(stream);
  analyser = audioCtx.createAnalyser(); analyser.fftSize = 1024;
  src.connect(analyser);
  sessionOn = true;
  $("#mic").classList.add("rec"); $("#mic").innerHTML = "&#9209;";
  $("#rec-status").textContent = "聆聽中…說完停頓約 2 秒自動上傳（句中短停不中斷），再按麥克風結束。";
  startSegment(); monitor();
}
function startSegment(){
  chunks = []; speechStarted = false; silentSince = 0; segStart = Date.now();
  let mime = ["audio/webm;codecs=opus","audio/webm","audio/mp4"].find(
      m => MediaRecorder.isTypeSupported(m)) || "";
  recorder = mime ? new MediaRecorder(stream,{mimeType:mime}) : new MediaRecorder(stream);
  recorder.ondataavailable = e => { if(e.data && e.data.size) chunks.push(e.data); };
  recorder.onstop = onSegStop;
  recorder.start();
}
function cutSegment(){ if(recorder && recorder.state==="recording") recorder.stop(); }

async function onSegStop(){
  const dur = Date.now() - segStart;
  const blob = new Blob(chunks, {type: chunks[0] ? chunks[0].type : "audio/webm"});
  const shouldUpload = speechStarted && dur >= MIN_SEG_MS && blob.size > 1200;
  // 先立刻接著錄下一段，避免上傳（可能數秒）期間漏掉使用者繼續說的話
  if(sessionOn) startSegment();
  else teardown();
  // 再背景上傳前一段（不阻塞錄音）；副檔名一律 .webm → 伺服器走 ffmpeg 抽軌
  if(shouldUpload){
    try{ await upload(blob, "recording.webm", $("#rec-status"), true); }
    catch(err){ $("#rec-status").textContent = "\\u274c " + err.message; }
  }
}
function monitor(){
  const buf = new Uint8Array(analyser.fftSize);
  const tick = () => {
    if(!sessionOn){ return; }
    analyser.getByteTimeDomainData(buf);
    let sum=0; for(let i=0;i<buf.length;i++){ const v=(buf[i]-128)/128; sum+=v*v; }
    const rms = Math.sqrt(sum/buf.length);
    $("#lvl").style.width = Math.min(100, rms*420) + "%";
    const now = Date.now();
    if(rms >= VAD_THRESH){ speechStarted = true; silentSince = 0; }
    else if(speechStarted){
      if(!silentSince) silentSince = now;
      else if(now - silentSince >= SILENCE_MS){ cutSegment(); }  // 停頓 → 切段上傳
    }
    if(Date.now() - segStart >= MAX_SEG_MS && speechStarted){ cutSegment(); } // 上限強制切
    rafId = requestAnimationFrame(tick);
  };
  rafId = requestAnimationFrame(tick);
}
function stopSession(){
  sessionOn = false;
  $("#mic").classList.remove("rec"); $("#mic").innerHTML = "&#127908;";
  $("#rec-status").textContent = "已停止。";
  if(rafId) cancelAnimationFrame(rafId);
  $("#lvl").style.width = "0%";
  cutSegment();   // 收尾段（onSegStop 會上傳並 teardown）
}
function teardown(){
  try{ if(stream) stream.getTracks().forEach(t=>t.stop()); }catch(e){}
  try{ if(audioCtx) audioCtx.close(); }catch(e){}
  stream = null; audioCtx = null; analyser = null; recorder = null;
}
</script>
</body>
</html>"""
