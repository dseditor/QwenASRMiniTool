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
_INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QwenASR 轉錄服務</title>
<style>
  :root { --bg:#1b1d23; --card:#252830; --fg:#e6e8ee; --mut:#8b94a6; --acc:#4a90d9; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font-family:"Microsoft JhengHei","Segoe UI",sans-serif; }
  .wrap { max-width:760px; margin:0 auto; padding:28px 18px 60px; }
  h1 { font-size:20px; font-weight:700; margin:0 0 4px; }
  .sub { color:var(--mut); font-size:13px; margin-bottom:20px; }
  .card { background:var(--card); border-radius:12px; padding:18px; margin-bottom:16px; }
  .drop { border:2px dashed #3a3f4b; border-radius:10px; padding:30px; text-align:center;
          color:var(--mut); cursor:pointer; transition:.15s; }
  .drop.hot { border-color:var(--acc); color:var(--fg); background:#2b2f3a; }
  .drop b { color:var(--fg); }
  .row { display:flex; gap:14px; flex-wrap:wrap; margin-top:14px; align-items:center; }
  label { font-size:13px; color:var(--mut); display:flex; gap:6px; align-items:center; }
  select, input[type=number] { background:#1b1d23; color:var(--fg); border:1px solid #3a3f4b;
          border-radius:6px; padding:5px 8px; font-size:13px; }
  button { background:var(--acc); color:#fff; border:0; border-radius:8px; padding:10px 22px;
           font-size:15px; cursor:pointer; margin-top:16px; }
  button:disabled { opacity:.5; cursor:default; }
  pre { white-space:pre-wrap; word-break:break-word; background:#15171c; border-radius:8px;
        padding:14px; font-size:13px; max-height:46vh; overflow:auto; margin:0; }
  .bar { display:flex; gap:10px; margin-bottom:8px; }
  .bar button { margin:0; padding:6px 14px; font-size:13px; background:#3a3f4b; }
  .status { color:var(--mut); font-size:13px; margin-top:10px; min-height:18px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>QwenASR 轉錄服務</h1>
  <div class="sub">上傳音檔／影片 → 取得字幕（OpenAI 相容端點 /v1/audio/transcriptions）</div>

  <div class="card">
    <div id="drop" class="drop">拖放音檔／影片到這裡，或 <b>點此選擇</b>
      <div id="fname" style="margin-top:8px;color:var(--acc)"></div>
    </div>
    <input id="file" type="file" accept="audio/*,video/*" style="display:none">
    <div class="row">
      <label>語言
        <select id="lang">
          <option value="">自動偵測</option>
          <option>Chinese</option><option>English</option><option>Japanese</option>
          <option>Korean</option><option>Cantonese</option>
        </select>
      </label>
      <label>輸出
        <select id="fmt">
          <option value="srt">SRT 字幕</option>
          <option value="text">純文字</option>
          <option value="verbose_json">verbose_json</option>
        </select>
      </label>
      <label><input id="align" type="checkbox" checked> 時間軸對齊</label>
      <label><input id="diar" type="checkbox"> 說話者分離</label>
    </div>
    <button id="go" disabled>開始轉錄</button>
    <div id="status" class="status"></div>
  </div>

  <div class="card">
    <div class="bar">
      <button id="copy">複製</button>
      <button id="dl">下載 .srt</button>
    </div>
    <pre id="out">（結果會顯示在這裡）</pre>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
const KEY = new URLSearchParams(location.search).get('k') || '';
let picked = null;
const drop = $("#drop"), fileEl = $("#file");
drop.onclick = () => fileEl.click();
fileEl.onchange = e => setFile(e.target.files[0]);
["dragover","dragenter"].forEach(ev => drop.addEventListener(ev, e=>{e.preventDefault();drop.classList.add("hot");}));
["dragleave","drop"].forEach(ev => drop.addEventListener(ev, e=>{e.preventDefault();drop.classList.remove("hot");}));
drop.addEventListener("drop", e => { if(e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
function setFile(f){ picked=f; $("#fname").textContent=f?f.name:""; $("#go").disabled=!f; }

$("#go").onclick = async () => {
  if(!picked) return;
  const fd = new FormData();
  fd.append("file", picked);
  fd.append("language", $("#lang").value);
  fd.append("response_format", $("#fmt").value);
  fd.append("align", $("#align").checked ? "1":"0");
  fd.append("diarize", $("#diar").checked ? "1":"0");
  $("#go").disabled=true; $("#status").textContent="轉錄中…（長音檔需要一些時間）";
  const t0=Date.now();
  try{
    const r = await fetch("/v1/audio/transcriptions?k="+encodeURIComponent(KEY),
                          {method:"POST", body:fd,
                           headers: KEY ? {"Authorization":"Bearer "+KEY} : {}});
    const ctype = r.headers.get("Content-Type")||"";
    let text;
    if(ctype.includes("application/json")){
      const j = await r.json();
      if(j.error){ throw new Error(j.error.message||"server error"); }
      text = j.segments ? JSON.stringify(j, null, 2) : (j.text||"");
    } else { text = await r.text(); }
    $("#out").textContent = text || "（無內容）";
    $("#status").textContent = "完成，耗時 "+((Date.now()-t0)/1000).toFixed(1)+" 秒";
  }catch(err){
    $("#out").textContent=""; $("#status").textContent="❌ "+err.message;
  }finally{ $("#go").disabled=false; }
};
$("#copy").onclick = () => navigator.clipboard.writeText($("#out").textContent);
$("#dl").onclick = () => {
  const blob = new Blob([$("#out").textContent], {type:"text/plain"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = (picked? picked.name.replace(/\\.[^.]+$/,""):"transcript")+".srt";
  a.click();
};
</script>
</body>
</html>"""
