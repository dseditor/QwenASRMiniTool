"""translator.py — 字幕翻譯（轉錄完成後的選用步驟）

做法
----
用 crispasr.exe 的 server 模式常駐一顆翻譯用的小型 LLM
（LinguaForge，Qwen3.5-0.8B 微調，繁中↔英↔日），逐行送
`/v1/chat/completions`。時間軸完全不動，只多出每行的譯文。

為什麼是 server 而不是每行開一次 crispasr.exe：模型載入約 5 秒，常駐後
每行只要 0.03～0.1 秒（RTX 5090 實測）；一集 500 行逐行開程序要等很久。

為什麼一行一次：這顆模型對「多行且各行不相關」的輸入只會翻第一行
（作者 README 的已知限制），一次丟多行回來的行數會對不上。

提示詞格式與訓練資料一致，不要改：
    system: You are a professional translator.
    user:   翻譯成繁體中文：\\n{原文}
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

# 目標語言代碼 → (提示詞開頭, 輸出檔名的語言標記, 顯示名稱)
TARGETS: dict[str, tuple[str, str, str]] = {
    "zh": ("翻譯成繁體中文：", "zh-TW", "繁體中文"),
    "en": ("翻譯成英文：",     "en",    "英文"),
    "ja": ("翻譯成日文：",     "ja",    "日文"),
}

_SYSTEM = "You are a professional translator."

if os.name == "nt":
    _CREATE_NO_WINDOW = 0x08000000
    _STARTUP_INFO = subprocess.STARTUPINFO()
    _STARTUP_INFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
else:                                    # pragma: no cover
    _CREATE_NO_WINDOW = 0
    _STARTUP_INFO = None

# 127.0.0.1 不該經過系統代理；有些環境設了 HTTP_PROXY 會把本機請求也送出去。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def normalize_target(value) -> str:
    """設定值／參數 → 合法的目標語言代碼；不翻譯回空字串。"""
    v = str(value or "").strip().lower()
    return v if v in TARGETS else ""


def lang_tag(target: str) -> str:
    return TARGETS[target][1]


# ── 語言判斷：內容已經是目標語言就不用翻 ──────────────────────────────
def _script_counts(text: str) -> tuple[int, int, int, int]:
    """回傳 (漢字, 假名, 諺文, 拉丁字母) 的字數。"""
    han = kana = hangul = latin = 0
    for ch in text:
        o = ord(ch)
        if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            han += 1
        elif 0x3040 <= o <= 0x30FF:
            kana += 1
        elif 0xAC00 <= o <= 0xD7A3:
            hangul += 1
        elif ch.isascii() and ch.isalpha():
            latin += 1
    return han, kana, hangul, latin


def detect_language(texts) -> str:
    """粗略判斷整份字幕的語言：zh / ja / ko / en / ""（判斷不出來）。

    只用來決定「要不要翻」，不需要很準：日文靠假名、韓文靠諺文、中文靠漢字
    且幾乎沒有假名、英文靠拉丁字母。
    """
    han, kana, hangul, latin = _script_counts("".join(texts))
    total = han + kana + hangul + latin
    if total == 0:
        return ""
    if kana / total > 0.10:
        return "ja"
    if hangul / total > 0.30:
        return "ko"
    if han / total > 0.50:
        return "zh"
    if latin / total > 0.70:
        return "en"
    return ""


def _has_words(text: str) -> bool:
    """這一行有沒有需要翻譯的內容（純符號、純數字的行原樣帶過）。"""
    return any(unicodedata.category(ch)[0] == "L" for ch in text)


def _clean(text: str) -> str:
    return " ".join((text or "").split())


# ══════════════════════════════════════════════════════════════════════
class TranslatorServer:
    """管理一個 `crispasr.exe --server --chat-model` 子程序。

    用法：
        with TranslatorServer(exe, model, stub, gpu_backend="vulkan") as srv:
            out = srv.translate("Hello.", "zh")
    """

    def __init__(self, exe: Path, model: Path, stub_model: Path,
                 gpu_backend: str | None = "vulkan", device_id: int = 0,
                 start_timeout: float = 180.0):
        self.exe = Path(exe)
        self.model = Path(model)
        self.stub_model = Path(stub_model)
        self.gpu_backend = gpu_backend or None      # None＝純 CPU
        self.device_id = int(device_id)
        self.start_timeout = start_timeout
        self.port = 0
        self._proc: subprocess.Popen | None = None
        self._log_path: Path | None = None
        self._log_fh = None

    # ── 啟動／關閉 ──────────────────────────────────────────
    def _command(self) -> list[str]:
        cmd = [str(self.exe), "--server", "--host", "127.0.0.1",
               "--port", str(self.port),
               # server 一定要有辨識模型；給最小的佔位模型，並關掉它會順手
               # 自動下載的標點模型與啟動暖機。
               "--backend", "whisper", "-m", str(self.stub_model),
               "--no-punctuation", "--no-warmup",
               "--chat-model", str(self.model), "--chat-ctx", "2048"]
        if self.gpu_backend:
            cmd += ["--gpu-backend", self.gpu_backend, "-dev", str(self.device_id)]
        else:
            # CPU 版核心沒有 GPU DLL；聊天模型也要明講不放 GPU
            cmd += ["-ng", "--chat-gpu-layers", "0"]
        return cmd

    def start(self):
        for p, what in ((self.exe, "crispasr.exe"), (self.model, "翻譯模型"),
                        (self.stub_model, "佔位模型")):
            if not p.exists():
                raise RuntimeError(f"找不到{what}：{p}")
        with socket.socket() as s:                   # 向系統要一個空的埠
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        fd, name = tempfile.mkstemp(prefix="qwenasr-translate-", suffix=".log")
        self._log_path = Path(name)
        self._log_fh = os.fdopen(fd, "wb")
        self._proc = subprocess.Popen(
            self._command(), stdout=self._log_fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, cwd=str(self.exe.parent),
            creationflags=_CREATE_NO_WINDOW, startupinfo=_STARTUP_INFO)
        deadline = time.time() + self.start_timeout
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("翻譯伺服器啟動失敗：" + self._log_tail())
            try:
                with _OPENER.open(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                    if r.status == 200:
                        return self
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.3)
        self.stop()
        raise RuntimeError("翻譯伺服器啟動逾時")

    def stop(self):
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except Exception:
                pass
            self._log_fh = None
        if self._log_path is not None:
            try:
                self._log_path.unlink(missing_ok=True)
            except Exception:
                pass
            self._log_path = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def _log_tail(self, n: int = 3) -> str:
        """子程序輸出的最後幾行（只挑看起來像錯誤的），給錯誤訊息用。"""
        try:
            if self._log_fh is not None:
                self._log_fh.flush()
            lines = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return "（沒有輸出）"
        bad = [ln.strip() for ln in lines
               if any(k in ln.lower() for k in ("error", "fail", "cannot", "unable"))]
        return " / ".join((bad or [ln.strip() for ln in lines if ln.strip()])[-n:])[:300] \
            or "（沒有輸出）"

    # ── 翻譯 ────────────────────────────────────────────────
    def translate(self, text: str, target: str, timeout: float = 120.0) -> str:
        """翻譯一行；回傳譯文（單行、已去頭尾空白）。"""
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("翻譯伺服器已停止：" + self._log_tail())
        text = _clean(text)
        body = {
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": TARGETS[target][0] + "\n" + text},
            ],
            "temperature": 0,
            # 譯文長度跟原文成比例；設上限是為了擋模型偶爾停不下來的情況
            "max_tokens": min(512, 64 + 3 * len(text)),
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        with _OPENER.open(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return _clean(data["choices"][0]["message"].get("content") or "")


# ══════════════════════════════════════════════════════════════════════
_CC = None


def _to_taiwan(text: str) -> str:
    """GGUF 路徑沒有 beam search，偶爾會夾簡體字 → 一律再過 OpenCC s2twp。"""
    global _CC
    if _CC is None:
        try:
            import opencc
            _CC = opencc.OpenCC("s2twp")
        except Exception:
            _CC = False
    return _CC.convert(text) if _CC else text


# ── 句子與字幕行不對齊：以句子為單位翻譯，再掛回字幕行 ───────────────
#   字幕行是為了「一行不要太長」切的，不是照句子切的。逐行翻譯時每一行常是
#   沒頭沒尾的半句話，實測會翻壞：
#     英文  "Any particular reason you put it" → 「任何特定的原因您把它放進去」
#           "where the alarm clock used to be?" → 「那臺鬧鐘原來在哪裡？」
#     日文  "の絵。子犬のぬい" / "ぐるみ。ネム先輩にもらった綺麗"
#           （句子在行中間結束、下一句從行中間開始，「ぬいぐるみ」還被切成兩行）
#   做法：把相鄰的行接起來，在句末標點處切成句子，一句翻一次。
#   譯文**不切回各行**：試過依原文長度比例切，中文會切在詞中間（「看起｜來」
#   「小｜心」），沒有斷詞器切不好。改成譯文掛在「句子開始的那一行」，同一句
#   的後續行標成接續行（trCont）；寫譯文字幕時接續行併進同一格，對照顯示時
#   整句一起出現。
#   整份字幕幾乎沒有句末標點時（中文字幕行本來就不帶標點、行就是子句）不做
#   跨行合併，退回一行翻一次。
_CJK_END = "。！？…"
_ASCII_END = ".?!"
_CLOSERS = "\"'”’」』)）]"
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "jr", "sr", "prof", "no", "etc"}
MAX_GROUP_LINES = 4          # 一句最多跨幾行（沒有標點的長段落不能無限接下去）
MAX_GROUP_GAP = 1.0          # 兩行間隔超過這個秒數就當作不同句
MIN_PUNCT_RATIO = 0.15       # 有句末標點的行少於這個比例 → 視為「沒有標點」的字幕


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF
            or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF)


def _last_sentence_end(text: str) -> int:
    """回傳行內最後一個句末位置（切點＝該位置之後）；沒有回 -1。

    英文句點只在「後面是空白或行尾」時算句末，且排除 Mr.／Dr. 這類縮寫與
    小數點（3.5 的點後面不是空白）。
    """
    i = len(text) - 1
    while i >= 0:
        ch = text[i]
        if ch in _CJK_END or (ch in _ASCII_END and (i + 1 == len(text)
                                                     or text[i + 1].isspace()
                                                     or text[i + 1] in _CLOSERS)):
            if ch == ".":
                word = text[:i].rsplit(None, 1)[-1].lower() if text[:i].split() else ""
                if word in _ABBREV:
                    i -= 1
                    continue
            j = i + 1
            while j < len(text) and text[j] in _CLOSERS:
                j += 1
            return j
        i -= 1
    return -1


def _join(a: str, b: str) -> str:
    """接兩段文字：兩邊都不是中日文字才補空白。"""
    if not a:
        return b
    if not b:
        return a
    return a + ("" if (_is_cjk(a[-1]) or _is_cjk(b[0])) else " ") + b


def _line_text(seg: dict) -> str:
    """字幕行的文字；行尾標點被分行器拿掉的，從字級項目的 ``punct`` 補回來。

    中文／日文的字幕行不帶標點（分行器在標點處換行後就把標點丟了），但分行器
    會把「這一行後面原本是什麼標點」記在最後一個字級項目上。補回來之後，這裡
    才分得出一行是句子結束（。）還是只是子句停頓（、），也才能把同一句的幾行
    接起來整句翻；模型拿到帶標點的原文，譯得也比較好。
    """
    text = _clean(seg.get("text", ""))
    words = seg.get("words") or []
    punct = words[-1].get("punct", "") if words and isinstance(words[-1], dict) else ""
    if text and punct and text[-1] not in punct:
        text += punct
    return text


def plan_sentences(segs: list[dict]) -> list[tuple[int, int, str]]:
    """把字幕行重組成句子，回傳 [(開始行, 結束行, 句子文字), ...]（依出現順序）。"""
    texts = [_line_text(seg) for seg in segs]
    worded = [t for t in texts if _has_words(t)]
    punctuated = bool(worded) and (
        sum(1 for t in worded if _last_sentence_end(t) >= 0) / len(worded) >= MIN_PUNCT_RATIO)

    out: list[tuple[int, int, str]] = []
    buf, start = "", 0

    def _emit(end: int):
        nonlocal buf
        if buf:
            out.append((start, end, buf))
        buf = ""

    for i, text in enumerate(texts):
        if not text or not _has_words(text):
            _emit(i - 1)
            if text:
                out.append((i, i, text))          # 純符號／數字行：獨立一句，原樣帶過
            continue
        cut = _last_sentence_end(text) if punctuated else len(text)
        head, tail = (text[:cut].strip(), text[cut:].strip()) if cut >= 0 else ("", text)
        if head:
            if not buf:
                start = i
            buf = _join(buf, head)
            _emit(i)
        if tail:
            if not buf:
                start = i
            buf = _join(buf, tail)
        if buf:                                   # 這一行結束時句子還沒完 → 能不能接下一行
            nxt = segs[i + 1] if i + 1 < len(segs) else None
            joinable = (nxt is not None
                        and i - start + 1 < MAX_GROUP_LINES
                        and nxt.get("speaker") == segs[i].get("speaker")
                        and float(nxt.get("start", 0)) - float(segs[i].get("end", 0)) <= MAX_GROUP_GAP
                        and _has_words(texts[i + 1]))
            if not joinable:
                _emit(i)
    _emit(len(texts) - 1)
    return out


def translate_segments(server, segs: list[dict], target: str,
                       progress_cb=None, cancel=None) -> tuple[list[str], list[bool]]:
    """翻譯整份字幕，回傳 (譯文, 是否為接續行)，兩者都與 segs 等長。

    以句子為單位翻譯（見 plan_sentences）：譯文掛在句子開始的那一行；同一句
    的後續行拿到同一筆譯文並標成接續行。
    • 沒有文字內容的行（純符號、數字）原樣帶過。
    • 相同的原文只翻一次（字幕裡「Yeah.」「謝謝」這類重複很多）。
    • 某一句翻譯失敗或回空字串 → 那幾行譯文留空，不讓整份翻譯失敗。
    • progress_cb(done, total)；cancel() 回 True 就中止並拋 RuntimeError。
    """
    n = len(segs)
    cache: dict[str, str] = {}

    def _one(src: str) -> str:
        if not src or not _has_words(src):
            return src
        if src not in cache:
            try:
                tr = server.translate(src, target)
            except (urllib.error.URLError, OSError, KeyError, ValueError):
                tr = ""
            cache[src] = _to_taiwan(tr) if (tr and target == "zh") else tr
        return cache[src]

    starts: list[list[str]] = [[] for _ in range(n)]     # 每行：從這行開始的句子的譯文
    cover: list[int] = [-1] * n                          # 每行：涵蓋它的那句從哪行開始
    for a, b, text in plan_sentences(segs):
        if cancel and cancel():
            raise RuntimeError("已取消翻譯")
        tr = _one(text)
        if tr:
            starts[a].append(tr)
            for i in range(a + 1, b + 1):
                if cover[i] < 0:
                    cover[i] = a
        if progress_cb:
            progress_cb(b + 1, n)

    sep = " " if target == "en" else ""
    out = [sep.join(x) for x in starts]
    cont = [False] * n
    for i in range(n):
        if not out[i] and cover[i] >= 0:
            out[i] = out[cover[i]]
            cont[i] = True
    return out, cont


def merge_translated(segs: list[dict]) -> list[dict]:
    """segments → 譯文字幕的格子：接續行併進前一格，沒譯出的行保留原文。

    回傳的每格是 {start, end, text, speaker}（text＝譯文）。
    """
    out: list[dict] = []
    for seg in segs:
        if seg.get("trCont") and out:
            out[-1]["end"] = seg["end"]
            continue
        out.append({"start": seg["start"], "end": seg["end"],
                    "text": seg.get("translation") or seg["text"],
                    "speaker": seg.get("speaker")})
    return out
