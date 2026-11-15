"""subtitle_lines.py — 字級時間軸 → 字幕行（後端無關，全引擎共用）

把「字級 (word, start_sec, end_sec) + ASR 原文」轉成字幕行的邏輯集中於此，
讓 OpenVINO / chatllm / CrispASR(Whisper) 三種引擎產出**一致**的字幕斷句與
時間軸（標點切行 + MAX_CHARS/MAX_WORDS 保護 + 孤兒行合併）。

公開符號：
    MAX_CHARS, _ZH_CLAUSE_END, _EN_SENT_END   斷句常數
    _srt_ts(s)                                秒 → SRT 時間戳
    _merge_orphan_lines(lines)                合併過短孤兒行
    _ts_chatllm_to_subtitle_lines(...)        字級 list → [(start,end,text,spk)]
"""
from __future__ import annotations

# ── 斷句常數（與舊 app.py 行為一致）──────────────────────────────────
MAX_CHARS      = 20
_ZH_CLAUSE_END = frozenset('，。？！；：…—、·')
_EN_SENT_END   = frozenset('.,!?;')
# 相鄰兩個字之間停頓超過這個秒數就換行。字幕不該橫跨一段沒人講話的時間：
# 沒有這條時，靜音前最後一個字與靜音後第一個字會落在同一行，那一行的顯示時間
# 就等於整段靜音（實測 120 秒靜音 → 單行 122 秒）。
MAX_LINE_GAP   = 2.0


def _strip_pua(s: str) -> str:
    """濾除 Unicode 私用區字元（BMP U+E000–U+F8FF 與兩個補充平面）。

    合法的中文／日文／英文字幕不會用到私用區；出現在這裡的一律是下游元件把
    「無法映射的內部 ID」渲染成字元的產物，屬於亂碼。濾掉比留著好——留著會
    一路污染 SRT、JSON 與外部 Agent。
    """
    if not s:
        return s
    return "".join(c for c in s
                   if not (0xE000 <= ord(c) <= 0xF8FF
                           or 0xF0000 <= ord(c) <= 0xFFFFD
                           or 0x100000 <= ord(c) <= 0x10FFFD))


def _has_latin(t: str) -> bool:
    """字串是否含半形英文字母。"""
    return any(c.isascii() and c.isalpha() for c in t)


def _word_sep(a: str, b: str) -> str:
    """兩個相鄰詞之間該不該放空白（逐對判斷）。

    只有「前詞結尾與後詞開頭都是半形非空白字元，且其中一邊含英文字母」才放
    空白：`hello`+`world`、`Hello,`+`world`、`iPhone`+`15` 需要；中英相鄰
    （`打高`+`golf`、`golf，`+`跟`）與純數字相鄰（`1`+`0`）都不需要。

    舊規則是「整行只要含任一英文字就全行用空白串接」，結果一個英文字會讓同一
    行所有中文字都被空白隔開（「golf， 跟 那 個 游 泳 池」）。
    """
    if not a or not b:
        return ""
    x, y = a[-1], b[0]
    if not (x.isascii() and y.isascii()) or x.isspace() or y.isspace():
        return ""
    return " " if (_has_latin(a) or _has_latin(b)) else ""


def _join_words(words) -> str:
    """把一行的詞串起來，空白只放在 `_word_sep` 認為需要的位置。"""
    out = ""
    for w in words:
        out = f"{out}{_word_sep(out, w)}{w}" if out else w
    return out


def _srt_ts(s: float) -> str:
    ms = int(round(s * 1000))
    hh = ms // 3_600_000; ms %= 3_600_000
    mm = ms // 60_000;    ms %= 60_000
    ss = ms // 1_000;     ms %= 1_000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


# ── 全域輸出格式（"srt" | "txt"）────────────────────────────────────────
# 由 app 啟動 / 設定變更時同步（app._on_output_format_change → 改寫此值）。
# write_transcript() 在 out_format=None 時讀此值，使「批次 / 單檔 / 錄製」
# 三條路徑全域一致；端點需內部解析 SRT，固定以 out_format="srt" 覆寫。
OUTPUT_FORMAT = "srt"


def lines_to_txt(lines: list[tuple[float, float, str, str | None]]) -> str:
    """字幕行 → 純文字（沿用端點既有慣例，與 api_server._parse_srt 後處理一致）。

    • 無說話者：整段文字相連成一行（中文不插空白，重建連續逐字稿）。
    • 有說話者：每段一行，保留「說話者N：」前綴，便於分辨發言者。
    """
    has_spk = any(spk for (_s, _e, _t, spk) in lines)
    if has_spk:
        return "\n".join(
            (f"{spk}：{t}" if spk else t) for (_s, _e, t, spk) in lines
        )
    return "".join(t for (_s, _e, t, _spk) in lines)


def write_transcript(
    ref,
    lines: list[tuple[float, float, str, str | None]],
    out_format: str | None = None,
):
    """把字幕行依格式寫成 .srt 或 .txt，回傳實際輸出路徑（Path）。

    所有引擎（OpenVINO / chatllm / CrispASR）與錄製轉換共用此單一寫出點，
    確保全域輸出格式一致。

    參數：
        ref        : 決定輸出目錄與主檔名的參考路徑（通常為原始音檔）。
        lines      : [(start_sec, end_sec, text, speaker|None), ...]
        out_format : "srt" | "txt"；None 時採用全域 OUTPUT_FORMAT。
                     端點固定傳 "srt"（內部需解析時間軸），不受全域影響。
    """
    from pathlib import Path
    fmt = (out_format or OUTPUT_FORMAT or "srt").lower()
    ref = Path(ref)
    if fmt == "txt":
        out = ref.parent / (ref.stem + ".txt")
        out.write_text(lines_to_txt(lines), encoding="utf-8")
        return out
    out = ref.parent / (ref.stem + ".srt")
    with open(out, "w", encoding="utf-8") as f:
        for idx, (s, e, line, spk) in enumerate(lines, 1):
            prefix = f"{spk}：" if spk else ""
            f.write(f"{idx}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{prefix}{line}\n\n")
    return out


def _merge_orphan_lines(
    lines: list[tuple[float, float, str, str | None]],
    min_chars: int = 1,
    max_gap: float = 0.8,
) -> list[tuple[float, float, str, str | None]]:
    """合併過短的孤立字幕行（如句尾「吧」單獨成行）到相鄰行。

    FA 斷句時 MAX_WORDS 與標點切行偶爾會疊加，在子句中間切一刀，把
    句尾語助詞（吧/啊/呢/了…）留成獨立一行。此處在輸出前把這類「孤兒行」
    併回相鄰行：優先併入前一行（時間連續、同說話者），首行孤兒則併入下一行。
    含拉丁詞時以空格 join，純中文直接相接。

    預設僅併「單字」孤兒：單字幾乎都是句尾語助詞，向後併入前一行最安全；
    兩字以上可能是句首短詞，向後併易誤接，故不處理。
    """
    if not lines:
        return lines

    def _has_latin(t: str) -> bool:
        return any(c.isascii() and c.isalpha() for c in t)

    def _vlen(t: str) -> int:
        return len(t.replace(" ", ""))

    def _is_orphan(t: str) -> bool:
        # 純中文且可見字數極少才視為孤兒；含拉丁詞（英文/數字詞）不併
        return (not _has_latin(t)) and 0 < _vlen(t) <= min_chars

    def _join(a: str, b: str) -> str:
        return f"{a}{_word_sep(a, b)}{b}"

    merged: list[tuple[float, float, str, str | None]] = []
    for (s, e, t, spk) in lines:
        if (_is_orphan(t) and merged
                and merged[-1][3] == spk
                and s - merged[-1][1] <= max_gap):
            ps, _pe, pt, pspk = merged[-1]
            merged[-1] = (ps, e, _join(pt, t), pspk)
        else:
            merged.append((s, e, t, spk))

    # 首行仍是孤兒（無前一行可併）→ 併入下一行
    if len(merged) >= 2 and _is_orphan(merged[0][2]):
        s0, _e0, t0, spk0 = merged[0]
        s1, e1, t1, spk1 = merged[1]
        if spk0 == spk1 and s1 - merged[0][1] <= max_gap:
            merged[1] = (s0, e1, _join(t0, t1), spk1)
            merged.pop(0)

    return merged


def _merge_orphan_lines_rich(
    lines: list[tuple],
    min_chars: int = 1,
    max_gap: float = 0.8,
) -> list[tuple]:
    """`_merge_orphan_lines` 的 5-tuple 版（卡拉OK字級用）。

    行為與 4-tuple 版完全相同（同樣的孤兒判定、間隔/說話者條件），差別只在
    併行時連同第 5 元素 ``words`` 字級清單一起串接，使字幕卡與卡拉OK逐字
    高亮的斷句保持一致。
    """
    if not lines:
        return lines

    def _has_latin(t: str) -> bool:
        return any(c.isascii() and c.isalpha() for c in t)

    def _vlen(t: str) -> int:
        return len(t.replace(" ", ""))

    def _is_orphan(t: str) -> bool:
        return (not _has_latin(t)) and 0 < _vlen(t) <= min_chars

    def _join(a: str, b: str) -> str:
        return f"{a}{_word_sep(a, b)}{b}"

    merged: list[tuple] = []
    for (s, e, t, spk, w) in lines:
        if (_is_orphan(t) and merged
                and merged[-1][3] == spk
                and s - merged[-1][1] <= max_gap):
            ps, _pe, pt, pspk, pw = merged[-1]
            merged[-1] = (ps, e, _join(pt, t), pspk, pw + w)
        else:
            merged.append((s, e, t, spk, w))

    if len(merged) >= 2 and _is_orphan(merged[0][2]):
        s0, _e0, t0, spk0, w0 = merged[0]
        s1, e1, t1, spk1, w1 = merged[1]
        if spk0 == spk1 and s1 - merged[0][1] <= max_gap:
            merged[1] = (s0, e1, _join(t0, t1), spk1, w0 + w1)
            merged.pop(0)

    return merged


# ══════════════════════════════════════════════════════════════════════
# 以空白分詞的語言（英文、韓文、歐語…）
# ══════════════════════════════════════════════════════════════════════
#   中文／日文的分行契約是「items＝單字、標點只留在 raw_text 當切點、輸出不帶
#   標點」。英文與韓文不適用：
#     • 引擎給的是整個詞，而且標點黏在詞尾（`you.`、`Hey,`、`안녕하세요.`），
#       分行器等不到獨立的標點，結果完全不照句子斷，只會每 8 個詞硬切一刀
#       （2026-11-16 用 13 集英文影集實測：一行「hate you. Good. Hey, hon.
#       Anything going on? No.」橫跨 6 秒，下一句的第一個詞卻掛在上一行行尾）。
#     • 輸出必須保留標點；`?` 與 `.` 拿掉後英文字幕很難讀。
#     • 長度要用字元數算（一行約 42 個字元是字幕慣例），不是詞數。
#   做法：先依「句尾標點／長停頓」切成語句，過長的語句挑最接近中間的逗號或詞
#   界對半分，最後把緊接著的極短語句（`Good.`、`Sorry.`）併回鄰行。
DEFAULT_SPACED_CHARS = 42   # 單行上限的預設值（字幕慣例）
# 單行上限（顯示寬度：英文字母 1 格、韓文／全形字 2 格）。可由設定頁調整
# （webview_backend._apply_line_chars 會改寫此值），分行時才讀取，改了立即生效。
MAX_SPACED_CHARS  = DEFAULT_SPACED_CHARS
MIN_SPACED_CHARS  = 12      # 短於此的語句會試著併入鄰行
FRAGMENT_CHARS    = 16      # 短於此且沒講完的句首殘片併入下一行（見步驟 ③）
SPACED_MERGE_GAP  = 0.6     # 併行時兩句之間允許的最大停頓（秒）
_SENT_END_CH   = frozenset('.?!…。？！')
_CLAUSE_END_CH = frozenset(',;:，；：、')
_CLOSER_CH     = frozenset('"\')]}»”’」』')
# 句點不代表句子結束的常見縮寫（小寫比對）
_ABBREV = frozenset({"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "jr.", "sr.",
                     "prof.", "no.", "mt.", "e.g.", "i.e.", "etc."})
# 不適合當行尾的英文虛詞：在這些詞後面換行，下一行會以它的受詞開頭，讀起來斷裂
_WEAK_TAIL = frozenset({"a", "an", "the", "to", "of", "and", "or", "but", "in", "on",
                        "at", "for", "with", "my", "your", "his", "her", "our",
                        "their", "i", "you", "we", "he", "she", "it", "they"})


def _is_cjk_char(c: str) -> bool:
    """漢字或日文假名（這些文字不以空白分詞）。韓文諺文不算。"""
    o = ord(c)
    return (0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF
            or 0x4E00 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF
            or 0x20000 <= o <= 0x2FA1F)


def is_spaced_script(ts_items) -> bool:
    """這批字級項目是不是「以空白分詞」的語言（英文、韓文…）。

    看內容而不是看使用者選的語言：語言常是自動偵測（None），而中英夾雜的中文
    內容仍應走中文契約。含漢字／假名的項目不到三成才算。
    """
    words = [w for (w, _s, _e) in ts_items if w and any(ch.isalnum() for ch in w)]
    if not words:
        return False
    cjk = sum(1 for w in words if any(_is_cjk_char(ch) for ch in w))
    return cjk / len(words) < 0.3


def _spaced_join(words) -> str:
    """以空白串接；只有兩側都是漢字／假名時不放空白。"""
    out = ""
    for w in words:
        if not out:
            out = w
        elif _is_cjk_char(out[-1]) and _is_cjk_char(w[0]):
            out += w
        else:
            out += " " + w
    return out


def _disp_len(text: str) -> int:
    """顯示寬度：全形／寬字元（韓文諺文、漢字、假名）算 2 格，其餘 1 格。

    行長上限要用寬度算而不是字元數——42 個英文字母與 42 個韓文字佔的畫面
    寬度差一倍。
    """
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in text)


def _tail_char(w: str) -> str:
    """詞尾去掉引號／括號後的最後一個字元（用來判斷句尾或逗號）。"""
    t = w.rstrip("".join(_CLOSER_CH))
    return t[-1] if t else ""


def _ends_sentence(w: str) -> bool:
    return _tail_char(w) in _SENT_END_CH and w.lower() not in _ABBREV


def reattach_punct(ts_items, raw_text: str):
    """把 raw_text 裡的標點接回詞尾 → 帶標點的字級項目；整體對不上時回傳 None。

    OpenVINO／chatllm 的對齊器給的是不含標點的詞，標點只在 ASR 原文裡；
    CrispASR 則本來就黏在詞尾（此時原樣通過）。做法是在原文裡依序找每個詞。

    個別的詞對不上時不放棄整份：往後找一小段重新對位，找不到就讓那個詞原樣
    通過。整份放棄的代價太大——CrispASR 一次給整個檔案，實測一集 22 分鐘的
    影集只因為一個 `'cause`（開頭的撇號被前一個詞當成收尾引號吃掉）就整集退回
    舊分行。超過兩成的詞都對不上才視為原文與項目根本不是同一份，回 None，
    由呼叫端退回既有的中文契約分行。
    """
    edge = _SENT_END_CH | _CLAUSE_END_CH | _CLOSER_CH
    items = list(ts_items)
    out, spaced_before, ri, n, missed = [], [], 0, len(raw_text), 0

    def _match(w: str, at: int):
        """從 at 開始比對 w（可跳過詞前的空白與標點）→ (結束位置, 詞前有無空白)。"""
        k, sp = at, False
        for pos, ch in enumerate(w):
            while k < n and raw_text[k] != ch and (raw_text[k].isspace()
                                                    or raw_text[k] in edge):
                if pos == 0 and raw_text[k].isspace():
                    sp = True
                k += 1                        # 跳過空白與詞前的標點（開引號等）
            if k >= n or raw_text[k] != ch:
                return None
            k += 1
        return k, sp

    for idx, (w, s, e) in enumerate(items):
        hit = _match(w, ri)
        if hit is None:
            j = raw_text.find(w, ri, ri + 80)         # 重新對位：往後找一小段
            hit = (j + len(w), True) if j >= 0 else None
        if hit is None:
            missed += 1
            out.append((w, s, e))
            spaced_before.append(True)
            continue
        ri, saw_space = hit
        # 詞尾的標點。若下一個詞本身以標點開頭（`'cause`、`"Hello`），那幾個
        # 字元是它的，不能吃掉。
        run = 0
        while ri + run < n and raw_text[ri + run] in edge:
            run += 1
        if run and idx + 1 < len(items):
            nxt = items[idx + 1][0]
            lead = len(nxt) - len(nxt.lstrip("".join(edge)))
            if lead and raw_text[ri:ri + run].endswith(nxt[:lead]):
                run -= lead
        out.append((w + raw_text[ri:ri + run], s, e))
        spaced_before.append(saw_space)
        ri += run

    if missed > 0.2 * max(len(items), 1):
        return None

    # 項目比詞還細的情況（CrispASR 對齊器給韓文的是「一個音節一項」，詞界用
    # 原文裡的空白標示：`다`+`리`+` 및`）→ 把中間沒有空白的項目併回同一個詞。
    # 原文完全沒有空白時（CrispASR 的英文：一項就是一個詞）不能這樣併，否則
    # 整段會黏成一個詞，所以要先確認原文確實用空白在標詞界。
    if len(out) > 1 and sum(spaced_before[1:]) >= 0.1 * (len(out) - 1):
        # 對齊器標的空白並不完整：句首、數字與英數前面的空白會漏掉（實測
        # `이다.`+`델`、`간격은`+`15m`、`대`+`6`）。補三條不靠空白的詞界判斷：
        # 前一項以標點結尾、兩項之間有停頓、諺文後面接半形英數。反方向
        # （`15m`+`이며`、`2011`+`년`）是助詞黏在數字後面，不能拆。
        def _boundary(prev, item, sp) -> bool:
            if sp or _tail_char(prev[0]) in (_SENT_END_CH | _CLAUSE_END_CH):
                return True
            if item[1] - prev[2] > 0.4:
                return True
            a, b = prev[0][-1], item[0][0]
            return (not a.isascii()) and b.isascii() and b.isalnum()

        words = [out[0]]
        for item, sp in zip(out[1:], spaced_before[1:]):
            if _boundary(words[-1], item, sp):
                words.append(item)
            else:
                pw, ps, _pe = words[-1]
                words[-1] = (pw + item[0], ps, item[2])
        out = words
    return out


def unstretch_words(ts_items, max_dur: float = 1.5):
    """收回被對齊器拉長的詞（保留終點）。

    CTC 對齊器把詞前的停頓／笑聲／配樂算進該詞的開頭：實測「be?」結束在
    33.67、下一句的「I」被標成 33.67–37.03，真正開口在 37 秒附近。不處理的話
    這一行字幕會提早 3 秒出現，而且長停頓換行（MAX_LINE_GAP）也偵測不到停頓。
    一個詞不可能唸超過 `max_dur` 秒，超過的把起點往後收到合理長度。
    """
    out = []
    for (w, s, e) in ts_items:
        est = 0.25 + 0.07 * len(w)            # 依字元數估的正常唸讀長度
        if e - s > max(max_dur, 2 * est):
            s = e - est
        out.append((w, s, e))
    return out


def _split_long(idx: list, items) -> list:
    """把超過單行上限的語句對半分（遞迴），回傳 [索引清單, …]。"""
    def _len(part):
        return _disp_len(_spaced_join([items[k][0] for k in part]))

    if len(idx) < 2 or _len(idx) <= MAX_SPACED_CHARS:
        return [idx]
    best, best_score = None, None
    for cut in range(1, len(idx)):
        left, right = idx[:cut], idx[cut:]
        ll, rl = _len(left), _len(right)
        score = abs(ll - rl)
        last = items[left[-1]][0]
        if _tail_char(last) in _CLAUSE_END_CH:
            score -= 10                       # 逗號後換行最自然
        elif last.lower().strip("\"'") in _WEAK_TAIL:
            score += 8                        # 別讓 a／the／to 掛在行尾
        if best_score is None or score < best_score:
            best, best_score = cut, score
    return _split_long(idx[:best], items) + _split_long(idx[best:], items)


def spaced_lines_from_text(text: str, g0: float, g1: float, spk=None):
    """沒有字級時間軸時的版本：整段文字＋段落起訖 → 字幕行；不是空白分詞語言回 None。

    給 OpenVINO／chatllm「關閉對齊」的比例估算路徑用。那條路徑原本逐字元掃描：
    撇號不算英文字母，`it's` 會被拆成 `it' s`；每個標點都切行且丟掉標點，英文
    變成「Sneaking up behind／your sister／pulling her hair」一行兩三個詞。
    這裡改成依字元數把時間攤給每個詞，再走同一套英文／韓文分行。
    回傳 5-tuple（words 為空清單：時間是估的，不拿來做逐字高亮）。
    """
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    toks = text.split()
    if not toks or not is_spaced_script([(w, 0.0, 0.0) for w in toks]):
        return None
    total = sum(len(w) for w in toks)
    dur = max(g1 - g0, 0.0)
    items, pos = [], 0
    for w in toks:
        items.append((w, g0 + dur * pos / total, g0 + dur * (pos + len(w)) / total))
        pos += len(w)
    lines = spaced_subtitle_lines(items, 0.0, spk, with_words=True, unstretch=False)
    return [(s, e, t, sp, []) for (s, e, t, sp, _w) in lines]


def spaced_subtitle_lines(ts_items, chunk_offset: float = 0.0, spk=None,
                          with_words: bool = False, unstretch: bool = True):
    """帶標點的詞級項目 → 字幕行（英文／韓文等以空白分詞的語言）。

    ts_items：[(word, start, end), …]，標點黏在詞尾。回傳格式與
    `_ts_chatllm_to_subtitle_lines` 相同（4-tuple；with_words=True 為 5-tuple）。
    unstretch=False：時間是估算而非對齊器給的（見 spaced_lines_from_text），
    不做「拉長的詞」修正。
    """
    items = [(_strip_pua(w).strip(), s, e) for (w, s, e) in ts_items]
    # 獨立成項的標點（少見）接回前一個詞
    merged: list = []
    for (w, s, e) in items:
        if not w:
            continue
        if merged and not any(ch.isalnum() for ch in w):
            pw, ps, _pe = merged[-1]
            merged[-1] = (pw + w, ps, e)
        else:
            merged.append((w, s, e))
    items = unstretch_words(merged) if unstretch else merged
    if not items:
        return []

    # ① 切成語句：句尾標點或長停頓
    units, cur = [], []
    for k, (w, s, _e) in enumerate(items):
        if cur and (s - items[cur[-1]][2] > MAX_LINE_GAP
                    or _ends_sentence(items[cur[-1]][0])):
            units.append(cur)
            cur = []
        cur.append(k)
    if cur:
        units.append(cur)

    # ② 過長的語句對半分
    parts = [p for u in units for p in _split_long(u, items)]

    def _text(part):
        return _spaced_join([items[k][0] for k in part])

    # ③ 句子開頭被長停頓切下來的殘片（`Where` ‖ `are the kids?`、`We` ‖ `have no
    #    juice.`）併入「下一行」。這種殘片幾乎都是對齊器把詞放錯位置造成的：話
    #    還沒講完就停兩秒以上並不常見。它的時間不可信，改貼在下一行開頭之前。
    k = 0
    while k < len(parts) - 1:
        p, nxt = parts[k], parts[k + 1]
        if (_disp_len(_text(p)) < FRAGMENT_CHARS and not _ends_sentence(items[p[-1]][0])
                and _disp_len(_text(p)) + 1 + _disp_len(_text(nxt)) <= MAX_SPACED_CHARS):
            t = items[nxt[0]][1]
            for j in reversed(p):
                w = items[j][0]
                dur = min(0.25 + 0.07 * len(w), max(items[j][2] - items[j][1], 0.1))
                items[j] = (w, max(0.0, t - dur), t)
                t = items[j][1]
            parts[k:k + 2] = [p + nxt]
        else:
            k += 1

    # ④ 極短的完整語句併入前一行（`Good.` `Sorry.`）：要緊接著、併完不超長
    lines: list = []
    for p in parts:
        if lines:
            prev = lines[-1]
            gap = items[p[0]][1] - items[prev[-1]][2]
            short = (min(_disp_len(_text(prev)), _disp_len(_text(p))) < MIN_SPACED_CHARS
                     and _ends_sentence(items[p[-1]][0]))
            if (short and gap <= SPACED_MERGE_GAP
                    and _disp_len(_text(prev)) + 1 + _disp_len(_text(p)) <= MAX_SPACED_CHARS):
                lines[-1] = prev + p
                continue
        lines.append(p)

    result = []
    for p in lines:
        start = chunk_offset + items[p[0]][1]
        end = chunk_offset + items[p[-1]][2]
        if result and start < result[-1][1]:
            start = result[-1][1]             # 不與前一行重疊（殘片併行後偶爾會交疊）
        if end <= start:
            end = start + 0.3                 # 對齊器偶爾給零長度的詞
        words = [{"start": chunk_offset + items[k][1],
                  "end":   chunk_offset + items[k][2],
                  "text":  items[k][0]} for k in p]
        result.append((start, end, _text(p), spk, words))
    if with_words:
        return result
    return [(s, e, t, sp) for (s, e, t, sp, _w) in result]


def _ts_chatllm_to_subtitle_lines(
    ts_items,
    raw_text: str,
    chunk_offset: float,
    spk: str | None,
    cc,
    simplified: bool,
    break_on_space: bool = False,
    with_words: bool = False,
):
    """字級 (word, start_sec, end_sec) + ASR 原文 → 字幕行。

    標點切行 + MAX_CHARS/MAX_WORDS 保護；word_list 直接取自字級時間軸，
    與時間 1:1 對應，後端無關（chatllm FA / Whisper 字級皆適用）。

    參數：
        ts_items: list[tuple[str, float, float]]  → (word, start_sec, end_sec)
        break_on_space: True 時把 raw_text 的「空白」也當切點。
            用於 Whisper（無標點，但以空白標記語句邊界）→ 等同 Qwen 在標點切，
            逼近 Qwen 斷句品質。Qwen/chatllm 路徑維持 False（空白僅分隔拉丁詞）。
        with_words: True 時每行多帶一個 ``words`` 字級清單（卡拉OK逐字高亮用），
            回傳 5-tuple ``(start, end, text, spk, words)``；
            ``words = [{"start": 秒, "end": 秒, "text": 顯示字}, ...]``。
            預設 False → 維持既有 4-tuple 行為（SRT/批次/端點完全不受影響）。

    回傳：
        with_words=False → list[(start, end, text, spk)]
        with_words=True  → list[(start, end, text, spk, words)]
    """
    _all_punct = _ZH_CLAUSE_END | _EN_SENT_END
    MAX_WORDS    = 8
    MAX_ZH_CHARS = MAX_CHARS
    # 內部一律以 5-tuple (start, end, text, spk, words) 累積，回傳前依 with_words 決定保留與否
    result: list[tuple] = []

    if not ts_items or not raw_text.strip():
        return result

    # 英文／韓文等以空白分詞的語言走專用分行（見 spaced_subtitle_lines）。
    # Whisper 的 break_on_space 契約另有用途（空白＝語句邊界），不在此列。
    if not break_on_space and is_spaced_script(ts_items):
        spaced = reattach_punct(ts_items, raw_text)
        if spaced is not None:
            return spaced_subtitle_lines(spaced, chunk_offset, spk, with_words)

    word_list = [w for (w, _s, _e) in ts_items]
    n = len(ts_items)

    # 私用區（PUA）字元防線：這些不是任何模型的合法輸出，而是下游元件把「無法
    # 映射的類別 ID」直接當成字元渲染的結果（實例：crispasr 的 qwen3 CTC 對齊器
    # 遇到其類別表外的繁體字，會輸出 U+E000+class_id）。它們對使用者是亂碼，
    # 也會污染下游（SRT／JSON／Agent）。在唯一的文字出口統一濾掉，任何引擎都受惠。

    # 繁化：text（整行）沿用既有「整行轉換」確保 SRT 輸出零變化；
    # words[].text 走「逐字轉換」（卡拉OK高亮單位）——兩者在極少數 s2twp
    # 詞組轉換情境可能有細微差異，但卡拉OK檢視只讀 words，不影響字幕卡/SRT。
    def _conv(s: str) -> str:
        s = _strip_pua(s)
        return cc.convert(s) if (not simplified and cc is not None) else s

    seg_idx:   list[int] = []   # 當前行的 ts_items 索引
    seg_words: list[str] = []   # 當前行的原始 word
    ri = 0                      # raw_text 掃描位置

    def _is_latin_word(w: str) -> bool:
        return any(c.isascii() and c.isalpha() for c in w)

    def _emit():
        nonlocal seg_idx, seg_words
        if not seg_idx:
            seg_idx = []; seg_words = []
            return
        start = chunk_offset + ts_items[seg_idx[0]][1]
        end   = chunk_offset + ts_items[seg_idx[-1]][2]
        text = _join_words(seg_words)
        text = _conv(text)
        # 字級清單：每個 ts_item → 一個高亮單位（中文＝單字、拉丁＝整詞）
        words = [
            {"start": chunk_offset + ts_items[k][1],
             "end":   chunk_offset + ts_items[k][2],
             "text":  _conv(word_list[k])}
            for k in seg_idx
        ]
        if end > start and text.strip():
            result.append((start, end, text.strip(), spk, words))
        seg_idx = []; seg_words = []

    def _over_limit() -> bool:
        if any(_is_latin_word(w) for w in seg_words):
            return len(seg_words) > MAX_WORDS
        return sum(len(w) for w in seg_words) > MAX_ZH_CHARS

    for wi in range(n):
        word = word_list[wi]

        # 在 raw_text 中前進到 word 位置；遇到標點（或 whisper 空白）→ 先切行
        hit_punct = False
        while ri < len(raw_text):
            c = raw_text[ri]
            if c in _all_punct:
                hit_punct = True; ri += 1; continue
            if c == " ":
                if break_on_space:
                    hit_punct = True   # whisper：空白＝語句邊界，視同切點
                ri += 1; continue
            break

        if hit_punct:
            _emit()
        elif seg_idx and ts_items[wi][1] - ts_items[seg_idx[-1]][2] > MAX_LINE_GAP:
            _emit()            # 長停頓：即使沒有標點也換行（見 MAX_LINE_GAP）

        seg_idx.append(wi)
        seg_words.append(word)

        # 跳過 word 在 raw_text 中佔用的字元（依長度計數，忽略標點/空格）
        consumed = 0
        word_len = len(word)
        while ri < len(raw_text) and consumed < word_len:
            c = raw_text[ri]
            if c in _all_punct or c == " ":
                ri += 1; continue
            ri += 1; consumed += 1

        if _over_limit():
            _emit()

    _emit()
    merged = _merge_orphan_lines_rich(result)
    if with_words:
        return merged
    return [(s, e, t, spk) for (s, e, t, spk, _w) in merged]
