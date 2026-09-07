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
        sep = " " if (_has_latin(a) or _has_latin(b)) else ""
        return f"{a}{sep}{b}"

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
        sep = " " if (_has_latin(a) or _has_latin(b)) else ""
        return f"{a}{sep}{b}"

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
        if any(_is_latin_word(w) for w in seg_words):
            text = " ".join(seg_words)
        else:
            text = "".join(seg_words)
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
