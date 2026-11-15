"""日文字幕：不做簡繁轉換、不切在詞中間；以及「項目帶標點」造成的切點偏移。

背景（2026-11-15 實測）：
  • CrispASR 0.8.54 的對齊器把標點黏在前一個字後面（`中、`、`呢。`）。共用分行
    器原本照 len() 推進原文指標，每遇到一個標點就多走一格，切點越切越偏：
    中文「主角小｜慧，她才剛變成天使就被丟到一｜個嗯。」、
    日文「の絵。子犬のぬい｜ぐるみ。ネム先輩にもらった綺麗」。
  • 語言設「自動」時日文會被 OpenCC 轉成繁體字形（学校→學校、本当→本當）。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import crisp_engine as ce      # noqa: E402
import subtitle_lines as sl    # noqa: E402
import translator as tr        # noqa: E402


class FakeCC:
    """假的簡→繁轉換器：只認得幾個字，足以看出「有沒有被轉」。"""
    TABLE = str.maketrans({"学": "學", "当": "當", "会": "會", "们": "們", "刚": "剛"})

    def convert(self, s):
        return s.translate(self.TABLE)


def _items(text, step=0.2, gaps=None):
    """每個字一個項目、等時排列；gaps={索引: 秒} 表示該字之前多停頓幾秒。"""
    out, t = [], 0.0
    for i, ch in enumerate(text):
        t += (gaps or {}).get(i, 0.0)
        out.append((ch, t, t + step))
        t += step
    return out


def _glued(raw):
    """模擬 CrispASR 0.8.54：標點黏在前一個字後面成為同一個項目。"""
    out, t = [], 0.0
    for ch in raw:
        if ch in sl._ZH_CLAUSE_END and out:
            w, s, e = out[-1]
            out[-1] = (w + ch, s, e)
        else:
            out.append((ch, t, t + 0.2))
            t += 0.2
    return out


def _lines(items, raw, cc=None, **kw):
    return sl._ts_chatllm_to_subtitle_lines(items, raw, 0.0, None, cc, False, **kw)


# ══════════════════════════ 語言判斷 ══════════════════════════

def test_looks_japanese_by_kana():
    assert sl.looks_japanese("本当のタイムマシンはオルゴール")
    assert sl.looks_japanese("はい")
    assert not sl.looks_japanese("今天我們來深入探索一本相當有趣的作品")
    assert not sl.looks_japanese("這是我の最愛的一首歌曲")          # 中文夾一個「の」
    assert not sl.looks_japanese("Hello world")
    assert not sl.looks_japanese("")


def test_to_traditional_skips_japanese_and_simplified_mode():
    cc = FakeCC()
    assert sl.to_traditional(cc, "我们刚到") == "我們剛到"
    assert sl.to_traditional(cc, "本当に学校へ行く") == "本当に学校へ行く"
    assert sl.to_traditional(cc, "我们刚到", simplified=True) == "我们刚到"
    assert sl.to_traditional(None, "我们刚到") == "我们刚到"


# ══════════════════════════ 簡繁轉換 ══════════════════════════

def test_japanese_is_not_converted_when_language_is_auto():
    raw = "本当に学校へ行く。"
    out = _lines(_items("本当に学校へ行く"), raw, cc=FakeCC(), with_words=True)
    assert [t for (_s, _e, t, _sp, _w) in out] == ["本当に学校へ行く"]
    assert "".join(w["text"] for w in out[0][4]) == "本当に学校へ行く"   # 逐字高亮也不轉


def test_a_single_kana_is_not_enough_to_call_it_japanese():
    """取捨：只有一個假名的片段當中文處理（中文夾「の」比純漢字的日文短句常見）。"""
    out = _lines(_items("我们刚到の学校"), "我们刚到の学校", cc=FakeCC())
    assert [t for (_s, _e, t, _sp) in out] == ["我們剛到の學校"]


def test_chinese_is_still_converted():
    out = _lines(_items("我们刚到学校"), "我们刚到学校。", cc=FakeCC())
    assert [t for (_s, _e, t, _sp) in out] == ["我們剛到學校"]


# ══════════════════════════ 項目帶標點 ══════════════════════════

RAW_ZH = "今天来探索一本有趣的作品呢。主角小慧，她才刚变成天使就被丢到一个嗯。"


def test_glued_punctuation_does_not_shift_the_cut_points():
    """分行器本身要耐得住帶標點的項目（任何引擎都可能這樣給）。"""
    out = _lines(_glued(RAW_ZH), RAW_ZH)
    texts = [t.rstrip("".join(sl._ZH_CLAUSE_END)) for (_s, _e, t, _sp) in out]
    assert texts == ["今天来探索一本有趣的作品呢", "主角小慧", "她才刚变成天使就被丢到一个嗯"]


def test_crisp_strips_glued_punctuation_back_to_the_contract():
    items = ce._strip_item_punct(_glued(RAW_ZH) + [("。", 9.0, 9.1), ("3.5", 9.2, 9.4)])
    assert all(not (set(w) & sl._ZH_CLAUSE_END) for (w, _s, _e) in items[:-1])
    assert items[-1][0] == "3.5"                      # 項目中間的點不動
    assert "。" not in [w for (w, _s, _e) in items]    # 只剩標點的項目丟掉
    out = _lines(items[:-1], RAW_ZH)
    assert [t for (_s, _e, t, _sp) in out] == [
        "今天来探索一本有趣的作品呢", "主角小慧", "她才刚变成天使就被丢到一个嗯"]


def test_pure_punctuation_items_do_not_shift_either():
    """沒有對齊器的內插路徑：標點自己是一個項目。"""
    raw = "你好嗎，我很好。再見"
    items = [(ch, i * 0.2, i * 0.2 + 0.2) for i, ch in enumerate(raw)]
    out = _lines(items, raw)
    assert [t for (_s, _e, t, _sp) in out] == ["你好嗎", "，我很好", "。再見"]


# ══════════════════════════ 行尾標點記號 ══════════════════════════

def test_line_remembers_the_punctuation_that_ended_it():
    raw = "子犬のぬいぐるみ。ネム先輩にもらった、綺麗なビー玉"
    out = _lines(_items(raw.replace("。", "").replace("、", "")), raw, with_words=True)
    assert [t for (_s, _e, t, _sp, _w) in out] == ["子犬のぬいぐるみ", "ネム先輩にもらった", "綺麗なビー玉"]
    assert [w[-1].get("punct", "") for (_s, _e, _t, _sp, w) in out] == ["。", "、", ""]
    assert all("punct" not in x for (_s, _e, _t, _sp, w) in out for x in w[:-1])


def test_translator_rebuilds_sentences_from_the_punctuation_marks():
    def seg(text, start, punct=""):
        words = [{"start": start, "end": start + 1, "text": text}]
        if punct:
            words[-1]["punct"] = punct
        return {"start": start, "end": start + 1, "text": text, "speaker": None, "words": words}

    segs = [seg("大好きだったお菓子の空き缶は", 0, "、"), seg("タイムマシン", 1.2, "。"),
            seg("子犬のぬいぐるみ", 2.4, "。"), seg("あとは何を入れたっけ", 3.6)]
    assert tr.plan_sentences(segs) == [
        (0, 1, "大好きだったお菓子の空き缶は、タイムマシン。"),
        (2, 2, "子犬のぬいぐるみ。"),
        (3, 3, "あとは何を入れたっけ"),
    ]


# ══════════════════════════ 日文超長行 ══════════════════════════

def test_long_japanese_line_breaks_at_a_pause_not_at_the_limit():
    text = "タイムマシン学校の授業で描いたお父さんとお母さんの絵"       # 27 字，無標點
    cut = text.index("お父さん")
    out = _lines(_items(text, gaps={cut: 0.4}), text)
    assert [t for (_s, _e, t, _sp) in out] == [text[:cut], text[cut:]]


def test_long_japanese_line_without_pauses_breaks_after_a_particle():
    text = "散りゆく桜を踏みしめながら遠くの山を静かに眺めていた"         # 26 字
    out = _lines(_items(text), text)
    first = out[0][2]
    assert first.endswith("を") or first.endswith("の") or first.endswith("ら")
    assert "".join(t for (_s, _e, t, _sp) in out) == text


def test_soft_cut_never_starts_a_line_with_small_kana_or_n():
    words = list("あいうえおかきくけこさしすせそたちつてとなっ")
    starts = [i * 0.2 for i in range(len(words))]
    ends = [s + 0.2 for s in starts]
    starts[-1] += 0.5                                   # 促音前有停頓也不能切
    cut = sl._ja_soft_cut(words, starts, ends)
    assert cut == len(words) or words[cut] not in sl._JA_NO_START


def test_long_chinese_line_is_unchanged():
    """中文不套日文的軟換行：維持超過上限就送出整行。"""
    text = "今天我們要來談談一個非常非常非常重要而且有趣的主題"
    out = _lines(_items(text, gaps={12: 0.4}), text)
    assert len(out[0][2]) == sl.MAX_CHARS + 1
