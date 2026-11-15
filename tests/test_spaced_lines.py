"""英文／韓文（以空白分詞的語言）字幕分行的回歸測試。

執行：
    python -m pytest tests/test_spaced_lines.py

全部是純邏輯，不需要模型或 GPU。測試資料取自 2026-11-16 的實測原始輸出：
英文是影集對白（CrispASR＋Qwen3-ASR-1.7B），韓文是 FLEURS ko_kr。三種引擎
給的字級項目格式都不一樣，這裡各放一份：

    CrispASR 英文          一項一個詞，標點黏在詞尾，原文沒有空白
    CrispASR 韓文＋對齊器  一項一個音節，詞界用原文裡的空白標示（且不完整）
    OpenVINO／chatllm 英文 一項一個詞、不含標點，標點與空白都只在原文裡

對應 GitHub issue #13（英文斷句切單字）與 #15（開對齊後韓文斷句異常）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import crisp_engine as ce                      # noqa: E402
import subtitle_lines as sl                    # noqa: E402


def _seq(words, start=0.0, dur=0.3, gaps=None):
    """詞清單 → 連續的 (word, start, end)；gaps={索引: 該詞前的停頓秒數}。"""
    out, t = [], start
    for k, w in enumerate(words):
        t += (gaps or {}).get(k, 0.0)
        out.append((w, t, t + dur))
        t += dur
    return out


def _lines(items, raw_text, **kw):
    return sl._ts_chatllm_to_subtitle_lines(items, raw_text, 0.0, None, None, True, **kw)


def _texts(items, raw_text, **kw):
    return [t for (_s, _e, t, _spk) in _lines(items, raw_text, **kw)]


# ══════════════════════ 語言判定 ══════════════════════

def test_script_detection():
    assert sl.is_spaced_script(_seq(["Hello,", "world."]))
    assert sl.is_spaced_script(_seq(["다", "리", "및"]))               # 韓文諺文
    assert not sl.is_spaced_script(_seq(list("今天天氣很好")))
    assert not sl.is_spaced_script(_seq(["きょう", "は", "晴れ"]))     # 日文
    # 中英夾雜的中文仍走中文契約
    assert not sl.is_spaced_script(_seq(["我", "們", "去", "打", "golf", "好", "嗎"]))
    assert not sl.is_spaced_script([])


def test_chinese_is_untouched():
    """中文路徑不能被這次改動影響：照標點切、輸出不帶標點、不插空白。"""
    words = list("今天天氣很好我們出去走走")
    assert _texts(_seq(words), "今天天氣很好，我們出去走走。") == ["今天天氣很好", "我們出去走走"]


# ══════════════════════ 英文 ══════════════════════

EN = ["I", "hate", "you.", "Good.", "Hey,", "hon.", "Anything", "going", "on?",
      "Any", "particular", "reason", "you", "put", "it", "where", "the", "alarm",
      "clock", "used", "to", "be?"]


def test_english_breaks_on_sentences_and_keeps_punctuation():
    """#13：照句子斷、標點留著、不切到單字中間。"""
    items = _seq(EN, gaps={4: 1.0, 6: 1.0, 9: 1.0})
    got = _texts(items, "".join(EN))                 # CrispASR：原文沒有空白
    assert got == ["I hate you. Good.", "Hey, hon.", "Anything going on?",
                   "Any particular reason you put",
                   "it where the alarm clock used to be?"]
    # 每個輸出的詞都必須是輸入裡的完整詞
    assert {w for line in got for w in line.split()} <= set(EN)


def test_english_lines_respect_the_width_limit():
    words = ("So when you come home tonight, if there's a can of beer missing and you "
             "can't find the remote control, that's where they are.").split()
    got = _texts(_seq(words), "".join(words))
    assert all(len(t) <= sl.MAX_SPACED_CHARS for t in got), got
    assert " ".join(got) == " ".join(words)          # 一個詞都沒少、順序不變
    assert got[0].endswith(",")                      # 優先在逗號後換行


def test_english_without_aligner_has_no_space_before_punctuation():
    """關對齊：2.0.1 會輸出「you . Good .」與行首空白。"""
    raw_items = [("hate you. Good. Hey, hon.", 0.0, 4.0)]     # 一項一整段
    items = ce._interpolate_words(raw_items)
    assert [w for (w, _s, _e) in items] == ["hate", "you.", "Good.", "Hey,", "hon."]
    assert items[0][1] == 0.0 and items[-1][2] == 4.0
    got = _texts(items, "hate you. Good. Hey, hon.")
    assert got == ["hate you. Good. Hey, hon."]
    assert " ." not in got[0] and " ," not in got[0]


def test_openvino_style_items_get_their_punctuation_back():
    """OpenVINO／chatllm：項目不含標點，從 ASR 原文接回來。"""
    words = ["Nobody", "likes", "it", "Nobody", "thinks", "it's", "funny"]
    got = _texts(_seq(words), "Nobody likes it. Nobody thinks it's funny.")
    assert got == ["Nobody likes it.", "Nobody thinks it's funny."]


def test_mismatched_raw_text_falls_back_instead_of_inventing_text():
    """原文對不上時退回既有分行，不能丟字。"""
    words = ["hello", "world"]
    assert sl.reattach_punct(_seq(words), "something else entirely") is None
    got = _texts(_seq(words), "something else entirely")
    assert "".join(got).replace(" ", "") == "helloworld"


def test_stretched_word_keeps_its_end():
    """對齊器把詞前的停頓算進該詞：「I」33.67–37.03，真正開口在 37 秒附近。"""
    items = [("be?", 33.59, 33.67), ("I", 33.67, 37.03), ("thought", 37.03, 37.27),
             ("it", 37.27, 37.35), ("would.", 37.35, 37.75)]
    lines = _lines(items, "be?Ithoughtitwould.")
    start = next(s for (s, _e, t, _spk) in lines if t.startswith("I thought"))
    assert 36.5 < start < 37.03


def test_fragment_cut_off_by_a_pause_joins_the_next_line():
    """`Where` ‖（停頓）‖ `are the kids?` 不能各自成行。"""
    items = [("slip.", 48.7, 49.03), ("Where", 49.19, 49.27), ("are", 49.27, 53.51),
             ("the", 53.51, 53.67), ("kids?", 53.67, 53.99)]
    lines = _lines(items, "slip.Wherearethekids?")
    assert [t for (_s, _e, t, _spk) in lines] == ["slip.", "Where are the kids?"]
    start = lines[1][0]
    assert 52.5 < start < 53.51                      # 貼在下一行開頭之前，不是 49 秒


def test_abbreviation_does_not_end_the_sentence():
    words = ["Mr.", "Bundy", "sells", "shoes."]
    assert _texts(_seq(words), "".join(words)) == ["Mr. Bundy sells shoes."]


def test_word_level_output_matches_the_line():
    items = _seq(["Now", "go", "to", "school."], start=12.89)
    (s, e, text, _spk, words), = _lines(items, "Nowgotoschool.", with_words=True)
    assert text == "Now go to school."
    assert [w["text"] for w in words] == ["Now", "go", "to", "school."]
    assert words[0]["start"] == s and words[-1]["end"] == e


def test_text_only_path_keeps_apostrophes_and_sentences():
    """OpenVINO／chatllm 關對齊：2.0.1 會輸出「Nobody thinks it' s」與兩三個詞一行。"""
    text = "Nobody likes it. Nobody thinks it's funny. So cut it out, okay?"
    lines = sl.spaced_lines_from_text(text, 8.0, 14.0, "說話者1")
    assert [t for (_s, _e, t, _spk, _w) in lines] == [
        "Nobody likes it.", "Nobody thinks it's funny.", "So cut it out, okay?"]
    assert lines[0][0] == 8.0 and lines[-1][1] == 14.0
    assert all(a[1] <= b[0] + 1e-9 for a, b in zip(lines, lines[1:]))    # 不重疊
    assert all(spk == "說話者1" and w == [] for (_s, _e, _t, spk, w) in lines)


def test_text_only_path_leaves_chinese_to_the_old_splitter():
    assert sl.spaced_lines_from_text("今天天氣很好，我們出去走走。", 0.0, 5.0) is None
    assert sl.spaced_lines_from_text("   ", 0.0, 5.0) is None


# ══════════════════════ 韓文 ══════════════════════

def _syllables(marked: str, start=1.22, dur=0.16):
    """「다|리| 및| 수|직」→ 音節項目＋原文（前導空白＝對齊器標的詞界）。"""
    toks = marked.split("|")
    items = _seq([t.strip() for t in toks], start=start, dur=dur)
    return items, "".join(toks)


def test_korean_syllables_are_regrouped_into_words():
    """#15：2.0.1 會輸出「다리및수직간격은15m」並從詞中間斷行。"""
    items, raw = _syllables("다|리| 및| 수|직| 간|격|은|15m|이|며| 공|사|는|2011|년|8|월|에"
                            "| 마|무|리")
    got = _texts(items, raw)
    assert " ".join(got) == "다리 및 수직 간격은 15m이며 공사는 2011년 8월에 마무리"


def test_korean_sentence_start_is_not_glued_to_the_previous_sentence():
    """對齊器在句首不標空白：`이다.`+`델` 要分開，而且各自留在自己的時間上。"""
    a, raw_a = _syllables("까|지|이|다.", start=9.62)
    b, raw_b = _syllables("델| 포|트|로|가| 이| 세|트|에|서", start=15.07)
    lines = _lines(a + b, raw_a + raw_b)
    assert [t for (_s, _e, t, _spk) in lines] == ["까지이다.", "델 포트로가 이 세트에서"]
    assert lines[0][1] < 11.0 and lines[1][0] >= 15.0


def test_korean_width_limit_counts_wide_characters():
    words = ("겨울에 북발트해를 건널 경우에는 빙판을 통과하면서 꽤 끔찍한 소음이 발생하기 "
             "때문에 객실의 위치를 확인하는 것이 좋다.").split()
    got = _texts(_seq(words), " ".join(words))
    assert len(got) >= 3
    assert all(sl._disp_len(t) <= sl.MAX_SPACED_CHARS for t in got), got
    assert " ".join(got) == " ".join(words)


def test_leading_apostrophe_does_not_derail_the_whole_file():
    """`'cause` 開頭的撇號曾被前一個詞吃掉，導致整集退回舊分行。"""
    words = ["I", "stayed", "home", "'cause", "it", "rained.", "Tell", "'em", "that."]
    items = sl.reattach_punct(_seq(words), "".join(words))
    assert [w for (w, _s, _e) in items] == words
    assert _texts(_seq(words), "".join(words)) == ["I stayed home 'cause it rained.",
                                                   "Tell 'em that."]


def test_one_unmatched_word_is_passed_through():
    """原文裡少一個詞：該詞原樣通過，其餘照常接回標點。"""
    items = sl.reattach_punct(_seq(["Well", "um", "you", "didn't", "tell", "me", "that"]),
                              "Well, you didn't tell me that.")
    assert [w for (w, _s, _e) in items] == ["Well,", "um", "you", "didn't", "tell", "me",
                                           "that."]


def test_zero_length_words_are_kept_and_lines_never_overlap():
    """對齊器常給零長度的詞（`huh.` 30.15–30.15）。2.0.1 遇到整行零長度會把那行丟掉，
    這就是 #15「開對齊後偶爾缺字」的來源。"""
    items = [("Uh", 29.59, 30.15), ("huh.", 30.15, 30.15), ("Okay.", 33.0, 33.0),
             ("Any", 33.0, 33.2), ("reason?", 33.2, 33.6)]
    lines = _lines(items, "Uhhuh.Okay.Anyreason?")
    assert " ".join(t for (_s, _e, t, _spk) in lines) == "Uh huh. Okay. Any reason?"
    assert all(e > s for (s, e, _t, _spk) in lines)
    assert all(a[1] <= b[0] for a, b in zip(lines, lines[1:]))
