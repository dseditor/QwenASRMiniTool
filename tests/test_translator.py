"""字幕翻譯（translator.py）與 WebBackend 接線的測試。

執行：
    python -m pytest tests/test_translator.py                  # 全部
    python -m pytest tests/test_translator.py -m "not slow"    # 只跑純邏輯（秒級）

分兩層：
  • 純邏輯：用假的翻譯伺服器驗證「以句子為單位翻譯、同一句的幾行共用譯文」「內容已是目標語言
    就略過」「原文＋譯文／只留譯文各寫出哪些檔」「翻譯失敗不拖垮轉錄」。
  • 整合（slow）：真的起 crispasr 伺服器載入翻譯模型翻幾句。缺核心或模型時略過
    （模型在第一次從介面使用翻譯時才會下載到 ov_models/translator/）。
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as core                             # noqa: E402
import downloader                              # noqa: E402
import translator as tr                        # noqa: E402
import webview_backend as wb                   # noqa: E402
from subtitle_lines import write_transcript    # noqa: E402


class FakeServer:
    """記下收到的原文，回傳「[目標]原文」當譯文；可指定個別原文的譯文。"""

    def __init__(self, answers=None):
        self.answers = answers or {}
        self.seen = []

    def translate(self, text, target, timeout=0):
        self.seen.append(text)
        return self.answers.get(text, f"[{target}]{text}")


def _segs(*texts, gap=0.1, speaker=None):
    out, t = [], 0.0
    for x in texts:
        out.append({"start": t, "end": t + 2.0, "text": x, "speaker": speaker})
        t += 2.0 + gap
    return out


# ══════════════════════════ 語言判斷 ══════════════════════════

@pytest.mark.parametrize("texts,lang", [
    (["Where are the kids?", "Oh, they've left."], "en"),
    (["今天天氣很好", "我們去公園散步吧"], "zh"),
    (["明日の会議は午後三時からに変更になりました"], "ja"),
    (["오늘 날씨가 정말 좋아서 공원에 갔어요"], "ko"),
    (["...", "1987"], ""),
])
def test_detect_language(texts, lang):
    assert tr.detect_language(texts) == lang


def test_normalize_target_rejects_unknown_values():
    assert tr.normalize_target("ZH") == "zh"
    assert tr.normalize_target("ko") == ""
    assert tr.normalize_target(None) == ""


# ══════════════════════════ 句子重組 ══════════════════════════

def _plan(segs):
    return [(a, b) for a, b, _t in tr.plan_sentences(segs)]


def test_a_sentence_split_over_two_lines_is_one_sentence():
    segs = _segs("Any particular reason you put it", "where the alarm clock used to be?",
                 "Nope.")
    assert tr.plan_sentences(segs) == [
        (0, 1, "Any particular reason you put it where the alarm clock used to be?"),
        (2, 2, "Nope.")]


def test_two_sentences_on_one_line_stay_together():
    """行內有兩句而且行尾就是句末 → 整行當一個單位，不拆開（拆了反而少了上下文）。"""
    assert tr.plan_sentences(_segs("Well you didn't. Sorry.")) == [(0, 0, "Well you didn't. Sorry.")]


def test_japanese_sentences_that_end_mid_line_are_rebuilt():
    """日文：句子在行中間結束、下一句從行中間開始，詞還被切成兩行。"""
    segs = _segs("お父さんとお母さん", "の絵。子犬のぬい", "ぐるみ。ネム先輩にもらった綺麗",
                 "なビー玉。")
    assert tr.plan_sentences(segs) == [
        (0, 1, "お父さんとお母さんの絵。"),
        (1, 2, "子犬のぬいぐるみ。"),
        (2, 3, "ネム先輩にもらった綺麗なビー玉。")]


def test_sentence_does_not_continue_across_pause_speaker_change_or_four_lines():
    paused = _segs("I was going to say", "something else.", "Right.", gap=2.0)
    assert _plan(paused) == [(0, 0), (1, 1), (2, 2)]

    spk = _segs("I was going to say", "something else.", "Right.")
    spk[1]["speaker"] = 2
    assert _plan(spk)[0] == (0, 0)

    long = _segs("one two", "three four", "five six", "seven eight", "nine ten.", "Yes.")
    assert _plan(long) == [(0, 3), (4, 4), (5, 5)]


def test_unpunctuated_subtitles_are_translated_line_by_line():
    """中文字幕行不帶標點、行就是子句；不能因此把好幾行黏成一句。"""
    assert _plan(_segs("今天天氣很好", "我們去公園散步吧", "好啊")) == [(0, 0), (1, 1), (2, 2)]


def test_abbreviations_and_decimals_do_not_end_a_sentence():
    segs = _segs("I met Mr. Smith and paid 3.5 dollars", "for the ticket.", "Fine.")
    assert _plan(segs) == [(0, 1), (2, 2)]


# ══════════════════════════ translate_segments ══════════════════════════

def test_multi_line_sentence_is_translated_once_and_shared_by_its_lines():
    whole = "Any particular reason you put it where the alarm clock used to be?"
    srv = FakeServer({whole: "你為什麼把它放在鬧鐘原來的位置？"})
    segs = _segs("Any particular reason you put it", "where the alarm clock used to be?",
                 "Nope.")
    out, cont = tr.translate_segments(srv, segs, "zh")
    assert srv.seen == [whole, "Nope."]           # 整句只送一次，沒有逐行送半句
    assert out == ["你為什麼把它放在鬧鐘原來的位置？"] * 2 + ["[zh]Nope."]
    assert cont == [False, True, False]


def test_a_line_where_one_sentence_ends_and_another_starts_shows_the_new_one():
    srv = FakeServer({"お父さんとお母さんの絵。": "爸爸和媽媽的畫。",
                      "子犬のぬいぐるみ。": "小狗布偶。"})
    out, cont = tr.translate_segments(
        srv, _segs("お父さんとお母さん", "の絵。子犬のぬい", "ぐるみ。"), "zh")
    assert out == ["爸爸和媽媽的畫。", "小狗布偶。", "小狗布偶。"]
    assert cont == [False, False, True]


def test_lines_of_a_sentence_that_failed_keep_no_translation():
    whole = "I was going to say something else."
    srv = FakeServer({whole: ""})                 # 整句沒翻出來
    out, cont = tr.translate_segments(
        srv, _segs("I was going to say", "something else.", "Right."), "zh")
    assert out == ["", "", "[zh]Right."]
    assert cont == [False, False, False]          # 不標接續 → 寫檔時兩行都保留原文


def test_merge_translated_joins_continuation_lines_into_one_cue():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "Any particular reason you put it",
         "translation": "你為什麼把它放在鬧鐘原來的位置？", "speaker": 1},
        {"start": 2.1, "end": 4.0, "text": "where the alarm clock used to be?",
         "translation": "你為什麼把它放在鬧鐘原來的位置？", "trCont": True, "speaker": 1},
        {"start": 5.0, "end": 6.0, "text": "Nope.", "translation": "", "speaker": 2},
    ]
    assert tr.merge_translated(segs) == [
        {"start": 0.0, "end": 4.0, "text": "你為什麼把它放在鬧鐘原來的位置？", "speaker": 1},
        {"start": 5.0, "end": 6.0, "text": "Nope.", "speaker": 2},     # 沒譯出 → 保留原文
    ]


def test_symbols_pass_through_and_repeats_are_translated_once():
    srv = FakeServer()
    out, cont = tr.translate_segments(srv, _segs("Nope.", "...", "1987", "Nope."), "zh")
    assert out == ["[zh]Nope.", "...", "1987", "[zh]Nope."]
    assert srv.seen == ["Nope."] and not any(cont)


def test_one_failing_line_does_not_fail_the_rest():
    class Flaky(FakeServer):
        def translate(self, text, target, timeout=0):
            if text == "Bad.":
                raise OSError("connection reset")
            return super().translate(text, target)

    out, _cont = tr.translate_segments(Flaky(), _segs("Good.", "Bad.", "Fine."), "zh")
    assert out == ["[zh]Good.", "", "[zh]Fine."]


def test_cancel_stops_translation():
    with pytest.raises(RuntimeError):
        tr.translate_segments(FakeServer(), _segs("One.", "Two."), "zh", cancel=lambda: True)


def test_chinese_output_is_converted_to_taiwan_traditional():
    srv = FakeServer({"Software.": "软件。"})
    assert tr.translate_segments(srv, _segs("Software."), "zh")[0] == ["軟體。"]


# ══════════════════════════ 伺服器命令列 ══════════════════════════

def test_server_command_avoids_extra_downloads_and_respects_cpu(tmp_path):
    """佔位模型＋關標點：否則 crispasr 會自己去抓 Whisper base 與標點模型。"""
    gpu = tr.TranslatorServer(tmp_path / "c.exe", tmp_path / "m.gguf", tmp_path / "s.bin",
                              gpu_backend="vulkan", device_id=1)
    gpu.port = 1234
    cmd = gpu._command()
    assert cmd[cmd.index("-m") + 1].endswith("s.bin")
    assert "--no-punctuation" in cmd and "--chat-model" in cmd
    assert cmd[cmd.index("--gpu-backend") + 1] == "vulkan" and cmd[cmd.index("-dev") + 1] == "1"

    cpu = tr.TranslatorServer(tmp_path / "c.exe", tmp_path / "m.gguf", tmp_path / "s.bin",
                              gpu_backend=None)._command()
    assert "-ng" in cpu and cpu[cpu.index("--chat-gpu-layers") + 1] == "0"
    assert "--gpu-backend" not in cpu


def test_quick_check_rejects_a_half_downloaded_model(tmp_path):
    model, stub = downloader.translator_paths(tmp_path)
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x" * 1000)                # 下載到一半的殘檔
    stub.write_bytes(b"x" * 1000)
    assert downloader.quick_check_translator(tmp_path) is False


# ══════════════════════════ WebBackend 接線 ══════════════════════════

class _Engine:
    """假引擎：寫出一份 SRT，並留下字級側通道（與真引擎相同的契約）。"""
    ready = True

    def __init__(self, texts):
        self.texts = texts
        self._last_segments_rich = None

    def process_file(self, audio_path, progress_cb=None, original_path=None, **kw):
        lines = [(i * 2.0, i * 2.0 + 1.9, t, None) for i, t in enumerate(self.texts)]
        self._last_segments_rich = [
            {"start": s, "end": e, "text": t, "speaker": None, "words": []}
            for (s, e, t, _spk) in lines]
        return write_transcript(original_path, lines, out_format="srt")


@pytest.fixture
def backend(tmp_path, monkeypatch):
    """接好假引擎與假翻譯伺服器的 WebBackend；回傳 (backend, 音檔路徑, 字幕資料夾)。"""
    f = tmp_path / "settings.json"
    real = Path(core.SETTINGS_FILE)
    if real.exists():
        shutil.copyfile(real, f)
    cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    cur.update(output_format="srt", translate_keep="both", crisp_variant="vulkan")
    f.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_FILE", f)
    srt_dir = tmp_path / "subtitles"
    monkeypatch.setattr(core, "SRT_DIR", srt_dir)

    monkeypatch.setattr(downloader, "quick_check_crispasr", lambda *a, **k: True)
    monkeypatch.setattr(downloader, "quick_check_translator", lambda *a, **k: True)

    class _Srv(FakeServer):
        def __init__(self, *a, **k):
            super().__init__()

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(tr, "TranslatorServer", _Srv)

    be = wb.WebBackend()
    be.engine = _Engine(["Where are the kids?", "Oh, they've left."])
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF")
    return be, audio, srt_dir


def _run(be, audio, **opts):
    return be.transcribe({"path": str(audio), "name": "clip.wav", "align": False, **opts})


def test_no_translation_unless_asked(backend):
    be, audio, srt_dir = backend
    res = _run(be, audio)
    assert "translation" not in res
    assert all("translation" not in s for s in res["segments"])
    assert sorted(p.name for p in srt_dir.iterdir()) == ["clip.srt"]


def test_both_keeps_the_original_and_adds_a_translated_file(backend):
    be, audio, srt_dir = backend
    res = _run(be, audio, translate="zh")
    assert res["translation"]["ok"] is True
    assert [s["translation"] for s in res["segments"]] == \
        ["[zh]Where are the kids?", "[zh]Oh, they've left."]
    assert sorted(p.name for p in srt_dir.iterdir()) == ["clip.srt", "clip.zh-TW.srt"]
    assert "Where are the kids?" in (srt_dir / "clip.srt").read_text(encoding="utf-8")
    translated = (srt_dir / "clip.zh-TW.srt").read_text(encoding="utf-8")
    assert "[zh]Where are the kids?" in translated and "00:00:02,000" in translated
    assert res["translation"]["path"].endswith("clip.zh-TW.srt")


def test_a_sentence_over_two_lines_becomes_one_cue_in_the_translated_file(backend):
    be, audio, srt_dir = backend
    be.engine = _Engine(["Any particular reason you put it",
                         "where the alarm clock used to be?", "Nope."])
    res = _run(be, audio, translate="zh")
    segs = res["segments"]
    assert segs[0]["translation"] == segs[1]["translation"]
    assert segs[1].get("trCont") is True and "trCont" not in segs[0]
    assert (srt_dir / "clip.srt").read_text(encoding="utf-8").count("-->") == 3
    translated = (srt_dir / "clip.zh-TW.srt").read_text(encoding="utf-8")
    assert translated.count("-->") == 2
    assert "00:00:00,000 --> 00:00:03,900" in translated      # 涵蓋整句的時間


def test_only_replaces_the_original_with_the_translation(backend):
    be, audio, srt_dir = backend
    res = _run(be, audio, translate="zh", translateKeep="only")
    assert sorted(p.name for p in srt_dir.iterdir()) == ["clip.srt"]
    text = (srt_dir / "clip.srt").read_text(encoding="utf-8")
    assert "[zh]Where are the kids?" in text and "\nWhere are the kids?" not in text
    assert res["srtPath"].endswith("clip.srt")


def test_plain_text_format_applies_to_both_files(backend, monkeypatch):
    be, audio, srt_dir = backend
    be.set_settings({"format": "txt"})
    _run(be, audio, translate="ja")
    assert sorted(p.name for p in srt_dir.iterdir()) == ["clip.ja.txt", "clip.txt"]


def test_same_language_is_skipped_without_starting_the_server(backend, monkeypatch):
    be, audio, srt_dir = backend

    def _boom(*a, **k):
        raise AssertionError("不該啟動翻譯伺服器")

    monkeypatch.setattr(tr, "TranslatorServer", _boom)
    res = _run(be, audio, translate="en")
    assert res["translation"]["ok"] is False and "略過" in res["translation"]["skipped"]
    assert sorted(p.name for p in srt_dir.iterdir()) == ["clip.srt"]


def test_translation_failure_keeps_the_transcript(backend, monkeypatch):
    be, audio, srt_dir = backend

    def _boom(*a, **k):
        raise RuntimeError("翻譯伺服器啟動失敗：no vulkan")

    monkeypatch.setattr(tr, "TranslatorServer", _boom)
    res = _run(be, audio, translate="zh", translateKeep="only")
    assert res["translation"]["ok"] is False and "no vulkan" in res["translation"]["error"]
    assert len(res["segments"]) == 2
    assert "Where are the kids?" in (srt_dir / "clip.srt").read_text(encoding="utf-8")


def test_translation_settings_round_trip(backend):
    be, _audio, _dir = backend
    assert be.get_settings()["translateKeep"] == "both"
    s = be.set_settings({"translate": "ja", "translateKeep": "only"})
    assert (s["translate"], s["translateKeep"]) == ("ja", "only")
    assert be.set_settings({"translate": "xx"})["translate"] == ""


# ══════════════════════════ 整合（slow）══════════════════════════

@pytest.mark.slow
def test_real_model_translates_english_and_japanese():
    exe = ROOT / "crispasr" / "crispasr.exe"
    model, stub = downloader.translator_paths(ROOT / "ov_models")
    if not exe.exists() or not downloader.quick_check_translator(ROOT / "ov_models"):
        pytest.skip("缺 crispasr 核心或翻譯模型（ov_models/translator/）")
    with tr.TranslatorServer(exe, model, stub, gpu_backend="vulkan") as srv:
        zh, cont = tr.translate_segments(srv, _segs(
            "Any particular reason you put it", "where the alarm clock used to be?",
            "The NVIDIA H200 has 141GB of HBM3e memory."), "zh")
        en, _ = tr.translate_segments(srv, _segs("明日の会議は午後三時からに変更になりました"), "en")
    assert all(zh) and "鬧鐘" in zh[0] and zh[0] == zh[1] and cont == [False, True, False]
    assert "NVIDIA" in zh[2] and "141" in zh[2]       # 專有名詞沒被吃掉
    assert tr.detect_language(zh) == "zh"
    assert "meeting" in en[0].lower()
