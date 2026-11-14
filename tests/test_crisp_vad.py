"""CrispASR 切段（VAD）與字幕分行的回歸測試。

執行：
    python -m pytest tests/test_crisp_vad.py            # 全部
    python -m pytest tests/test_crisp_vad.py -m "not slow"   # 只跑純邏輯（秒級）

分兩層：
  • 純邏輯：不需要 crispasr.exe／模型／GPU，驗證「依模型選哪一種 VAD」「命令列
    長什麼樣」「字元內插」「拉長字元收回」「中英夾雜的空白」。
  • 整合（slow）：實際呼叫 crispasr.exe 跑「人聲＋120 秒靜音＋人聲」，驗證靜音段
    不出字、時間軸不倒退、字幕不會一行十幾秒。缺核心／模型／測試音檔／ffmpeg
    時自動略過。這份測試音檔就是 2026-11-15 重現 GitHub issue #31 用的那一份。
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import crisp_engine as ce                      # noqa: E402
import subtitle_lines as sl                    # noqa: E402

CRISP_DIR = ROOT / "crispasr"
MODEL_DIR = ROOT / "ov_models"
NEWS = ROOT / "test_audio" / "News.mp3"

# 測試音檔的結構（秒）：[0,30) 人聲、[30,150) 靜音、[150,180) 人聲
SPEECH1_END, SILENCE_END, TOTAL = 30.0, 150.0, 180.0


# ══════════════════════════ 純邏輯 ══════════════════════════

def _engine(tmp_path: Path, backend: str, *, silero: bool, fa: bool,
            vad: bool = True) -> ce.CrispWhisperEngine:
    """組一個不經 load() 的引擎（只填 _vad_plan／_build_cmd 會讀的欄位）。"""
    eng = ce.CrispWhisperEngine()
    eng._exe = tmp_path / "crispasr.exe"
    eng._model_path = tmp_path / "model.gguf"
    eng._backend = backend
    eng._crispasr_dir = tmp_path
    eng.vad_for_whisper = True      # 模擬使用者明確選了「原生 VAD」
    if silero:
        (tmp_path / "ggml-silero-v6.2.0.bin").write_bytes(b"\0" * 200_000)
    firered = tmp_path / "firered-vad.gguf"
    firered.write_bytes(b"\0" * 16)
    eng.vad_model = firered if vad else None
    if fa:
        eng._aligner_path = tmp_path / "aligner.gguf"
        eng.use_aligner = True
    return eng


@pytest.mark.parametrize("backend,silero,fa,mode,model_name", [
    # Qwen／TEA：一律 FireRed，有無對齊器都一樣
    ("qwen3-1.7b", True,  True,  "unified", "firered-vad.gguf"),
    ("qwen3-1.7b", False, False, "unified", "firered-vad.gguf"),
    # Whisper／Breeze：有 Silero 就走 legacy＋Silero，與對齊器無關
    ("whisper",    True,  True,  "legacy",  "ggml-silero-v6.2.0.bin"),
    ("whisper",    True,  False, "legacy",  "ggml-silero-v6.2.0.bin"),
    # 沒有 Silero：沒掛對齊器才退而用 FireRed，掛了對齊器就不掛 VAD
    ("whisper",    False, False, "unified", "firered-vad.gguf"),
    ("whisper",    False, True,  None,      None),
])
def test_vad_plan_matrix(tmp_path, backend, silero, fa, mode, model_name):
    eng = _engine(tmp_path, backend, silero=silero, fa=fa)
    got_mode, got_model = eng._vad_plan(True)
    assert got_mode == mode
    assert (Path(got_model).name if got_model else None) == model_name


def test_vad_plan_off_when_route_disabled_or_live_mode(tmp_path):
    assert _engine(tmp_path, "qwen3-1.7b", silero=True, fa=True,
                   vad=False)._vad_plan(True) == (None, None)
    # 即時模式（word_level=False）不掛 VAD：前端已切過，再切會削掉字頭字尾
    assert _engine(tmp_path, "whisper", silero=True, fa=False)._vad_plan(False) == (None, None)


def test_cmd_whisper_uses_legacy_path_with_silero(tmp_path):
    eng = _engine(tmp_path, "whisper", silero=True, fa=True)
    cmd = eng._build_cmd(tmp_path / "a.wav", tmp_path / "out", None, word_level=True)
    assert "--backend" not in cmd          # legacy 路徑：顯式 --backend 會讓 Silero 失效
    assert cmd[cmd.index("-vm") + 1].endswith("ggml-silero-v6.2.0.bin")
    assert "--vad" in cmd


def test_cmd_qwen_uses_firered(tmp_path):
    eng = _engine(tmp_path, "qwen3-1.7b", silero=True, fa=True)
    cmd = eng._build_cmd(tmp_path / "a.wav", tmp_path / "out", None, word_level=True)
    assert cmd[cmd.index("--backend") + 1] == "qwen3-1.7b"
    assert cmd[cmd.index("-vm") + 1].endswith("firered-vad.gguf")
    assert "-falign" in cmd


def test_cmd_without_vad_is_unchanged(tmp_path):
    eng = _engine(tmp_path, "whisper", silero=True, fa=False, vad=False)
    cmd = eng._build_cmd(tmp_path / "a.wav", tmp_path / "out", None, word_level=True)
    assert "--vad" not in cmd and "--backend" not in cmd


def test_interpolate_chars_spreads_long_item_evenly():
    out = ce._interpolate_chars([("你好嗎", 10.0, 13.0), ("啊", 13.0, 13.2)])
    assert [t for t, _, _ in out] == ["你", "好", "嗎", "啊"]
    assert out[0][1:] == (10.0, 11.0) and out[2][1:] == (12.0, 13.0)
    assert out[3] == ("啊", 13.0, 13.2)            # 單字元原樣保留


def test_clamp_stretched_snaps_to_normal_neighbour():
    # 實測樣本：「游」「泳」兩個字被拉長跨過 120 秒靜音
    items = [("去", 29.52, 29.78), ("游", 29.78, 126.02),
             ("泳", 126.02, 150.10), ("池", 150.10, 150.39)]
    out = ce._clamp_stretched(items)
    assert out[1] == ("游", 29.78, pytest.approx(30.08))     # 貼前一個字
    assert out[2] == ("泳", pytest.approx(149.80), 150.10)   # 貼後一個字
    assert out[0] == items[0] and out[3] == items[3]
    assert all(e - s <= 2.0 for _, s, e in out)


@pytest.mark.parametrize("words,expected", [
    (["打", "高", "golf，", "跟", "那", "個"], "打高golf，跟那個"),   # issue：逐字空白
    (["hello", "world"],                      "hello world"),
    (["Hello,", "world"],                     "Hello, world"),
    (["iPhone", "15"],                        "iPhone 15"),
    (["1", "0", "點"],                         "10點"),              # 純數字不拆
    (["你", "好"],                             "你好"),
])
def test_join_words_spacing(words, expected):
    assert sl._join_words(words) == expected


# ══════════════════════════ 整合（實際呼叫 crispasr.exe）══════════════════════════

def _need(*paths: Path):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip("缺少：" + ", ".join(missing))


@pytest.fixture(scope="session")
def gap_wav(tmp_path_factory) -> Path:
    """News 30–60s ＋ 120s 純靜音 ＋ News 60–90s（16k mono）。"""
    ff = shutil.which("ffmpeg") or (str(ROOT / "ffmpeg" / "ffmpeg.exe")
                                    if (ROOT / "ffmpeg" / "ffmpeg.exe").exists() else None)
    if not ff:
        pytest.skip("找不到 ffmpeg")
    _need(NEWS)
    out = tmp_path_factory.mktemp("gap") / "gap120.wav"
    graph = ("[0:a]atrim=30:60,asetpts=PTS-STARTPTS[a];"
             "anullsrc=r=16000:cl=mono,atrim=0:120[s];"
             "[0:a]atrim=60:90,asetpts=PTS-STARTPTS[b];"
             "[a][s][b]concat=n=3:v=0:a=1,aresample=16000,"
             "aformat=channel_layouts=mono[o]")
    subprocess.run([ff, "-y", "-v", "error", "-i", str(NEWS), "-filter_complex",
                    graph, "-map", "[o]", str(out)], check=True)
    return out


def _run(model: Path, wav: Path, *, vad: bool, align: bool):
    """照正式流程載入並轉錄，回傳 [{start,end,text,...}]。輸出寫在 wav 旁。"""
    import downloader as dl
    eng = ce.CrispWhisperEngine()
    eng.load(model_path=model, crispasr_dir=CRISP_DIR,
             vad_model=(CRISP_DIR / dl.firered_vad_filename()) if vad else None)
    eng.use_aligner = align
    eng.vad_for_whisper = vad
    out = eng.process_file(wav, language="Chinese", out_format="srt")
    assert out is not None, "未產生字幕"
    Path(out).unlink(missing_ok=True)
    return eng._last_segments_rich, eng


def _in_silence(segs, margin: float = 1.0):
    """整行都落在靜音區內的字幕（留 margin 容忍邊界誤差）。"""
    return [s for s in segs
            if s["start"] >= SPEECH1_END + margin and s["end"] <= SILENCE_END - margin]


def _check_common(segs):
    starts = [s["start"] for s in segs]
    assert starts == sorted(starts), "時間軸倒退"
    assert max(s["end"] for s in segs) <= TOTAL + 1.0, "時間超出音檔長度"
    assert all(len(s["text"].replace(" ", "")) > 0 for s in segs)


QWEN = MODEL_DIR / "qwen3-asr-1.7b-q8_0.gguf"
TEA = MODEL_DIR / "tea-asr-1.1-q8_0.gguf"
BREEZE = CRISP_DIR / "ggml-model-q8_0.bin"
CORE = CRISP_DIR / "crispasr.exe"
FIRERED = CRISP_DIR / "firered-vad.gguf"
SILERO = CRISP_DIR / "ggml-silero-v6.2.0.bin"


@pytest.mark.slow
@pytest.mark.parametrize("vad", [False, True], ids=["builtin", "firered"])
def test_qwen_aligned_keeps_silence_empty(gap_wav, vad):
    """Qwen＋對齊：靜音段不出字（上游 #471，crispasr ≥ 0.8.38）。"""
    _need(CORE, QWEN, FIRERED)
    segs, eng = _run(QWEN, gap_wav, vad=vad, align=True)
    if not eng._aligner_path:
        pytest.skip("缺對齊器模型")
    _check_common(segs)
    assert _in_silence(segs) == []
    assert max(s["end"] - s["start"] for s in segs) < 12.0


@pytest.mark.slow
def test_qwen_unaligned_lines_are_short(gap_wav):
    """Qwen 關閉對齊：仍要切成正常長度的字幕行（原本一行 15 秒以上）。"""
    _need(CORE, QWEN)
    segs, _ = _run(QWEN, gap_wav, vad=False, align=False)
    _check_common(segs)
    assert len(segs) >= 12
    assert max(len(s["text"]) for s in segs) <= 2 * sl.MAX_CHARS


@pytest.mark.slow
@pytest.mark.parametrize("model", [QWEN, TEA], ids=["qwen", "tea"])
def test_unaligned_with_firered_stays_out_of_silence(gap_wav, model):
    """無對齊＋FireRed：字幕貼著人聲邊界，不落進靜音。"""
    _need(CORE, model, FIRERED)
    segs, eng = _run(model, gap_wav, vad=True, align=False)
    assert eng._last_vad_used == "firered"
    _check_common(segs)
    assert _in_silence(segs) == []


@pytest.mark.slow
def test_breeze_with_silero_does_not_hallucinate_in_silence(gap_wav):
    """Breeze＋Silero：120 秒靜音不再幻覺出數千字。"""
    _need(CORE, BREEZE, FIRERED, SILERO)
    segs, eng = _run(BREEZE, gap_wav, vad=True, align=True)
    assert eng._last_vad_used == "silero"
    _check_common(segs)
    assert _in_silence(segs) == []
    assert max(s["end"] - s["start"] for s in segs) < 12.0
    assert sum(len(s["text"]) for s in segs) < 400      # 無 VAD 時約 2000+ 字


def test_splitter_breaks_line_at_long_pause():
    """靜音前後的字不可併在同一行（否則該行會顯示整段靜音的時間）。"""
    items = [("你", 28.0, 28.3), ("好", 28.3, 28.6), ("當", 150.0, 150.3), ("初", 150.3, 150.6)]
    lines = sl._ts_chatllm_to_subtitle_lines(items, "你好當初", 0.0, None, None, True)
    assert [t for (_s, _e, t, _spk) in lines] == ["你好", "當初"]
    assert max(e - s for (s, e, _t, _spk) in lines) < 2.0


def test_interpolate_chars_keeps_latin_words_whole():
    """英文單字不可逐字母拆開（否則輸出成「g o l f」）。"""
    out = ce._interpolate_chars([("打高golf跟", 0.0, 7.0)])
    assert [t for t, _, _ in out] == ["打", "高", "golf", "跟"]
    assert out[2][1:] == (2.0, 6.0)                 # 4 個字元佔 4/7 的時間
    assert sl._join_words([t for t, _, _ in out]) == "打高golf跟"


def test_whisper_skips_vad_by_default(tmp_path):
    """預設（自動）下 Whisper／Breeze 不掛 VAD：Silero 會把歌聲切掉。"""
    eng = _engine(tmp_path, "whisper", silero=True, fa=False)
    eng.vad_for_whisper = False
    assert eng._vad_plan(True) == (None, None)
    cmd = eng._build_cmd(tmp_path / "a.wav", tmp_path / "out", None, word_level=True)
    assert "--vad" not in cmd
    # Qwen 不受這個旗標影響
    q = _engine(tmp_path, "qwen3-1.7b", silero=True, fa=True)
    q.vad_for_whisper = False
    assert q._vad_plan(True)[0] == "unified"
