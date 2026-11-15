"""WebBackend 載入路徑的回歸測試（tests/test_crisp_vad.py 只測到引擎層）。

執行：
    python -m pytest tests/test_webview_load.py                  # 全部
    python -m pytest tests/test_webview_load.py -m "not slow"    # 只跑純邏輯（秒級）

分兩層：
  • 純邏輯：用假引擎驗證「GPU 核心載入失敗 → 退回 CPU 時有沒有講出來」「VAD 路線
    有沒有真的帶到引擎上」「選 F16 之後設定與就緒判斷對不對」。
    起因是 2026-11-15 的實際事故：_load_crispasr 內一行順序寫錯 → 載入拋例外 →
    退回 OpenVINO，CLI 照樣回成功，兩個 profile 的輸出一模一樣才被發現。
  • 整合（slow）：起真正的 WebViewServer，走與瀏覽器前端完全相同的 HTTP 端點
    （/api/model → /api/load → /api/transcribe）選用並載入 F16，確認實際跑的是
    crispasr 而不是退回的 CPU 核心。缺模型／核心／測試音檔時自動略過。

所有測試都把 app.SETTINGS_FILE 與 app.SRT_DIR 指到暫存位置，不會動到使用者的
settings.json，也不會把字幕丟進 subtitles/。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.request
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app as core                             # noqa: E402
import webview_backend as wb                   # noqa: E402

MODEL_DIR = ROOT / "ov_models"
CRISP_DIR = ROOT / "crispasr"
NEWS = ROOT / "test_audio" / "News.mp3"
F16_LABEL = wb._QWEN_CASR_QUANT_LABEL["f16"]


@pytest.fixture
def scratch(tmp_path, monkeypatch):
    """暫存的 settings.json 與字幕輸出資料夾；回傳一個改設定用的函式。"""
    f = tmp_path / "settings.json"
    real = Path(core.SETTINGS_FILE)
    if real.exists():
        shutil.copyfile(real, f)
    monkeypatch.setattr(core, "SETTINGS_FILE", f)
    monkeypatch.setattr(core, "SRT_DIR", tmp_path / "subtitles")

    def patch(**kw):
        cur = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
        cur.update(kw)
        f.write_text(json.dumps(cur, ensure_ascii=False), encoding="utf-8")
        return cur
    patch.path = f
    return patch


class _Ready:
    ready = True


# ══════════════════════════ 純邏輯 ══════════════════════════

def test_gpu_load_failure_falls_back_and_says_so(scratch, monkeypatch):
    """GPU 核心載入失敗 → 退回 OpenVINO，而且狀態／事件／錯誤訊息都要反映出來。"""
    scratch(backend="crispasr")
    events = []
    be = wb.WebBackend(on_event=lambda ev, payload: events.append((ev, payload)))

    def boom():
        raise RuntimeError("crispasr 起不來")
    monkeypatch.setattr(be, "_load_crispasr", boom)
    monkeypatch.setattr(be, "_load_openvino", lambda: setattr(be, "engine", _Ready()))

    be._load_worker()

    st = be.get_status()
    assert st["modelReady"] is True
    assert st["activeBackend"] == "openvino"          # 實際在跑的核心
    assert st["backendKey"] == "crispasr"             # 使用者選的核心沒被偷改
    assert st["device"] == "CPU"
    assert "crispasr" in st["error"] and "起不來" in st["error"]
    assert any(ev == "status" and p.get("error") for ev, p in events)
    # 速度統計也要如實標成 CPU 核心，才不會把退回後的結果誤認成 GPU 的
    assert "OpenVINO" in be._speed_stats(10.0, 1.0)["backend"]


def test_load_failure_without_fallback_reports_not_ready(scratch, monkeypatch):
    """連 CPU 核心也載不起來 → modelReady=False，錯誤訊息保留。"""
    scratch(backend="crispasr")
    be = wb.WebBackend()

    def boom():
        raise RuntimeError("都壞了")
    monkeypatch.setattr(be, "_load_crispasr", boom)
    monkeypatch.setattr(be, "_load_openvino", boom)

    be._load_worker()

    assert be.get_status()["modelReady"] is False
    assert "都壞了" in be._load_err
    assert be._loading is False


class _FakeCrisp:
    """記下 _load_crispasr 交給引擎的東西；load() 時 vad_for_whisper 必須已設好。"""
    last = None

    def __init__(self):
        self.ready = False
        self.vad_for_whisper = False
        self.vad_model = None
        self.load_kw = None
        type(self).last = self

    def load(self, **kw):
        self.load_kw = kw
        self.vad_model = kw.get("vad_model")
        self.ready = True


@pytest.mark.parametrize("route,whisper_vad,has_vad_model", [
    ("auto",    False, True),      # Qwen／TEA 掛 FireRed，Breeze 不掛
    ("firered", True,  True),      # 「原生 VAD」：Breeze 也掛（Silero）
    ("builtin", False, False),     # 「固定視窗」：全部不掛
    ("bogus",   False, True),      # 不認得的值 → 當成預設 auto
])
def test_load_crispasr_passes_vad_route_to_engine(scratch, monkeypatch, tmp_path,
                                                  route, whisper_vad, has_vad_model):
    """VAD 路線要真的落到引擎上，而且整個函式能跑完（不能拋例外後被當成退回）。"""
    import crisp_engine
    import downloader
    scratch(backend="crispasr", crisp_model="qwen3", crisp_qwen_quant="q8",
            vad_route=route, crisp_variant="vulkan")
    for name in ("quick_check_crispasr", "quick_check_qwen3_asr_gguf",
                 "quick_check_breeze", "quick_check_aligner_gguf"):
        monkeypatch.setattr(downloader, name, lambda *a, **k: True)
    monkeypatch.setattr(crisp_engine, "CrispWhisperEngine", _FakeCrisp)
    be = wb.WebBackend()
    firered = tmp_path / "firered-vad.gguf"
    monkeypatch.setattr(be, "_firered_vad_path", lambda: firered)

    be._load_worker()

    eng = _FakeCrisp.last
    assert be.engine is eng, f"退回了別的核心：{be._load_err}"
    assert be._load_err is None
    assert be.get_status()["activeBackend"] == "crispasr"
    assert eng.vad_for_whisper is whisper_vad
    assert (eng.load_kw["vad_model"] == firered) is has_vad_model
    assert eng.load_kw["aligner_path"] is not None      # 預設開對齊


def test_select_f16_writes_settings_and_checks_the_f16_file(scratch, monkeypatch):
    """選 F16：設定鍵正確、下拉回填正確、就緒判斷看的是 F16 那個檔而不是 Q8。"""
    import downloader
    scratch(backend="openvino")
    be = wb.WebBackend()

    labels = [m["label"] for c in be.get_model_options()["cores"] for m in c["models"]]
    assert F16_LABEL in labels

    res = be.set_model("Qwen", F16_LABEL)
    s = json.loads(scratch.path.read_text(encoding="utf-8"))
    assert (s["backend"], s["crisp_model"], s["crisp_qwen_quant"]) == ("crispasr", "qwen3", "f16")
    assert res["model"] == F16_LABEL
    assert "4.7GB" in res["message"]                    # 體積／顯存提醒有帶出來
    assert be._current_selection() == ("Qwen", F16_LABEL)

    asked = []
    monkeypatch.setattr(downloader, "quick_check_crispasr", lambda *a, **k: True)
    monkeypatch.setattr(downloader, "quick_check_qwen3_asr_gguf",
                        lambda d, q="q8", *a, **k: asked.append(q) or q == "f16")
    assert be.selected_model_present() is True
    assert asked == ["f16"]


@pytest.mark.parametrize("attrs,supported", [
    ({"_backend": "qwen3-1.7b", "_fa_blocked": False}, True),
    ({"_backend": "whisper",    "_fa_blocked": False}, False),   # Breeze：對齊器不生效
    ({"_backend": "qwen3-1.7b", "_fa_blocked": True},  False),   # TEA：與對齊器不相容
    ({}, True),                                                   # OpenVINO／chatllm
])
def test_status_tells_the_ui_whether_alignment_applies(scratch, attrs, supported):
    """「時間軸對齊」開關對目前模型沒作用時，狀態要講出來讓介面停用它。"""
    be = wb.WebBackend()
    assert be.get_status()["alignSupported"] is True      # 尚未載入：先不擋
    eng = _Ready()
    for k, v in attrs.items():
        setattr(eng, k, v)
    be.engine = eng
    st = be.get_status()
    assert st["alignSupported"] is supported
    assert bool(st["alignNote"]) is (not supported)


def test_cli_load_fails_when_the_selected_core_fell_back(scratch, monkeypatch):
    """CLI 載入：選定的核心退回 CPU 時要失敗，不能拿另一個模型的結果回成功。"""
    import cli_mode
    scratch(backend="crispasr")
    be = wb.WebBackend()

    def boom():
        raise RuntimeError("crispasr 起不來")
    monkeypatch.setattr(be, "_load_crispasr", boom)
    monkeypatch.setattr(be, "_load_openvino", lambda: setattr(be, "engine", _Ready()))

    with pytest.raises(RuntimeError, match="crispasr 核心載入失敗"):
        cli_mode._load(be)


def test_line_width_setting_reaches_the_splitter(scratch):
    """設定頁的「英文／韓文單行字數上限」要真的改到分行器，且夾在合理範圍。"""
    import subtitle_lines as sl
    be = wb.WebBackend()
    default = sl.DEFAULT_SPACED_CHARS
    try:
        assert be.get_settings()["lineChars"] == default
        be.set_settings({"lineChars": 60})
        assert sl.MAX_SPACED_CHARS == 60 and be.get_settings()["lineChars"] == 60
        # 49 個字元的句子：上限 42 時要分兩行，調到 60 就是一行
        words = "I hope he brings that stuff back before the game.".split()
        items = [(w, k * 0.3, k * 0.3 + 0.3) for k, w in enumerate(words)]
        got = [t for (_s, _e, t, _spk) in sl.spaced_subtitle_lines(items)]
        assert got == [" ".join(words)]
        sl.MAX_SPACED_CHARS = default
        assert len(sl.spaced_subtitle_lines(items)) == 2
        be.set_settings({"lineChars": 500})
        assert sl.MAX_SPACED_CHARS == 90          # 夾到上限
        be.set_settings({"lineChars": "abc"})
        assert sl.MAX_SPACED_CHARS == default      # 不合法 → 預設
    finally:
        sl.MAX_SPACED_CHARS = default


# ══════════════════════════ 整合（slow）══════════════════════════

def _http(port: int, path: str, body=None, ctype="application/json", timeout=600):
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                 headers={"Content-Type": ctype} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _multipart(fields: dict, filename: str, blob: bytes):
    b = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'
                     .encode("utf-8"))
    parts.append(f'--{b}\r\nContent-Disposition: form-data; name="file"; '
                 f'filename="{filename}"\r\nContent-Type: application/octet-stream\r\n\r\n'
                 .encode("utf-8") + blob + b"\r\n")
    parts.append(f"--{b}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={b}"


@pytest.mark.slow
def test_f16_selected_and_loaded_through_the_http_api(scratch):
    """走前端的同一組端點選用並載入 F16，確認真的是 crispasr 在跑。"""
    import downloader
    from webview_server import WebViewServer
    if not NEWS.exists():
        pytest.skip("缺測試音檔 test_audio/News.mp3")
    if not downloader.quick_check_qwen3_asr_gguf(MODEL_DIR, "f16"):
        pytest.skip("本機沒有 F16 模型（4.7GB，不在測試裡自動下載）")
    if not (CRISP_DIR / "crispasr.exe").exists():
        pytest.skip("缺 crispasr 核心")
    scratch(backend="openvino", model_dir=str(MODEL_DIR), crispasr_dir=str(CRISP_DIR))

    srv = WebViewServer(port=0)
    port = srv.start()
    try:
        opts = _http(port, "/api/model-options")
        assert F16_LABEL in [m["label"] for c in opts["cores"] for m in c["models"]]

        sel = _http(port, "/api/model", {"core": "Qwen", "model": F16_LABEL})
        assert sel["ok"] and sel["model"] == F16_LABEL
        assert _http(port, "/api/status")["selectedReady"] is True

        assert _http(port, "/api/load", {})["ok"]
        deadline = time.time() + 300
        st = {}
        while time.time() < deadline:
            st = _http(port, "/api/status")
            if not st["loading"] and (st["modelReady"] or st["error"]):
                break
            time.sleep(0.5)
        assert st.get("modelReady"), f"載入逾時或失敗：{st}"
        assert not st["error"], st["error"]
        assert st["activeBackend"] == "crispasr"
        eng = srv.backend.engine
        assert Path(eng._model_path).name == "qwen3-asr-1.7b-f16.gguf"

        body, ctype = _multipart({"language": "Chinese", "align": "1"},
                                 NEWS.name, NEWS.read_bytes())
        res = _http(port, "/api/transcribe", body, ctype)
        text = "".join(s["text"] for s in res["segments"])
        assert len(text) > 300, text
        assert "CRISPASR" in res["stats"]["backend"]
        assert res["stats"]["aligned"] is True
        assert res["stats"]["vad"]                      # 預設 auto → Qwen 掛 FireRed
        starts = [s["start"] for s in res["segments"]]
        assert starts == sorted(starts)
        print(f"\n[F16 via HTTP] {len(res['segments'])} 段 {len(text)} 字  {res['stats']}")
    finally:
        srv.stop()
