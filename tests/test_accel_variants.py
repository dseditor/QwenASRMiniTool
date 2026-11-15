"""推理加速版本：RTX Spark（Windows ARM＋NVIDIA）專用編譯版。

CrispASR 官方沒有 Windows ARM64 的 CUDA 版；Spark 版是我們自己編的，放在本專案
的 Release。這裡驗證：
  • 只有 ARM64＋NVIDIA 的機器會被推薦／允許選 Spark 版；
  • ARM 機器上 x64 的 CUDA 版標成不適用；
  • Spark 版從自己的網址下載，官方版仍走 CrispASR 的 release；
  • 連核心一起附的安裝包（核心資料夾已有對得上版本的核心）不會被重抓成別的版本。
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as core                 # noqa: E402
import downloader as dl            # noqa: E402
import webview_backend as wb       # noqa: E402


def _hw(arch="x86_64", vendor="nvidia", name="NVIDIA GeForce RTX 5090", driver=581.0):
    gpus = [{"vendor": vendor, "name": name, "vram_mb": 0, "driver": ""}] if vendor else []
    return {"gpus": gpus, "vendors": [vendor] if vendor else [],
            "nvidia_driver": driver if vendor == "nvidia" else None,
            "has_discrete": vendor == "nvidia", "arch": arch, "error": None}


def _opts(info):
    return {o["key"]: o for o in info["options"]}


# ══════════════════════════ 推薦 ══════════════════════════

def test_spark_is_recommended_on_arm64_nvidia():
    info = dl.recommend_crispasr_variant(
        _hw("arm64", name="NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)", driver=617.69))
    assert info["recommended"] == "cuda-spark"
    o = _opts(info)
    assert o["cuda-spark"]["suitable"] and o["cuda-spark"]["recommended"]
    assert not o["cuda12"]["suitable"] and not o["cuda13"]["suitable"]
    assert "Windows ARM" in o["cuda13"]["note"]
    assert o["vulkan"]["suitable"] and "模擬" in o["vulkan"]["note"]   # 能選，但據實標示


def test_spark_is_not_offered_on_x64():
    info = dl.recommend_crispasr_variant(_hw("x86_64"))
    o = _opts(info)
    assert info["recommended"] == "vulkan"
    assert not o["cuda-spark"]["suitable"] and "Windows ARM" in o["cuda-spark"]["note"]
    assert o["cuda13"]["suitable"]                                    # 既有行為不變


def test_spark_needs_nvidia_even_on_arm64():
    info = dl.recommend_crispasr_variant(_hw("arm64", vendor="other", name="Qualcomm Adreno"))
    assert info["recommended"] == "vulkan"
    assert not _opts(info)["cuda-spark"]["suitable"]


def test_spark_variant_runs_on_the_cuda_backend():
    assert dl.crispasr_gpu_backend("cuda-spark") == "cuda"
    meta = dl.crispasr_variants()["cuda-spark"]
    assert meta["arch"] == "arm64" and "arm64" in meta["zip"]


def test_host_arch_is_one_of_the_two_known_values():
    assert dl.host_arch() in ("x86_64", "arm64")
    assert dl.detect_hardware()["arch"] == dl.host_arch()


# ══════════════════════════ 下載來源 ══════════════════════════

@pytest.mark.parametrize("variant, host", [
    ("cuda-spark", "github.com/dseditor/QwenASRMiniTool/releases/download/"),
    ("vulkan", "github.com/CrispStrobe/CrispASR/releases/download/"),
])
def test_each_variant_downloads_from_its_own_source(tmp_path, monkeypatch, variant, host):
    seen = []

    def fake_download(url, dest, progress_cb=None):
        seen.append(url)
        with zipfile.ZipFile(dest, "w") as z:
            z.writestr("pkg/crispasr.exe", b"MZ")
            z.writestr("pkg/cublas64_13.dll", b"dll")

    monkeypatch.setattr(dl, "_download_file", fake_download)
    core_dir = tmp_path / "crispasr"
    core_dir.mkdir()
    (core_dir / "ggml-vulkan.dll").write_bytes(b"old")       # 舊版本的 DLL 要清掉
    (core_dir / "model.gguf").write_bytes(b"model")          # 模型檔要留著
    dl.download_crispasr_core(core_dir, variant=variant)

    assert len(seen) == 1 and host in seen[0]
    assert seen[0].endswith(dl.crispasr_variants()[variant]["zip"])
    assert (core_dir / "crispasr.exe").exists() and (core_dir / "cublas64_13.dll").exists()
    assert not (core_dir / "ggml-vulkan.dll").exists() and (core_dir / "model.gguf").exists()
    assert dl.installed_crispasr_info(core_dir) == {"version": dl._CRISPASR_VERSION,
                                                    "variant": variant}


# ══════════════════════════ 連核心一起附的安裝包 ══════════════════════════

@pytest.fixture
def backend(tmp_path, monkeypatch):
    f = tmp_path / "settings.json"
    core_dir = tmp_path / "crispasr"
    core_dir.mkdir()
    f.write_text(json.dumps({"crispasr_dir": str(core_dir)}), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_FILE", f)
    # 這台測試機的偵測結果是 x64 → 推薦 Vulkan；用來確認「已附的核心」優先於偵測
    monkeypatch.setattr(dl, "recommend_crispasr_variant",
                        lambda hw=None: {"recommended": "vulkan"})
    return wb.WebBackend(), core_dir, f


def test_bundled_core_is_adopted_instead_of_the_detected_one(backend):
    be, core_dir, f = backend
    (core_dir / "crispasr.exe").write_bytes(b"MZ")
    dl._write_crispasr_marker(core_dir, "cuda-spark")
    assert be._crisp_variant() == "cuda-spark"
    assert json.loads(f.read_text(encoding="utf-8"))["crisp_variant"] == "cuda-spark"
    assert dl.quick_check_crispasr(core_dir, "cuda-spark")           # 不會觸發重新下載


def test_outdated_bundled_core_falls_back_to_detection(backend):
    be, core_dir, _f = backend
    (core_dir / dl._CRISPASR_MARKER).write_text(
        json.dumps({"version": "0.8.44", "variant": "cuda-spark"}), encoding="utf-8")
    assert be._crisp_variant() == "vulkan"


def test_explicit_choice_wins_over_the_bundled_core(backend):
    be, core_dir, f = backend
    dl._write_crispasr_marker(core_dir, "cuda-spark")
    cur = json.loads(f.read_text(encoding="utf-8"))
    cur["crisp_variant"] = "cpu"
    f.write_text(json.dumps(cur), encoding="utf-8")
    assert be._crisp_variant() == "cpu"


# ══════════════════════════ Windows ARM：只開放 CrispASR ══════════════════════════

@pytest.fixture
def arm_backend(tmp_path, monkeypatch):
    """設定檔記著 OpenVINO（像是整個資料夾從 x64 機器搬過來）的 ARM 機器。"""
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"backend": "openvino", "cpu_model_size": "0.6B",
                             "crisp_variant": "cuda-spark"}), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_FILE", f)
    monkeypatch.setattr(wb, "crisp_only", lambda: True)
    monkeypatch.setattr(dl, "detect_hardware", lambda: _hw("arm64"))
    return wb.WebBackend(), f


def test_arm_moves_a_foreign_backend_back_to_crispasr(arm_backend):
    be, f = arm_backend
    cur = json.loads(f.read_text(encoding="utf-8"))
    assert (cur["backend"], cur["crisp_model"], cur["crisp_qwen_quant"]) == ("crispasr", "qwen3", "q8")
    assert be._persisted_backend() == "crispasr"


def test_arm_lists_only_crispasr_models(arm_backend):
    be, _f = arm_backend
    models = [m for c in be.get_model_options()["cores"] for m in c["models"]]
    assert models and {m["backend"] for m in models} == {"crispasr"}
    health = be.health_check()
    assert [c["backend"] for c in health["cores"]] == ["crispasr"]


def test_arm_refuses_to_select_the_other_engines(arm_backend):
    be, f = arm_backend
    res = be.set_model("Qwen", "Qwen3-ASR-0.6B")
    assert res["ok"] is False and "Windows ARM" in res["message"]
    assert be.set_backend(0)["ok"] is False                           # 0 = OpenVINO
    assert json.loads(f.read_text(encoding="utf-8"))["backend"] == "crispasr"
    assert be.set_model("Qwen", "Qwen3-ASR-1.7B Q8 (CRISPASR)")["ok"] is True


def test_arm_profiles_never_suggest_openvino(arm_backend, monkeypatch):
    be, _f = arm_backend
    monkeypatch.setattr(be, "_hw_tier", lambda: ("cpu", "no gpu"))    # 純 CPU 檔位原本會給 OpenVINO
    profs = be.get_basic_profiles()["profiles"]
    assert len(profs) == 4 and all("OpenVINO" not in p["arch"] for p in profs)


def test_x64_keeps_all_engines(backend):
    be, _core_dir, _f = backend
    backends = {m["backend"] for c in be.get_model_options()["cores"] for m in c["models"]}
    assert "openvino" in backends and "crispasr" in backends


# ══════════════════════════ 自動更新挑套件 ══════════════════════════

def test_updater_never_picks_another_editions_package(monkeypatch):
    import updater
    monkeypatch.setattr(updater, "exe_name", lambda: "QwenASR-WebView.exe")
    assets = [{"name": "QwenASR-WebView-Spark_2.1.0.7z", "browser_download_url": "spark"},
              {"name": "crispasr-windows-arm64-cuda13-spark.zip", "browser_download_url": "core"},
              {"name": "QwenASR-WebView_2.1.0.7z", "browser_download_url": "x64"}]
    assert updater._pick_asset(assets) == ("QwenASR-WebView_2.1.0.7z", "x64")
    # 沒有自己的套件 → 維持舊行為（由 is_migration_package 擋下）
    assert updater._pick_asset(assets[:1])[1] == "spark"
