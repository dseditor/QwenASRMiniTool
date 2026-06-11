"""fa_aligner.py — chatllm 原生 ForcedAligner（後端無關，無需 PyTorch）

封裝「字級時間軸對齊」：以 subprocess 呼叫 chatllm `main.exe`，給定
「參考文字 + 音訊」即可輸出字級毫秒時間軸（format json）。

設計重點：
  • 對齊本身與 ASR 後端無關——OpenVINO（CPU）與 chatllm（Vulkan GPU）
    兩種引擎共用同一份邏輯，避免 _align_chunk 在兩處分歧。
  • CPU 用 `-ngl 0`，GPU 用 `-ngl {device_id}:all`。
  • 失敗、無輸出時一律回傳 []（呼叫端據此退回比例估算）。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

# chatllm 來源檔名沿用上游拼字（foced 非 forced），與 downloader 一致。
FA_BIN_NAME = "qwen3-focedaligner-0.6b.bin"

# Windows：隱藏子程序主控台視窗
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _startup_info():
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0  # SW_HIDE
        return si
    return None


def _base_dir() -> Path:
    """App 根目錄（凍結時為 EXE 旁，否則為原始碼目錄）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def find_chatllm_dir() -> Path | None:
    """定位含 main.exe 的 chatllm 目錄。

    與 app.py 的解析順序一致：<app>/chatllm → chatllmtest 備援。
    """
    base = _base_dir()
    for cand in (
        base / "chatllm",
        base / "chatllmtest" / "chatllm_win_x64" / "bin",
        base,
    ):
        if (cand / "main.exe").exists():
            return cand
    return None


class ChatLLMAligner:
    """chatllm 原生 ForcedAligner 的輕量包裝。

    用法
    ----
    fa = ChatLLMAligner(fa_dir=model_dir)        # CPU（-ngl 0）
    if fa.load(cb=status):                        # 驗證 .bin 可用
        items = fa.align_chunk(wav, ref_text)     # [(word, start_s, end_s), ...]
    """

    FA_BIN_NAME = FA_BIN_NAME

    def __init__(
        self,
        fa_dir: str | Path | None,
        chatllm_dir: str | Path | None = None,
        n_gpu_layers: int = 0,
        device_id: int = 0,
    ):
        self._fa_dir = Path(fa_dir) if fa_dir else None
        self._chatllm_dir = (
            Path(chatllm_dir) if chatllm_dir else find_chatllm_dir()
        )
        self._n_gpu_layers = n_gpu_layers
        self._device_id = device_id
        self._fa_bin: Path | None = None
        self.ready = False

    # ── 偵測 / 驗證 ────────────────────────────────────────────────────
    def fa_bin_path(self) -> Path | None:
        """FA .bin 應在的位置（fa_dir / FA_BIN_NAME）。"""
        if self._fa_dir is None:
            return None
        return self._fa_dir / self.FA_BIN_NAME

    def load(self, cb: Callable[[str], None] | None = None) -> bool:
        """驗證 chatllm FA .bin 可用。成功回傳 True 並設定 ready/_fa_bin。

        實際對齊在 align_chunk() 以 subprocess 完成；此處僅以 `--show`
        確認模型確為 ForcedAligner，避免日後拿到錯誤模型。
        """
        def _s(msg: str):
            if cb:
                cb(msg)

        self.ready = False
        self._fa_bin = None

        fa_path = self.fa_bin_path()
        if fa_path is None or not fa_path.exists():
            return False
        if self._chatllm_dir is None:
            _s("⚠ 找不到 chatllm（main.exe），時間軸對齊改用比例估算")
            return False

        try:
            _s("驗證時間軸對齊模型（chatllm ForcedAligner）…")
            exe = self._chatllm_dir / "main.exe"
            r = subprocess.run(
                [str(exe), "-m", str(fa_path), "-ngl", "0",
                 "--hide_banner", "--show"],
                capture_output=True, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                timeout=30, cwd=str(self._chatllm_dir),
                creationflags=_CREATE_NO_WINDOW, startupinfo=_startup_info(),
            )
            out = r.stdout + r.stderr
            if "ForcedAligner" not in out:
                _s("⚠ FA 模型驗證失敗，改用比例估算")
                return False
            self._fa_bin = fa_path
            self.ready = True
            _s("時間軸對齊模型就緒（chatllm，無需 PyTorch）")
            return True
        except Exception as _e:
            _s(f"⚠ ForcedAligner 載入失敗（{_e}），改用比例估算")
            return False

    # ── 對齊 ──────────────────────────────────────────────────────────
    def align_chunk(
        self,
        wav_path: str,
        ref_text: str,
        language: str = "Chinese",
    ) -> list[tuple[str, float, float]]:
        """對齊「參考文字 + 音訊」，回傳字級 [(word, start_s, end_s), ...]。

        失敗或無輸出時回傳 []。
        """
        if not self._fa_bin or self._chatllm_dir is None:
            return []

        gpu_args = (["-ngl", f"{self._device_id}:all"]
                    if self._n_gpu_layers > 0 else ["-ngl", "0"])
        exe = self._chatllm_dir / "main.exe"
        prompt = f"{ref_text}{{{{audio:{wav_path}}}}}"
        cmd = [
            str(exe), "-m", str(self._fa_bin), *gpu_args, "--hide_banner",
            "--multimedia_file_tags", "{{", "}}",
            "-p", prompt,
            "--set", "format", "json",
            "--set", "language", language,
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                timeout=120, cwd=str(self._chatllm_dir),
                creationflags=_CREATE_NO_WINDOW, startupinfo=_startup_info(),
            )
        except Exception:
            return []

        out = r.stdout + r.stderr
        # main.exe 末尾會附 timings 文字，需先切出 JSON 陣列 [...]
        i = out.find("[")
        j = out.rfind("]")
        if i < 0 or j <= i:
            return []
        try:
            data = json.loads(out[i:j + 1])
        except (ValueError, json.JSONDecodeError):
            return []

        items: list[tuple[str, float, float]] = []
        for d in data:
            try:
                w = str(d["text"])
                s = float(d["start"]) / 1000.0   # 毫秒 → 秒
                e = float(d["end"]) / 1000.0
            except (KeyError, TypeError, ValueError):
                continue
            if w.strip():
                items.append((w, s, e))
        return items
