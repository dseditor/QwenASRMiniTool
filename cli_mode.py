"""cli_mode.py — 無頭命令列模式（給外部 Agent／腳本使用）

為什麼能這麼薄
--------------
`webview_backend.WebBackend` 本來就是為「無 UI」設計的那一層：
  • `_load_worker()` 是**完全同步**的，且缺模型會自動下載
  • `transcribe(opts)` 直接回 `{"segments": [...], "srtPath": ...}`
所以 CLI 不需要 [[cli-mode-plan]] 當初評估的引擎抽離重構 —— WebView EXE 早就
把 app.py 整包進去了（`webview_backend` 第一行就 `import app as core`），
再為了「避開 tkinter」而重構沒有意義。

輸出契約（Agent 友善）
----------------------
  • **stdout 只放結果**：`--json` 時是一份可直接 `json.loads` 的乾淨 JSON。
    載入訊息、進度、警告一律走 **stderr**（連引擎內部的 print 也被導去 stderr，
    見 `_stdout_to_stderr`），避免污染 stdout。
  • **退出碼**：0 成功／1 失敗／2 參數錯誤／130 使用者中斷。
  • 失敗時 stderr 印人話錯誤，`--json` 模式另在 stdout 給 `{"ok": false, "error": …}`。

子命令
------
  transcribe <音檔…>   轉錄，輸出 SRT／TXT／JSON
  apply <edited.json>  把（Agent 校正過的）text 依 index 寫回，**保留原時間軸**
  status               核心／模型／加速版本就緒狀況
  profiles             可用的用途 profile 清單

`apply` 存在的理由：讓 Agent 修字幕時「只准改 text」。若直接把整份 SRT 交給
模型重寫，時間戳很容易被順手改壞且難以察覺；改成回填 index → 時間軸在結構上
不可能被破壞。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

# 這些子命令名稱同時是 app_webview 判斷「要不要走無頭模式」的依據
SUBCOMMANDS = ("transcribe", "apply", "status", "profiles")


# ── stdout 保護 ────────────────────────────────────────────────────────
#   引擎與下載器內部有不少 print()，若直接落到 stdout 會讓 Agent 的
#   json.loads 炸掉。載入／轉錄期間把 sys.stdout 換成 stderr，最後再用
#   保存下來的「真 stdout」寫結果。
@contextlib.contextmanager
def _stdout_to_stderr():
    real = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield real
    finally:
        sys.stdout = real


def _log(msg: str):
    print(msg, file=sys.stderr, flush=True)


# ── 暫時性設定覆寫 ─────────────────────────────────────────────────────
#   --profile / --core-variant 這類「只想影響這一次執行」的選項，不應該偷改
#   使用者 GUI 的持久化設定。作法：把 settings.json 複製到暫存檔，改指
#   app.SETTINGS_FILE 到那份副本 —— WebBackend 底下所有讀寫設定的程式碼都走
#   同一個常數，故一次改點全體生效，且真正的 settings.json 完全不被碰到。
_ORIG_SETTINGS: Path | None = None      # 被暫存副本取代前的真實設定檔路徑


def _use_scratch_settings(core) -> Path:
    global _ORIG_SETTINGS
    real = Path(core.SETTINGS_FILE)
    _ORIG_SETTINGS = real
    tmp = Path(tempfile.mkdtemp(prefix="qwenasr-cli-")) / "settings.json"
    try:
        if real.exists():
            shutil.copyfile(real, tmp)
    except Exception:
        pass
    core.SETTINGS_FILE = tmp
    return tmp


def _make_backend(args):
    """建好 WebBackend 並（必要時）套用暫時設定，回傳 (backend, core)。"""
    import app as core
    from webview_backend import WebBackend

    if getattr(args, "profile", None) or getattr(args, "persist", False) is False:
        # 沒有 --persist 就一律用暫存設定副本，確保 CLI 不會改到 GUI 的設定。
        _use_scratch_settings(core)

    be = WebBackend()                      # on_event=None → 不推事件、全靜音
    if getattr(args, "profile", None):
        prof = next((p for p in be.get_basic_profiles()["profiles"]
                     if p["key"] == args.profile), None)
        if prof is None:
            raise SystemExit(f"未知的 profile：{args.profile}（用 `profiles` 子命令查看）")
        be.set_model(prof["core"], prof["model"])
        _log(f"[profile] {args.profile} → {prof['core']} · {prof['model']}")
    return be, core


def _load(be):
    """同步載入模型（缺檔會自動下載，進度走 stderr）。"""
    _log("[load] 載入模型中…（首次使用會自動下載）")
    be._load_worker()
    if not getattr(be.engine, "ready", False):
        raise RuntimeError(be._load_err or "模型載入失敗")
    _log(f"[load] 就緒：{be.get_status()['backend']}")


# ══════════════════════════════════════════════════════════════════════
# transcribe
# ══════════════════════════════════════════════════════════════════════
def _segments_payload(res: dict) -> list[dict]:
    """WebBackend 的 segments → CLI 對外契約（穩定欄位名，含 1-based index）。"""
    out = []
    for i, s in enumerate(res.get("segments") or [], 1):
        item = {"i": i,
                "start": round(float(s.get("start", 0.0)), 3),
                "end": round(float(s.get("end", 0.0)), 3),
                "text": s.get("text", "")}
        if s.get("speaker"):
            item["speaker"] = s["speaker"]
        out.append(item)
    return out


def _srt_ts(sec: float) -> str:
    sec = max(0.0, float(sec))
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    ms = int(round((sec - int(sec)) * 1000))
    if ms == 1000:                      # 進位溢出
        s, ms = s + 1, 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(segs: list[dict], dest: Path) -> Path:
    lines = []
    for n, s in enumerate(segs, 1):
        lines += [str(n),
                  f"{_srt_ts(s['start'])} --> {_srt_ts(s['end'])}",
                  s.get("text", ""), ""]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    return dest


def cmd_transcribe(args) -> int:
    be, _core = _make_backend(args)
    with _stdout_to_stderr() as out:
        _load(be)
        results, failed = [], 0
        for raw in args.audio:
            p = Path(raw)
            if not p.exists():
                _log(f"[錯誤] 找不到檔案：{p}")
                failed += 1
                results.append({"input": str(p), "ok": False, "error": "找不到檔案"})
                continue
            _log(f"[轉錄] {p.name}")
            try:
                res = be.transcribe({
                    "path": str(p),
                    "language": args.language or "",
                    "hint": args.hint or "",
                    "diarize": bool(args.diarize),
                    "nSpeakers": args.speakers,
                    "align": not args.no_align,
                }, progress_cb=lambda pct, msg: _log(f"  {pct:3d}%  {msg}"))
            except Exception as e:
                _log(f"[錯誤] {p.name}：{e}")
                failed += 1
                results.append({"input": str(p), "ok": False, "error": str(e)})
                continue
            segs = _segments_payload(res)
            item = {"input": str(p), "ok": True,
                    "srtPath": res.get("srtPath"), "segments": segs,
                    "text": "".join(s["text"] for s in segs)}
            # -o 只在單檔時有意義；多檔一律用引擎的預設輸出位置
            if args.output and len(args.audio) == 1:
                dest = Path(args.output)
                if args.format == "txt":
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text("\n".join(s["text"] for s in segs), encoding="utf-8")
                elif args.format == "json":
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(json.dumps(segs, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
                else:
                    _write_srt(segs, dest)
                item["outPath"] = str(dest)
            results.append(item)

    # ── 結果寫回真正的 stdout ──
    if args.json:
        payload = results[0] if len(results) == 1 else {"results": results}
        print(json.dumps(payload, ensure_ascii=False), file=out)
    else:
        for r in results:
            print(r.get("outPath") or r.get("srtPath") or f"FAILED\t{r['input']}", file=out)
    return 1 if failed else 0


# ══════════════════════════════════════════════════════════════════════
# apply —— 把校正過的 text 寫回，時間軸原封不動
# ══════════════════════════════════════════════════════════════════════
def cmd_apply(args) -> int:
    src = Path(args.edited)
    if not src.exists():
        _log(f"[錯誤] 找不到檔案：{src}")
        return 2
    data = json.loads(src.read_text(encoding="utf-8"))
    segs = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(segs, list) or not segs:
        _log("[錯誤] JSON 需為 segments 陣列，或含 segments 欄位的物件。")
        return 2

    base = None
    if args.base:                      # 有基準檔 → 只採用它的 text，時間軸以基準為準
        bdata = json.loads(Path(args.base).read_text(encoding="utf-8"))
        base = bdata.get("segments") if isinstance(bdata, dict) else bdata
        by_i = {int(s.get("i", n)): s for n, s in enumerate(segs, 1)}
        merged = []
        for n, b in enumerate(base, 1):
            i = int(b.get("i", n))
            merged.append({"i": i, "start": b["start"], "end": b["end"],
                           "text": (by_i.get(i) or b).get("text", ""),
                           **({"speaker": b["speaker"]} if b.get("speaker") else {})})
        segs = merged
    else:
        bad = [n for n, s in enumerate(segs, 1)
               if "start" not in s or "end" not in s]
        if bad:
            _log(f"[錯誤] 第 {bad[:5]} 段缺 start/end；"
                 f"請保留原本的時間欄位，或用 --base 指定原始 JSON。")
            return 2
        # 沒有基準檔時做一次健檢：時間軸退化（全零、全同、倒退）幾乎必然是
        # 校正端把時間欄位改壞了。這裡不擋（呼叫端可能真的自備時間軸），但要
        # 出聲——靜默產出一份時間全錯的字幕比報錯更難查。
        degenerate = all(float(s["end"]) <= float(s["start"]) for s in segs)
        backwards = any(float(segs[i]["start"]) < float(segs[i - 1]["start"])
                        for i in range(1, len(segs)))
        if degenerate or backwards:
            _log("[警告] 時間軸看起來不正常（"
                 + ("每段長度皆為 0；" if degenerate else "")
                 + ("段落時間倒退；" if backwards else "")
                 + "）。校正字幕時請加 --base <原始JSON>，"
                   "讓時間軸以原始檔為準。")

    dest = Path(args.output) if args.output else src.with_suffix(".srt")
    if args.format == "txt":
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(s.get("text", "") for s in segs), encoding="utf-8")
    else:
        _write_srt(segs, dest)
    if args.json:
        print(json.dumps({"ok": True, "outPath": str(dest), "count": len(segs)},
                         ensure_ascii=False))
    else:
        print(dest)
    return 0


# ══════════════════════════════════════════════════════════════════════
# status / profiles
# ══════════════════════════════════════════════════════════════════════
def cmd_status(args) -> int:
    be, core = _make_backend(args)
    with _stdout_to_stderr() as out:
        st = be.get_status()
        accel = be.get_accel()
        health = be.health_check()
        # status 是在「還沒載入任何模型」時查的，故 get_status()["backend"]
        # （= 實際載入中的核心）會是預設值而非使用者選的。這裡改報**已選定**
        # 的核心，才不會誤導 Agent。
        payload = {
            "appName": st.get("appName"), "version": st.get("version"),
            "backend": be._backend_label(st.get("backendKey"), st.get("backend")),
            "backendKey": st.get("backendKey"),
            "selectedReady": st.get("selectedReady"),
            "hasAnyModel": st.get("hasAnyModel"),
            "accel": {"selected": accel.get("selected"),
                      "recommended": accel.get("recommended"),
                      "installed": accel.get("installed"),
                      "needsDownload": accel.get("needsDownload")},
            "hardware": (accel.get("hardware") or {}).get("gpus"),
            "cores": [{"label": c["label"],
                       "items": [{"label": i["label"], "status": i["status"]}
                                 for i in c["items"]]}
                      for c in health.get("cores", [])],
            # 路徑診斷：模型下載到哪、設定從哪讀。凍結成 EXE 後這幾條路徑最容易
            # 與預期不符（BASE_DIR vs _MEIPASS），沒有它就只能猜。
            "paths": {
                "settings":     str(_ORIG_SETTINGS or getattr(core, "SETTINGS_FILE", "")),
                "settingsInUse": str(getattr(core, "SETTINGS_FILE", "")),
                "settingsExists": Path(_ORIG_SETTINGS or
                                       getattr(core, "SETTINGS_FILE", ".")).exists(),
                "modelDir":    str(be._model_dir()),
                "crispasrDir": str(be._crispasr_dir()),
                "baseDir":     str(getattr(core, "BASE_DIR", "")),
            },
        }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"{payload['appName']} {payload['version']}")
        print(f"已選核心：{payload['backend']}　模型就緒：{payload['selectedReady']}")
        print(f"加速：{payload['accel']['selected']}"
              f"（建議 {payload['accel']['recommended']}）")
    return 0


def cmd_profiles(args) -> int:
    be, _core = _make_backend(args)
    with _stdout_to_stderr() as out:
        d = be.get_basic_profiles()
        payload = {"tier": d["tier"], "hardware": d["hardware"],
                   "profiles": [{"key": p["key"], "title": p["title"],
                                 "desc": p["desc"], "note": p["note"],
                                 "core": p["core"], "model": p["model"],
                                 "present": p["present"]}
                                for p in d["profiles"]]}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(payload["hardware"])
        for p in payload["profiles"]:
            mark = "✓" if p["present"] else " "
            print(f"  [{mark}] {p['key']:6s} {p['title']}  →  {p['model']}")
    return 0


# ══════════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="QwenASR", add_help=True,
        description="聲音辨識小工具 — 無頭命令列模式（給 Agent／腳本使用）。"
                    "不帶子命令執行則開啟圖形介面。")
    ap.add_argument("--json", action="store_true",
                    help="stdout 輸出 JSON（進度一律走 stderr）")
    ap.add_argument("--persist", action="store_true",
                    help="允許本次執行改寫 settings.json（預設用暫存副本，"
                         "不影響圖形介面的設定）")

    # 讓 --json / --persist 放在子命令**前後皆可**（Agent 兩種寫法都很自然）。
    # 子命令這份用 SUPPRESS 當預設：沒給就不覆寫 namespace，於是頂層那份的值
    # 得以保留 —— 否則 argparse 會用子解析器的預設值蓋掉頂層已解析的結果。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="stdout 輸出 JSON")
    common.add_argument("--persist", action="store_true", default=argparse.SUPPRESS,
                        help="允許改寫 settings.json")

    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("transcribe", parents=[common], help="轉錄音檔／影片為字幕")
    t.add_argument("audio", nargs="+", help="音訊或影片檔路徑（可多個）")
    t.add_argument("-o", "--output", help="輸出檔路徑（僅單一輸入時有效）")
    t.add_argument("-f", "--format", choices=("srt", "txt", "json"), default="srt",
                   help="-o 的輸出格式（預設 srt）")
    t.add_argument("-l", "--language", default="",
                   help="語言，如 Chinese／English／Japanese；留空＝自動偵測")
    t.add_argument("--hint", default="", help="辨識提示（人名、術語、背景說明）")
    t.add_argument("--profile", choices=("zh", "tea", "ja", "twmix"),
                   help="依用途選模型（見 profiles 子命令）")
    t.add_argument("--diarize", action="store_true", help="啟用說話者分離")
    t.add_argument("--speakers", type=int, help="說話者人數（留空＝自動）")
    t.add_argument("--no-align", action="store_true", help="關閉字級時間軸對齊")
    t.set_defaults(func=cmd_transcribe)

    a = sub.add_parser("apply", parents=[common], help="把校正過的 segments JSON 寫回字幕（保留時間軸）")
    a.add_argument("edited", help="校正後的 JSON（segments 陣列或含 segments 的物件）")
    a.add_argument("-o", "--output", help="輸出路徑（預設與輸入同名的 .srt）")
    a.add_argument("-f", "--format", choices=("srt", "txt"), default="srt")
    a.add_argument("--base", help="原始 JSON；給了就以它的時間軸為準，"
                                  "只採用校正檔的 text（最安全）")
    a.set_defaults(func=cmd_apply)

    s = sub.add_parser("status", parents=[common], help="核心／模型／加速版本就緒狀況")
    s.set_defaults(func=cmd_status)

    p = sub.add_parser("profiles", parents=[common], help="列出用途 profile 與各自的建議模型")
    p.set_defaults(func=cmd_profiles)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _log("已中斷。")
        return 130
    except SystemExit as e:
        _log(str(e))
        return 2
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main())
