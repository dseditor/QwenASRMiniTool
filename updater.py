"""updater.py — 線上自動更新（GitHub Releases）

機制總覽
========
1. check_latest()      向 GitHub Releases API 查最新版本，與本機 __version__ 比對。
2. download_asset()    下載該 Release 的 ZIP（或 7z）資產到暫存資料夾，支援進度回報。
3. apply_update()      解壓 → 找出內含 QwenASR.exe 的根目錄 → 產生英文 helper .bat：
                       等待本程式結束 → xcopy 疊加覆蓋安裝目錄 → 重新啟動 → 自我刪除。

設計重點
========
* 執行中的 QwenASR.exe / _internal 會被 Windows 鎖定，無法自我覆寫，
  因此覆蓋動作交給「程式結束後才執行」的 helper .bat。
* 採 xcopy 疊加（非鏡像），只覆蓋 ZIP 內含的程式檔，
  保留使用者的 GPUModel / ov_models / venv-gpu / settings.json 等大型/個人資料。
* ZIP 用 Python 內建 zipfile 解壓（零外部依賴，對應使用者要求的「ZIP 套件」）。
  若資產為 .7z，嘗試使用安裝目錄旁的 7zr.exe / 7z.exe，找不到則提示手動更新。
* 版本字串可能帶後綴（如 1.0.2(20260222)、1.0.3_Vulkan_Branch），
  以正則擷取開頭的數字段比較，忽略括號/底線後綴。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import urllib.error
import urllib.request

try:
    # 重用 downloader 的 certifi SSL Context 與斷點續傳下載
    from downloader import _ssl_ctx, _download_file
except Exception:  # pragma: no cover - downloader 一定會一起打包
    import ssl

    def _ssl_ctx():  # type: ignore
        try:
            import certifi
            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return ssl.create_default_context()

    def _download_file(url, dest, progress_cb=None):  # type: ignore
        req = urllib.request.Request(url, headers={"User-Agent": "QwenASR-updater"})
        with urllib.request.urlopen(req, timeout=30, context=_ssl_ctx()) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            done = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 16)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if progress_cb and total:
                        progress_cb(done, total)

from version import __version__, GITHUB_API_LATEST, GITHUB_RELEASES_PAGE

_UA = "Mozilla/5.0 (compatible; QwenASR-updater)"


# ── 安裝目錄與執行檔 ──────────────────────────────────────────────────

def is_frozen() -> bool:
    """是否為 PyInstaller 凍結的 EXE（只有凍結版才能自我更新）。

    同時檢查 sys.frozen 與 sys._MEIPASS：正常凍結版兩者皆有，加 _MEIPASS
    作為防禦，避免極少數環境 sys.frozen 未設時誤判為開發模式。
    """
    return bool(getattr(sys, "frozen", False)) or hasattr(sys, "_MEIPASS")


def install_dir() -> Path:
    """回傳安裝根目錄（凍結版＝EXE 所在資料夾）。"""
    if is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).parent


def exe_name() -> str:
    """主執行檔名稱（凍結版＝自身；開發版回傳預設名供測試）。"""
    if is_frozen():
        return Path(sys.executable).name
    return "QwenASR.exe"


# ── 版本解析與比較 ────────────────────────────────────────────────────

_VER_RE = re.compile(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?")


def parse_version(tag: str) -> tuple[int, int, int]:
    """從 tag 擷取開頭的數字版本，回傳 (major, minor, patch)。

    例：'1.0.5' → (1,0,5)；'v1.2' → (1,2,0)；
        '1.0.2(20260222)' → (1,0,2)；'1.0.3_Vulkan_Branch' → (1,0,3)。
    解析失敗回傳 (0,0,0)。
    """
    if not tag:
        return (0, 0, 0)
    m = _VER_RE.search(tag.strip().lstrip("vV"))
    if not m:
        return (0, 0, 0)
    return (int(m.group(1)), int(m.group(2) or 0), int(m.group(3) or 0))


def is_newer(latest_tag: str, current: str = __version__) -> bool:
    """latest_tag 是否比 current 新。"""
    return parse_version(latest_tag) > parse_version(current)


# ── 查詢最新 Release ──────────────────────────────────────────────────

def _pick_asset(assets: list[dict]) -> tuple[str, str] | None:
    """從 release assets 選擇可更新的套件，優先 .zip，其次 .7z。

    回傳 (asset_name, download_url)，找不到則 None。
    """
    zips = [a for a in assets if a.get("name", "").lower().endswith(".zip")]
    if zips:
        return zips[0]["name"], zips[0]["browser_download_url"]
    sevenz = [a for a in assets if a.get("name", "").lower().endswith(".7z")]
    if sevenz:
        return sevenz[0]["name"], sevenz[0]["browser_download_url"]
    return None


def package_app_name(asset_name: str) -> str:
    """從資產檔名推出它裝的是哪個應用程式。

    兩條發行線的命名慣例一致，都是 `<應用名>_<版本>.<副檔名>`：
        QwenASR_1.0.9.zip            → "QwenASR"
        QwenASR-WebView_2.0.0.7z     → "QwenASR-WebView"
    取第一個底線前的部分即可（版本號本身不含底線）。
    """
    stem = Path(asset_name).stem            # 去副檔名（.7z/.zip）
    return stem.split("_", 1)[0]


def is_migration_package(asset_name: str) -> bool:
    """該資產是否屬於「另一個應用程式」（＝需要遷移而非就地更新）。

    2.0.0 起 WebView 取代 CTk 成為主線，兩者的執行檔名不同
    （QwenASR.exe ↔ QwenASR-WebView.exe）。若讓 CTk 的自動更新照常執行，
    apply_update 會把 WebView 的檔案疊進 CTk 的安裝夾、然後**重新啟動舊的
    QwenASR.exe** —— 使用者更新完看到的還是舊程式，資料夾卻多了一堆檔案。
    故此處先攔下來，改走「提示使用者手動下載」的遷移路徑。
    """
    if not asset_name:
        return False
    return package_app_name(asset_name).lower() != Path(exe_name()).stem.lower()


def check_latest(timeout: int = 15) -> dict:
    """查詢最新 Release，回傳資訊字典。

    回傳鍵：
        ok            (bool)  查詢是否成功
        error         (str)   失敗訊息（ok=False 時）
        current       (str)   本機版本
        latest_tag    (str)   遠端最新 tag
        latest_name   (str)   Release 標題
        body          (str)   更新說明（changelog）
        has_update    (bool)  遠端是否較新
        asset_name    (str)   可下載套件檔名（可能為 None）
        asset_url     (str)   套件下載網址（可能為 None）
        html_url      (str)   Release 頁面網址
    """
    info: dict = {
        "ok": False, "error": "", "current": __version__,
        "latest_tag": "", "latest_name": "", "body": "",
        "has_update": False, "asset_name": None, "asset_url": None,
        "html_url": GITHUB_RELEASES_PAGE,
        # 遷移模式：最新版屬於另一個應用程式（WebView 取代 CTk）→ 不自動更新
        "migration": False, "migration_note": "",
    }
    try:
        req = urllib.request.Request(
            GITHUB_API_LATEST,
            headers={"User-Agent": _UA, "Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=timeout, context=_ssl_ctx()) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        info["error"] = f"GitHub API 回應 {e.code}"
        return info
    except (urllib.error.URLError, OSError, ValueError) as e:
        info["error"] = f"連線失敗：{e}"
        return info

    tag = data.get("tag_name", "")
    info.update({
        "ok": True,
        "latest_tag": tag,
        "latest_name": data.get("name", "") or tag,
        "body": data.get("body", "") or "",
        "html_url": data.get("html_url", GITHUB_RELEASES_PAGE),
        "has_update": is_newer(tag),
    })
    picked = _pick_asset(data.get("assets", []))
    if picked:
        name, url = picked
        if is_migration_package(name):
            # 不同應用程式的套件 → 不提供自動更新，改請使用者手動下載安裝。
            # 刻意把 asset_url 留成 None：既有 UI 已有「無可自動更新的套件 →
            # 禁用『立即更新』」的分支，直接沿用那條已驗證過的路徑。
            info["migration"] = True
            info["migration_note"] = (
                f"這是全新的「{package_app_name(name)}」版本，不會覆蓋你現有的安裝。\n"
                f"請點「發行頁」下載 {name}，解壓到一個**新的資料夾**後直接執行；\n"
                f"舊版可保留，模型與設定不受影響。"
            )
        else:
            info["asset_name"], info["asset_url"] = name, url
    return info


# ── 下載套件 ──────────────────────────────────────────────────────────

def download_asset(url: str, asset_name: str, progress_cb=None) -> Path:
    """下載更新套件至暫存資料夾，回傳本機檔案路徑。

    progress_cb(done_bytes:int, total_bytes:int)
    """
    staging = Path(tempfile.gettempdir()) / "QwenASR_update"
    staging.mkdir(parents=True, exist_ok=True)
    dest = staging / asset_name
    if dest.exists():
        try:
            dest.unlink()
        except OSError:
            pass
    _download_file(url, dest, progress_cb=progress_cb)
    return dest


# ── 解壓 ──────────────────────────────────────────────────────────────

def _find_7z() -> Path | None:
    """尋找可用的 7z 解壓工具（安裝目錄旁 → 系統安裝路徑）。"""
    cands = [
        install_dir() / "7zr.exe",
        install_dir() / "7z.exe",
        Path(r"C:\Program Files\7-Zip\7z.exe"),
        Path(r"C:\Program Files (x86)\7-Zip\7z.exe"),
    ]
    for c in cands:
        if c.exists():
            return c
    return None


def _extract(archive: Path, dest: Path) -> None:
    """解壓 .zip（內建）或 .7z（外部 7z）到 dest。"""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif name.endswith(".7z"):
        seven = _find_7z()
        if not seven:
            raise RuntimeError(
                "此更新套件為 .7z 格式，但找不到 7z 解壓工具。\n"
                "請改用內含 ZIP 套件的版本，或手動下載更新。"
            )
        r = subprocess.run(
            [str(seven), "x", str(archive), f"-o{dest}", "-y"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"7z 解壓失敗：{r.stderr or r.stdout}")
    else:
        raise RuntimeError(f"不支援的套件格式：{archive.name}")


def _find_app_root(extracted: Path) -> Path:
    """在解壓內容中找出含主執行檔的資料夾（作為覆蓋來源）。"""
    target = exe_name().lower()
    # 直接命中
    if (extracted / exe_name()).exists():
        return extracted
    # 往下找最多三層
    for p in extracted.rglob("*"):
        if p.is_file() and p.name.lower() == target:
            return p.parent
    # 找不到自己的主執行檔 → 這包不是「本程式的新版」。以前這裡會退回「第一層
    # 唯一子資料夾」照樣覆蓋，結果是把別的應用程式的檔案疊進來、再重啟舊 exe。
    # 改為明確拒絕：縱深防禦，即使前面的 is_migration_package 判斷漏掉也擋得住。
    raise RuntimeError(
        f"這個更新套件裡沒有 {exe_name()}，看起來是另一個版本的程式。\n"
        "為避免覆蓋你現有的安裝，已停止自動更新——請到發行頁手動下載。"
    )


# ── 套用更新（產生 helper bat 後結束本程式）──────────────────────────

def apply_update(archive: Path, relaunch: bool = True) -> Path:
    """解壓套件並產生覆蓋/重啟用的 helper .bat（全英文）。

    回傳 helper .bat 路徑。呼叫端應在啟動此 bat 後立即關閉本程式，
    讓 bat 能在程式釋放檔案鎖後完成 xcopy 覆蓋與重啟。
    """
    if not is_frozen():
        raise RuntimeError("開發模式（非 EXE）不支援自我更新，請使用 build.bat 重新編譯。")

    staging = archive.parent
    extracted = staging / "extracted"
    if extracted.exists():
        # 清掉上一次殘留
        import shutil
        shutil.rmtree(extracted, ignore_errors=True)
    _extract(archive, extracted)

    src_root = _find_app_root(extracted)
    target = install_dir()
    exe = exe_name()
    log = staging / "update_log.txt"
    bat = staging / "apply_update.bat"

    # helper .bat 全英文（CLAUDE.md 規範：BAT 不可含中文，避免 CP950 亂碼）
    relaunch_line = (
        f'start "" "{target / exe}"' if relaunch else "REM relaunch disabled"
    )
    bat_text = f"""@echo off
REM ===== QwenASR auto-update applier (generated) =====
chcp 437 > nul
setlocal
set "LOG={log}"
echo [%date% %time%] update started > "%LOG%"

REM 1) Wait until the running app releases its file locks
echo Waiting for {exe} to close...
:waitloop
tasklist /FI "IMAGENAME eq {exe}" 2>nul | find /I "{exe}" >nul
if not errorlevel 1 (
    ping -n 2 127.0.0.1 >nul
    goto waitloop
)

REM 2) Overlay-copy new files over the install dir (keeps models / settings)
REM    No trailing backslash on the destination: "...\" would escape the
REM    closing quote in cmd. /I treats the destination as a directory.
echo Copying new files... >> "%LOG%"
xcopy "{src_root}\\*" "{target}" /E /I /Y /C >> "%LOG%" 2>&1

REM 3) Relaunch the updated app
{relaunch_line}

REM 4) Clean up staging (best-effort; ignore errors)
rmdir /S /Q "{extracted}" >nul 2>&1
del "{archive}" >nul 2>&1

REM 5) Self-delete
(goto) 2>nul & del "%~f0"
"""
    bat.write_text(bat_text, encoding="ascii")
    return bat


def launch_and_exit(bat: Path) -> None:
    """以分離的視窗啟動 helper bat（不阻塞），呼叫端隨後關閉主程式。"""
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_CONSOLE  # type: ignore[attr-defined]
    subprocess.Popen(
        ["cmd", "/c", str(bat)],
        cwd=str(bat.parent),
        creationflags=creationflags,
        close_fds=True,
    )


# ── CLI（手動測試用）──────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"目前版本：{__version__}")
    print("查詢最新 Release…")
    info = check_latest()
    if not info["ok"]:
        print(f"查詢失敗：{info['error']}")
        sys.exit(1)
    print(f"最新版本：{info['latest_tag']}（{info['latest_name']}）")
    print(f"有更新：{info['has_update']}")
    print(f"套件：{info['asset_name']}")
    print(f"網址：{info['asset_url']}")
