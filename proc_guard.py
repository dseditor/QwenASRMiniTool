"""proc_guard.py — 行程清理：父行程結束 = 連帶終止所有子程序（Windows）

問題背景
--------
本專案的 GPU 推理核心會以 subprocess 衍生子程序：
  • crispasr.exe（CrispASR / Whisper Vulkan）
  • chatllm main.exe（ForcedAligner 時間軸對齊）
Windows 不像 POSIX 會在父行程死亡時連帶回收子程序——若使用者在「辨識
到一半」關窗、或主行程崩潰／被工作管理員強制結束，這些子程序就成為孤兒
殘留（仍佔 GPU / 記憶體）。

解法
----
把主行程綁進一個 **KILL_ON_JOB_CLOSE 的 Windows Job Object**：之後衍生的
子程序會自動繼承同一個 Job；當主行程的最後一個 Job handle 關閉（即主行程
結束，含被強制結束／崩潰）時，OS 會自動終止 Job 內所有成員 → 子程序無一
倖免。`app.py`（CTk）與 `app_webview.py`（WebView）兩個進入點共用本模組。

注意：chatllm 在 webview 走的是 **in-process libchatllm.dll**（非子程序），
Job Object 管不到它——那一條由各進入點以 `os._exit(0)` 硬退出、交給 OS
回收（見各進入點關閉流程）。本模組只負責「真正的子程序」。
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

# 保留 job handle 的模組級引用：若被 GC 回收，handle 提早關閉會「提早觸發
# kill」把自己與子程序一起殺掉。存在這裡確保其生命週期 == 行程生命週期。
_JOB_HANDLE = None


def setup_kill_on_close_job():
    """把目前行程加入一個 KILL_ON_JOB_CLOSE 的 Job Object（冪等）。

    回傳 job handle（成功）或 None（非 Windows／失敗，靜默降級）。重複呼叫
    只會建立一次；handle 由模組級變數持有，呼叫端無須自行保管。
    """
    global _JOB_HANDLE
    if _JOB_HANDLE is not None:
        return _JOB_HANDLE
    if sys.platform != "win32":
        return None

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x00000800   # 容許瀏覽器 helper 自請脫離
    JobObjectExtendedLimitInformation = 9

    class _BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class _EXTENDED(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC),
            ("IoInfo", _IO),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        k32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = _EXTENDED()
        info.BasicLimitInformation.LimitFlags = (
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK)
        if not k32.SetInformationJobObject(
                job, JobObjectExtendedLimitInformation,
                ctypes.byref(info), ctypes.sizeof(info)):
            return None
        # Win8+ 允許巢狀 Job；即使本行程已在別的 Job 內也多半成功。失敗則降級。
        if not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            return None
        _JOB_HANDLE = job
        return job
    except Exception:
        return None
