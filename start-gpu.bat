@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

REM ============================================================
REM   Qwen3 ASR GPU Launcher  (PyTorch version - maintenance mode)
REM   Uses PyTorch CUDA backend (no OpenVINO required)
REM   Model: GPUModel\Qwen3-ASR-1.7B
REM
REM   NOTE: This PyTorch launcher is in maintenance mode.
REM         Main GPU support has moved to chatllm_engine.py
REM         (Vulkan backend) inside QwenASR.exe.
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "GPU_MODEL_DIR=%SCRIPT_DIR%GPUModel"
set "ASR_MODEL_DIR=%GPU_MODEL_DIR%\Qwen3-ASR-1.7B"
set "VENV_DIR=%SCRIPT_DIR%venv-gpu"
set "APP_SCRIPT=%SCRIPT_DIR%app-gpu.py"
set "PYTHON_EXE=python"

REM ---- Clean up leftover temp files from previous runs -------
if exist "%SCRIPT_DIR%__sl_run__.bat" del "%SCRIPT_DIR%__sl_run__.bat" 2>nul
if exist "%SCRIPT_DIR%.tmp_ip"        del "%SCRIPT_DIR%.tmp_ip"        2>nul


REM ---- Check Python ------------------------------------------
python --version > nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found in PATH.
    echo          Please install Python 3.10+ from https://python.org
    echo          and ensure "Add Python to PATH" is checked.
    pause & exit /b 1
)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo  [OK] Python %PY_VER% found.
echo.

REM ---- Step 1/3: Choose environment --------------------------
echo  Step 1/3: Python Environment
echo  --------------------------------------------------------
echo   [1] Use system Python (recommended if torch+CUDA already installed)
echo   [2] Create / reuse virtual environment in venv-gpu\
echo.
set /p ENV_CHOICE=" Select [1/2, default=1]: "
if "!ENV_CHOICE!"=="" set ENV_CHOICE=1

if "!ENV_CHOICE!"=="2" goto :setup_venv
goto :env_ready

REM ---- Virtual environment setup -----------------------------
:setup_venv
echo.
if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo  [OK] Found existing venv-gpu, activating...
    call "%VENV_DIR%\Scripts\activate.bat"
    set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"
    goto :check_torch
)

echo  [>>] Creating virtual environment in venv-gpu\ ...
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo  [ERROR] Failed to create virtual environment.
    pause & exit /b 1
)
call "%VENV_DIR%\Scripts\activate.bat"
set "PYTHON_EXE=%VENV_DIR%\Scripts\python.exe"

:check_torch
echo.
echo  [??] Checking torch CUDA...
"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available(), 'no cuda'" > nul 2>&1
if errorlevel 1 (
    echo.
    echo  [WARN] torch with CUDA not found in this environment.
    echo         Please install it manually, then re-run this launcher.
    echo.
    echo         Example ^(CUDA 12.8^):
    echo           pip install torch --extra-index-url https://download.pytorch.org/whl/cu128
    echo.
    echo         Example ^(CUDA 13.0, RTX 50xx Blackwell^):
    echo           pip install torch --pre --extra-index-url https://download.pytorch.org/whl/nightly/cu130
    echo.
    echo  Installing other GPU requirements first...
    "%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements-gpu.txt" --quiet
    echo.
    echo  [!] After installing torch, run this launcher again.
    pause & exit /b 0
)
echo  [OK] torch CUDA available.

echo.
echo  [>>] Installing / updating GPU requirements...
"%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements-gpu.txt" --quiet
if errorlevel 1 (
    echo  [WARN] Some packages may have failed. Continuing...
)
echo  [OK] Requirements ready.
goto :env_ready

REM ---- Environment determined --------------------------------
:env_ready
echo.
echo  [OK] Using Python:
"%PYTHON_EXE%" -c "import sys; print('        ' + sys.executable)"
echo.

REM ---- Package check (system Python only; venv installs during setup) ----
if not "!ENV_CHOICE!"=="1" goto :pkg_check_done

echo  Checking required packages...
set "CHK=%SCRIPT_DIR%__chkpkg__.py"
echo import importlib.util                                                    > "%CHK%"
echo pkgs=[('qwen_asr','qwen-asr'),('customtkinter','customtkinter'),        >> "%CHK%"
echo       ('onnxruntime','onnxruntime'),('numpy','numpy'),                  >> "%CHK%"
echo       ('librosa','librosa'),('sounddevice','sounddevice'),              >> "%CHK%"
echo       ('opencc','opencc-python-reimplemented'),                         >> "%CHK%"
echo       ('huggingface_hub','huggingface-hub'),                            >> "%CHK%"
echo       ('torch','torch')]                                                 >> "%CHK%"
echo missing=[p for m,p in pkgs if importlib.util.find_spec(m) is None]     >> "%CHK%"
echo print(','.join(missing) if missing else 'OK')                           >> "%CHK%"

for /f "tokens=*" %%L in ('"%PYTHON_EXE%" "%CHK%" 2^>nul') do set PKG_RESULT=%%L
del "%CHK%" 2>nul

if "!PKG_RESULT!"=="OK" (
    echo  [OK] All required packages present.
    goto :pkg_check_done
)
if "!PKG_RESULT!"=="" (
    echo  [WARN] Package check failed, continuing anyway...
    goto :pkg_check_done
)

echo.
echo  [WARN] Missing packages detected:
echo         !PKG_RESULT!
echo.
echo   [1] Install now  ^(pip install -r requirements-gpu.txt^)
echo   [2] Continue without installing
echo.
set /p INST_CHOICE=" Select [1/2, default=1]: "
if "!INST_CHOICE!"=="" set INST_CHOICE=1

if "!INST_CHOICE!"=="1" (
    echo.
    echo  [>>] Installing missing packages...
    "%PYTHON_EXE%" -m pip install -r "%SCRIPT_DIR%requirements-gpu.txt"
    if errorlevel 1 (
        echo  [WARN] Some packages may have failed. Check output above.
    ) else (
        echo  [OK] Installation complete.
    )
) else (
    echo  [!] Continuing without installing. App may fail if packages are missing.
)

:pkg_check_done
echo.

REM ---- Step 2/3: Check / download GPU models -----------------
echo  Step 2/3: GPU Models
echo  --------------------------------------------------------
if exist "%ASR_MODEL_DIR%\config.json" (
    echo  [OK] Found: %ASR_MODEL_DIR%
    goto :models_ready
)

echo  [WARN] ASR model not found: %ASR_MODEL_DIR%
echo.
echo   [1] Download Qwen3-ASR-1.7B to GPUModel\  (approx 3.5 GB)
echo   [2] Skip download (I will place the model manually)
echo.
set /p DL_CHOICE=" Select [1/2, default=2]: "
if "!DL_CHOICE!"=="" set DL_CHOICE=2

if "!DL_CHOICE!"=="1" goto :download_models
echo  [!] Skipping download. Please place the model in:
echo      %ASR_MODEL_DIR%
echo  Then re-run this launcher.
pause & exit /b 0

:download_models
echo.
echo  [>>] Downloading Qwen3-ASR-1.7B...
echo       This may take a while depending on your connection.
echo.
"%PYTHON_EXE%" -c ^
"from huggingface_hub import snapshot_download; import os; ^
os.makedirs(r'%GPU_MODEL_DIR%', exist_ok=True); ^
snapshot_download('Qwen/Qwen3-ASR-1.7B', local_dir=r'%ASR_MODEL_DIR%', ^
    ignore_patterns=['*.md', '*.txt', 'flax_model*', 'tf_model*']); ^
print('[OK] Qwen3-ASR-1.7B downloaded.')"
if errorlevel 1 (
    echo  [ERROR] Download failed. Check network connection and try again.
    pause & exit /b 1
)

REM Optional: download ForcedAligner
set "ALIGNER_DIR=%GPU_MODEL_DIR%\Qwen3-ForcedAligner-0.6B"
if not exist "%ALIGNER_DIR%\config.json" (
    echo.
    echo  [?] Also download Qwen3-ForcedAligner-0.6B for word-level timestamps?
    echo      ^(approx 1.2 GB, optional^)
    echo.
    set /p AL_CHOICE=" Download aligner? [y/N]: "
    if /i "!AL_CHOICE!"=="y" (
        echo  [>>] Downloading Qwen3-ForcedAligner-0.6B...
        "%PYTHON_EXE%" -c ^
"from huggingface_hub import snapshot_download; ^
snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B', local_dir=r'%ALIGNER_DIR%', ^
    ignore_patterns=['*.md', '*.txt', 'flax_model*', 'tf_model*']); ^
print('[OK] ForcedAligner downloaded.')"
    )
)

REM Optional: copy VAD model if not present
if not exist "%GPU_MODEL_DIR%\silero_vad_v4.onnx" (
    if exist "%SCRIPT_DIR%ov_models\silero_vad_v4.onnx" (
        echo  [>>] Copying Silero VAD from ov_models\...
        copy "%SCRIPT_DIR%ov_models\silero_vad_v4.onnx" "%GPU_MODEL_DIR%\" > nul
        echo  [OK] VAD model copied.
    )
)

:models_ready
echo.

REM ---- GPU check ---------------------------------------------
"%PYTHON_EXE%" -c "import torch; avail=torch.cuda.is_available(); print('[OK] CUDA:', torch.cuda.get_device_name(0)) if avail else print('[WARN] CUDA not available - CPU mode')"

REM ---- Step 3/3: Launch desktop app --------------------------
echo.
echo  Step 3/3: Launch
echo  --------------------------------------------------------
echo  [NOTE] Streamlit frontend has been removed from this launcher.
echo         For Vulkan GPU support (NVIDIA/AMD), use QwenASR.exe instead.
echo         This launcher starts the PyTorch CustomTkinter desktop app.
echo.
echo  [>>] Starting desktop app (app-gpu.py)...
echo.
"%PYTHON_EXE%" "%APP_SCRIPT%"
if errorlevel 1 (
    echo.
    echo  [!] App exited with error. See message above.
    pause
)
goto :done

:done
endlocal
