@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

REM ============================================================
REM   Qwen3 ASR GPU Launcher  (PyTorch version)
REM   Uses PyTorch CUDA backend (no OpenVINO required)
REM   Model: cudagpu\GPUModel\Qwen3-ASR-1.7B
REM
REM   LAYOUT: this launcher stays at the package root; all GPU
REM   scripts, models and the venv live under cudagpu\ . Only
REM   start-gpu.bat is exposed at the root, keeping it tidy.
REM
REM   Step 3 launches the CustomTkinter desktop app (app-gpu.py).
REM   For a web interface, use the built-in Endpoint tab instead.
REM ============================================================

REM All GPU resources live under cudagpu\ (scripts, models, venv).
set "SCRIPT_DIR=%~dp0cudagpu\"
if not exist "%SCRIPT_DIR%" (
    echo  [ERROR] cudagpu folder not found next to start-gpu.bat.
    echo          Expected: %SCRIPT_DIR%
    pause & exit /b 1
)
set "GPU_MODEL_DIR=%SCRIPT_DIR%GPUModel"
set "ASR_MODEL_DIR=%GPU_MODEL_DIR%\Qwen3-ASR-1.7B"
set "ALIGNER_DIR=%GPU_MODEL_DIR%\Qwen3-ForcedAligner-0.6B"
set "OV_DIR=%SCRIPT_DIR%ov_models"
set "VENV_DIR=%SCRIPT_DIR%venv-gpu"
set "APP_SCRIPT=%SCRIPT_DIR%app-gpu.py"
set "PYTHON_EXE=python"

REM ---- Clean up leftover temp files from previous runs -------
if exist "%SCRIPT_DIR%__sl_run__.bat"  del "%SCRIPT_DIR%__sl_run__.bat"  2>nul
if exist "%SCRIPT_DIR%.tmp_ip"         del "%SCRIPT_DIR%.tmp_ip"         2>nul
if exist "%SCRIPT_DIR%__chkpkg__.py"   del "%SCRIPT_DIR%__chkpkg__.py"   2>nul
if exist "%SCRIPT_DIR%__pkgout__.txt"  del "%SCRIPT_DIR%__pkgout__.txt"  2>nul

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
"%PYTHON_EXE%" -c "import torch; assert torch.cuda.is_available()" > nul 2>&1
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

REM ---- Package check (system Python only; venv installs during :check_torch) ----
REM NOTE: Uses output-redirect instead of echo-to-.py to avoid temp script files.
if not "!ENV_CHOICE!"=="1" goto :pkg_check_done

echo  Checking required packages...
"%PYTHON_EXE%" -c "import importlib.util as u;pkgs=[('qwen_asr','qwen-asr'),('customtkinter','customtkinter'),('onnxruntime','onnxruntime'),('numpy','numpy'),('librosa','librosa'),('sounddevice','sounddevice'),('soundfile','soundfile'),('tokenizers','tokenizers'),('opencc','opencc-python-reimplemented'),('huggingface_hub','huggingface-hub'),('torch','torch')];missing=[p for m,p in pkgs if u.find_spec(m) is None];print(','.join(missing) if missing else 'OK')" > "%SCRIPT_DIR%__pkgout__.txt" 2>nul

set "PKG_RESULT="
for /f "delims=" %%L in ('type "%SCRIPT_DIR%__pkgout__.txt" 2^>nul') do if not defined PKG_RESULT set "PKG_RESULT=%%L"
del "%SCRIPT_DIR%__pkgout__.txt" 2>nul

if "!PKG_RESULT!"=="OK" (
    echo  [OK] All required packages present.
    goto :pkg_check_done
)
if "!PKG_RESULT!"=="" (
    echo  [WARN] Package check script failed, continuing anyway...
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
    goto :vad_check
)

echo  [WARN] ASR model not found: %ASR_MODEL_DIR%
echo.
echo   [1] Download Qwen3-ASR-1.7B to GPUModel\  (approx 3.5 GB)
echo   [2] Skip download (I will place the model manually)
echo.
set /p DL_CHOICE=" Select [1/2, default=2]: "
if "!DL_CHOICE!"=="" set DL_CHOICE=2

if "!DL_CHOICE!"=="1" goto :download_17b
echo  [!] Skipping download. Please place the model in:
echo      %ASR_MODEL_DIR%
echo  Then re-run this launcher.
pause & exit /b 0

REM ---- Download 1.7B model (single-line, no ^ continuation) --
:download_17b
echo.
echo  [>>] Downloading Qwen3-ASR-1.7B...
echo       This may take a while depending on your connection.
echo.
"%PYTHON_EXE%" -c "from huggingface_hub import snapshot_download; import os; os.makedirs(r'%GPU_MODEL_DIR%', exist_ok=True); snapshot_download('Qwen/Qwen3-ASR-1.7B', local_dir=r'%ASR_MODEL_DIR%', ignore_patterns=['*.md', 'flax_model*', 'tf_model*']); print('[OK] Qwen3-ASR-1.7B downloaded.')"
if errorlevel 1 (
    echo  [ERROR] Download failed. Check network connection and try again.
    pause & exit /b 1
)

REM ---- Check / download VAD ----------------------------------
:vad_check
if exist "%OV_DIR%\silero_vad_v4.onnx" goto :diar_check
echo.
echo  [>>] Downloading Silero VAD model to ov_models\ ...
"%PYTHON_EXE%" -c "import urllib.request, os; os.makedirs(r'%OV_DIR%', exist_ok=True); urllib.request.urlretrieve('https://github.com/snakers4/silero-vad/raw/v4.0/files/silero_vad.onnx', r'%OV_DIR%\silero_vad_v4.onnx'); print('[OK] VAD downloaded.')"
if errorlevel 1 (
    echo  [WARN] VAD download failed. Real-time recognition may not work.
)

REM ---- Check / download Diarization --------------------------
:diar_check
if exist "%OV_DIR%\diarization\embedding_model.onnx" goto :aligner_check
echo.
echo  [?] Download speaker diarization models to ov_models\diarization\ ?
echo      (approx 32 MB, required for speaker separation feature)
echo.
set /p DIAR_CHOICE=" Download? [Y/n, default=Y]: "
if "!DIAR_CHOICE!"=="" set DIAR_CHOICE=Y
if /i "!DIAR_CHOICE!"=="n" goto :aligner_check
echo  [>>] Downloading diarization models...
"%PYTHON_EXE%" -c "import urllib.request, os; d=r'%OV_DIR%\diarization'; os.makedirs(d, exist_ok=True); base='https://huggingface.co/altunenes/speaker-diarization-community-1-onnx/resolve/main'; urllib.request.urlretrieve(base+'/segmentation-community-1.onnx', d+'\\segmentation-community-1.onnx'); urllib.request.urlretrieve(base+'/embedding_model.onnx', d+'\\embedding_model.onnx'); print('[OK] Diarization models downloaded.')"
if errorlevel 1 (
    echo  [WARN] Diarization download failed. Speaker separation will not be available.
)

REM ---- Optional: ForcedAligner (word-level timestamps) -------
:aligner_check
if exist "%ALIGNER_DIR%\config.json" goto :models_ready
echo.
echo  [?] Also download Qwen3-ForcedAligner-0.6B for word-level timestamps?
echo      (approx 1.2 GB, optional)
echo.
set /p AL_CHOICE=" Download aligner? [y/N, default=N]: "
if "!AL_CHOICE!"=="" set AL_CHOICE=N
if /i not "!AL_CHOICE!"=="y" goto :models_ready
echo  [>>] Downloading Qwen3-ForcedAligner-0.6B...
"%PYTHON_EXE%" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-ForcedAligner-0.6B', local_dir=r'%ALIGNER_DIR%', ignore_patterns=['*.md', 'flax_model*', 'tf_model*']); print('[OK] ForcedAligner downloaded.')"

REM ---- Models ready ------------------------------------------
:models_ready
echo.

REM ---- GPU check ---------------------------------------------
"%PYTHON_EXE%" -c "import torch; avail=torch.cuda.is_available(); print('[OK] CUDA: '+torch.cuda.get_device_name(0)) if avail else print('[WARN] CUDA not available - will run in CPU mode')"

REM ---- Step 3/3: Launch desktop app --------------------------
echo.
echo  Step 3/3: Launch Desktop App
echo  --------------------------------------------------------
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
