@echo off
REM =======================================================
REM  Qwen3 ASR - PyInstaller Build Script (onedir mode)
REM
REM  OUTPUT STRUCTURE:
REM    dist\QwenASR\
REM      QwenASR.exe        <- launcher (~5 MB)
REM      prompt_template.json
REM      _internal\         <- Python runtime + packages
REM
REM  DISTRIBUTION:
REM    Run setup.iss with Inno Setup to produce
REM    QwenASR_Setup.exe (~400 MB installer).
REM    Models (~1.2 GB) are downloaded at first run.
REM
REM  STARTUP TIME:
REM    onedir  -> 3-5 s  (DLLs loaded directly)
REM    onefile -> 20-35 s (must extract to %%TEMP%% first)
REM =======================================================

REM Use build_venv (no torch) for smaller output.
REM Run build_venv.bat first if build_venv\ doesn't exist.
IF EXIST "F:\AIStudio\QwenASR\build_venv\Scripts\python.exe" (
    SET VENV=F:\AIStudio\QwenASR\build_venv
) ELSE (
    SET VENV=F:\AIStudio\QwenASR\venv
)
SET PYTHON=%VENV%\Scripts\python.exe
SET SRC=F:\AIStudio\QwenASR

echo === Step 1: Install PyInstaller ===
%PYTHON% -m pip install pyinstaller --quiet

echo.
echo === Step 2: Locate dependency paths ===

FOR /F "delims=" %%i IN ('%PYTHON% -c "import opencc, os; print(os.path.dirname(opencc.__file__))"') DO SET OPENCC_DIR=%%i
FOR /F "delims=" %%i IN ('%PYTHON% -c "import customtkinter, os; print(os.path.dirname(customtkinter.__file__))"') DO SET CTK_DIR=%%i
FOR /F "delims=" %%i IN ('%PYTHON% -c "import openvino, os; print(os.path.dirname(openvino.__file__))"') DO SET OV_PKG=%%i

echo opencc       : %OPENCC_DIR%
echo customtkinter: %CTK_DIR%
echo openvino     : %OV_PKG%

echo.
echo === Step 3: Build with PyInstaller (onedir) ===

REM --onedir is the DEFAULT (no --onefile flag).
REM _internal/ keeps the root folder tidy (PyInstaller >= 6.0).
REM
REM prompt_template.json and mel_filters.npy are bundled inside _internal/
REM so LightProcessor can find them via Path(__file__).parent fallback.

%PYTHON% -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "QwenASR" ^
    --icon NONE ^
    --add-data "%CTK_DIR%;customtkinter" ^
    --add-data "%OPENCC_DIR%;opencc" ^
    --add-data "%OV_PKG%;openvino" ^
    --add-data "%SRC%\prompt_template.json;." ^
    --add-data "%SRC%\ov_models\mel_filters.npy;ov_models" ^
    --hidden-import openvino ^
    --hidden-import openvino.runtime ^
    --hidden-import onnxruntime ^
    --hidden-import opencc ^
    --hidden-import customtkinter ^
    --hidden-import sounddevice ^
    --hidden-import librosa ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module transformers ^
    --exclude-module qwen_asr ^
    --exclude-module triton ^
    --exclude-module bitsandbytes ^
    --noconfirm ^
    %SRC%\app.py

echo.
IF EXIST "%SRC%\dist\QwenASR\QwenASR.exe" (
    echo ===================================================
    echo  Build SUCCESS
    echo  Launcher : dist\QwenASR\QwenASR.exe
    echo  Runtime  : dist\QwenASR\_internal\
    echo.
    echo  Next step: open setup.iss with Inno Setup
    echo  to produce QwenASR_Setup.exe for distribution.
    echo ===================================================
) ELSE (
    echo ===================================================
    echo  Build FAILED. Check output above.
    echo ===================================================
)
pause
