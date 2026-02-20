@echo off
REM =======================================================
REM  Create a minimal build venv WITHOUT torch/transformers
REM  to reduce the final EXE package size.
REM
REM  Run this ONCE before running build.bat.
REM  This will create F:\AIStudio\QwenASR\build_venv\
REM =======================================================

SET BASE=F:\AIStudio\QwenASR
SET BVENV=%BASE%\build_venv

echo === Creating minimal build venv (no torch) ===
python -m venv "%BVENV%"
IF ERRORLEVEL 1 (
    echo ERROR: python -m venv failed. Is Python 3.10+ on PATH?
    pause
    exit /b 1
)

SET PIP=%BVENV%\Scripts\pip.exe
SET PYTHON=%BVENV%\Scripts\python.exe

echo.
echo === Installing packages (no torch, no transformers) ===

REM Core inference
%PIP% install openvino --quiet
%PIP% install onnxruntime --quiet

REM Audio I/O (supports MP3/M4A via Windows MF, FLAC, OGG, WAV)
%PIP% install librosa sounddevice --quiet

REM GUI + text
%PIP% install customtkinter --quiet
%PIP% install opencc-python-reimplemented --quiet

REM (huggingface_hub removed - downloader uses urllib stdlib directly)

REM PyInstaller
%PIP% install pyinstaller --quiet

echo.
echo === Verifying no torch ===
%PYTHON% -c "import sys; mods=[m for m in sys.modules if 'torch' in m]; print('torch modules:', mods)"

echo.
echo ===================================================
echo  build_venv ready at: %BVENV%
echo.
echo  Next step: edit build.bat line 21:
echo    SET VENV=%BVENV%
echo  Then run build.bat
echo ===================================================
pause
