@echo off
REM =======================================================
REM  Qwen3 ASR - PyInstaller Build Script (onedir mode)
REM
REM  OUTPUT STRUCTURE (dist2\):
REM    dist2\QwenASR\
REM      QwenASR.exe        <- launcher (~5 MB), CPU / Vulkan edition
REM      prompt_template.json
REM      _internal\         <- Python runtime + packages
REM      chatllm\           <- v2026.05 chatllm binaries (GPU/CPU backend)
REM      ffmpeg\            <- bundled ffmpeg.exe (video support)
REM      start-gpu.bat + app-gpu.py + *.py  <- GPU (PyTorch CUDA) edition
REM
REM  DISTRIBUTION:
REM    Run make_release_zip.bat to produce QwenASR_<version>.zip for a
REM    GitHub Release. The in-app updater (Settings tab) downloads and
REM    applies that ZIP. Models (~2.3 GB) are downloaded at first run.
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
FOR /F "delims=" %%i IN ('%PYTHON% -c "import kaldi_native_fbank, os; print(os.path.dirname(kaldi_native_fbank.__file__))"') DO SET KNF_DIR=%%i

echo opencc            : %OPENCC_DIR%
echo customtkinter     : %CTK_DIR%
echo openvino          : %OV_PKG%
echo kaldi_native_fbank: %KNF_DIR%

echo.
echo === Step 2b: Ensure silero_vad_v4.onnx is present before bundling ===
REM VAD model must exist locally so --add-data can bundle it into _internal/ov_models/
REM If missing, download it now (small file ~2 MB from GitHub).
IF NOT EXIST "%SRC%\ov_models\silero_vad_v4.onnx" (
    echo   silero_vad_v4.onnx not found, downloading...
    %PYTHON% -c "from downloader import _download_file, _VAD_URL; from pathlib import Path; p=Path(r'%SRC%\ov_models'); p.mkdir(exist_ok=True); _download_file(_VAD_URL, p/'silero_vad_v4.onnx')"
    IF ERRORLEVEL 1 (
        echo   WARNING: VAD download failed - bundling skipped. Users will download at runtime.
    ) ELSE (
        echo   silero_vad_v4.onnx downloaded OK.
    )
) ELSE (
    echo   silero_vad_v4.onnx already present.
)

echo.
echo === Step 3: Build with PyInstaller (onedir) ===

REM --onedir is the DEFAULT (no --onefile flag).
REM _internal/ keeps the root folder tidy (PyInstaller >= 6.0).
REM
REM prompt_template.json and mel_filters.npy are bundled inside _internal/
REM so LightProcessor can find them via Path(__file__).parent fallback.
REM
REM runtime_hook_utf8.py: sets PYTHONUTF8=1 before any user code runs.
REM This prevents "utf-8 codec can't decode byte 0xa6" on Traditional
REM Chinese Windows (cp950 default encoding).

%PYTHON% -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "QwenASR" ^
    --distpath "%SRC%\dist2" ^
    --icon NONE ^
    --add-data "%CTK_DIR%;customtkinter" ^
    --add-data "%OPENCC_DIR%;opencc" ^
    --add-data "%OV_PKG%;openvino" ^
    --add-data "%KNF_DIR%;kaldi_native_fbank" ^
    --add-data "%SRC%\prompt_template.json;." ^
    --add-data "%SRC%\ov_models\mel_filters.npy;ov_models" ^
    --add-data "%SRC%\ov_models\silero_vad_v4.onnx;ov_models" ^
    --runtime-hook "%SRC%\runtime_hook_utf8.py" ^
    --collect-data certifi ^
    --hidden-import certifi ^
    --collect-all tokenizers ^
    --hidden-import openvino ^
    --hidden-import openvino.runtime ^
    --hidden-import onnxruntime ^
    --hidden-import opencc ^
    --hidden-import customtkinter ^
    --hidden-import sounddevice ^
    --hidden-import librosa ^
    --hidden-import soundfile ^
    --hidden-import kaldi_native_fbank ^
    --hidden-import scipy ^
    --hidden-import scipy.cluster ^
    --hidden-import scipy.cluster.hierarchy ^
    --hidden-import scipy.spatial ^
    --hidden-import scipy.spatial.distance ^
    --hidden-import scipy._lib.messagestream ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module transformers ^
    --exclude-module qwen_asr ^
    --exclude-module triton ^
    --exclude-module bitsandbytes ^
    --noconfirm ^
    --add-data "%SRC%\batch_tab.py;." ^
    --add-data "%SRC%\ffmpeg_utils.py;." ^
    --add-data "%SRC%\subtitle_editor.py;." ^
    --add-data "%SRC%\setting.py;." ^
    --add-data "%SRC%\version.py;." ^
    --add-data "%SRC%\updater.py;." ^
    --hidden-import version ^
    --hidden-import updater ^
    %SRC%\app.py

echo.
REM NOTE: a GOTO flow (not IF (...) ELSE (...)) is used below on purpose.
REM The post-build messages contain parentheses, e.g. "(CPU / Vulkan edition)".
REM Inside a parenthesized IF block cmd counts parens, so any such text could
REM close the block early and run the ELSE branch ("Build FAILED") by mistake.
REM GOTO removes the enclosing parens entirely, making echo text paren-safe.
IF NOT EXIST "%SRC%\dist2\QwenASR\QwenASR.exe" GOTO :build_failed

echo ===================================================
echo  Build SUCCESS - Copying chatllm + GPU scripts...
echo ===================================================

REM Copy the ENTIRE chatllm folder (v2026.05 release binaries) to
REM dist2\QwenASR\chatllm\. Whole-folder copy keeps every ggml-cpu-*
REM variant and stays in sync with future chatllm releases.
IF NOT EXIST "%SRC%\chatllm\libchatllm.dll" GOTO :no_chatllm
xcopy "%SRC%\chatllm\*" "%SRC%\dist2\QwenASR\chatllm\" /E /I /Y /Q
echo  chatllm/    : full folder copied to dist2\QwenASR\chatllm\
GOTO :after_chatllm
:no_chatllm
echo  WARNING: chatllm\libchatllm.dll not found - GPU backend unavailable
echo  Place the chatllm release binaries in %SRC%\chatllm\ before building.
:after_chatllm

echo.
REM Copy bundled ffmpeg.exe to dist2\QwenASR\ffmpeg\
IF NOT EXIST "%SRC%\ffmpeg\ffmpeg.exe" GOTO :no_ffmpeg
IF NOT EXIST "%SRC%\dist2\QwenASR\ffmpeg\" mkdir "%SRC%\dist2\QwenASR\ffmpeg\"
xcopy "%SRC%\ffmpeg\ffmpeg.exe" "%SRC%\dist2\QwenASR\ffmpeg\" /Y /Q
echo  ffmpeg/     : ffmpeg.exe copied to dist2\QwenASR\ffmpeg\
GOTO :after_ffmpeg
:no_ffmpeg
echo  WARNING: ffmpeg\ffmpeg.exe not found - users download it at runtime
:after_ffmpeg

echo.
REM ===== GPU edition scripts (PyTorch CUDA, runs on system Python) =====
REM Copied to the app ROOT so start-gpu.bat finds GPUModel\, ov_models\
REM and app-gpu.py beside itself, exactly as in the source repo layout.
FOR %%F IN (app-gpu.py streamlit_vulkan.py streamlit_app.py subtitle_editor.py batch_tab.py diarize.py ffmpeg_utils.py chatllm_engine.py processor_numpy.py downloader.py generate_srt.py version.py updater.py requirements-gpu.txt start-gpu.bat) DO IF EXIST "%SRC%\%%F" xcopy "%SRC%\%%F" "%SRC%\dist2\QwenASR\" /Y /Q >nul
IF EXIST "%SRC%\setting.py"           xcopy "%SRC%\setting.py"           "%SRC%\dist2\QwenASR\" /Y /Q >nul
IF EXIST "%SRC%\prompt_template.json" xcopy "%SRC%\prompt_template.json" "%SRC%\dist2\QwenASR\" /Y /Q >nul
echo  GPU edn     : scripts copied to app root (start-gpu.bat, app-gpu.py)

echo.
echo  Launcher : dist2\QwenASR\QwenASR.exe   (CPU / Vulkan edition)
echo  Runtime  : dist2\QwenASR\_internal\
echo  GPU DLLs : dist2\QwenASR\chatllm\      (~71 MB, Vulkan backend)
echo  ffmpeg   : dist2\QwenASR\ffmpeg\ffmpeg.exe  (video support)
echo  GPU edn  : dist2\QwenASR\start-gpu.bat (PyTorch CUDA, app-gpu.py)
echo  Update   : in-app check in Settings tab (GitHub Releases)
echo.
echo  Next step: run make_release_zip.bat to produce
echo  QwenASR_^<version^>.zip for a GitHub Release (auto-update source).
echo ===================================================
GOTO :build_end

:build_failed
echo ===================================================
echo  Build FAILED. Check output above.
echo ===================================================

:build_end
pause
