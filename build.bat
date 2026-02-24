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
    --icon NONE ^
    --add-data "%CTK_DIR%;customtkinter" ^
    --add-data "%OPENCC_DIR%;opencc" ^
    --add-data "%OV_PKG%;openvino" ^
    --add-data "%KNF_DIR%;kaldi_native_fbank" ^
    --add-data "%SRC%\prompt_template.json;." ^
    --add-data "%SRC%\ov_models\mel_filters.npy;ov_models" ^
    --add-data "%SRC%\ov_models\silero_vad_v4.onnx;ov_models" ^
    --runtime-hook "%SRC%\runtime_hook_utf8.py" ^
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
    %SRC%\app.py

echo.
IF EXIST "%SRC%\dist\QwenASR\QwenASR.exe" (
    echo ===================================================
    echo  Build SUCCESS - Copying chatllm DLLs...
    echo ===================================================

    REM Copy chatllm DLLs + main.exe to dist\QwenASR\chatllm\
    REM These are needed for Vulkan GPU backend (libchatllm.dll, ggml-vulkan.dll, etc.)
    REM  libchatllm.dll  - newly built (2026-02-23), supports ASR models
    REM  ggml-vulkan.dll - Vulkan GPU backend (52 MB shader kernels)
    REM  ggml-cpu-*.dll  - CPU fallback variants (selected at runtime)
    REM  main.exe        - used for --show_devices GPU detection only

    IF EXIST "%SRC%\chatllm" (
        xcopy "%SRC%\chatllm\libchatllm.dll"         "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml.dll"               "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-base.dll"          "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-cpu-alderlake.dll" "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-cpu-haswell.dll"   "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-cpu-icelake.dll"   "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-cpu-sandybridge.dll" "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-cpu-skylakex.dll"  "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-cpu-sse42.dll"     "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-cpu-x64.dll"       "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-rpc.dll"           "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\ggml-vulkan.dll"        "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\libcrypto-1_1-x64.dll"  "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\libssl-1_1-x64.dll"     "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\vulkan-1.dll"            "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        xcopy "%SRC%\chatllm\main.exe"               "%SRC%\dist\QwenASR\chatllm\" /Y /Q
        echo  chatllm/    : DLLs copied to dist\QwenASR\chatllm\
    ) ELSE (
        echo  WARNING: chatllm\ not found - GPU backend will not be available
        echo  Copy chatllm DLLs to %SRC%\chatllm\ before building.
    )

    echo.
    REM Copy streamlit_app.py (needed by the service tab to run Streamlit)
    xcopy "%SRC%\streamlit_app.py"               "%SRC%\dist\QwenASR\" /Y /Q
    echo  streamlit_app.py copied to dist\QwenASR\

    REM Copy python.exe + python3*.dll to _python\ subdirectory (NOT the EXE root).
    REM Reason: placing python3XX.dll in the same dir as QwenASR.exe causes Windows
    REM DLL loader to load TWO different copies of the Python DLL (one for QwenASR.exe
    REM via PyInstaller _internal/, one for python.exe via the root dir), which can
    REM produce undefined behavior and per-chunk window flashes in windowed mode.
    REM python.exe finds python3XX.dll in its own directory (_python\), which is correct.
    xcopy "%VENV%\Scripts\python.exe"            "%SRC%\dist\QwenASR\_python\" /Y /Q
    xcopy "%VENV%\Scripts\python3*.dll"          "%SRC%\dist\QwenASR\_python\" /Y /Q 2>NUL
    echo  python.exe  copied to dist\QwenASR\_python\  (required for Streamlit service)

    echo  Launcher : dist\QwenASR\QwenASR.exe
    echo  Runtime  : dist\QwenASR\_internal\
    echo  GPU DLLs : dist\QwenASR\chatllm\   (~71 MB, Vulkan backend)
    echo.
    echo  Model downloaded at first run from:
    echo    https://huggingface.co/dseditor/Collection/resolve/main/qwen3-asr-1.7b.bin
    echo  Saved to: {app}\GPUModel\qwen3-asr-1.7b.bin  (~2.3 GB)
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
