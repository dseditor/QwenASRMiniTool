@echo off
REM =======================================================
REM  Qwen3 ASR - WebView edition build (PyInstaller, onedir)
REM
REM  Builds app_webview.py into its own exe:
REM    dist2\QwenASR-WebView\QwenASR-WebView.exe
REM
REM  The WebView UI = local stdlib HTTP server (webview_server.py) that
REM  serves the webview\ folder + /api, opened in a native WebView2 window
REM  via pywebview (URL only, no js_api). Same engine as the CTk edition
REM  (app_webview -> webview_backend -> import app), so ALL engine deps must
REM  be bundled too.
REM
REM  NOTE: ALL content in this .bat is English (Windows CP950 safety).
REM =======================================================

IF EXIST "F:\AIStudio\QwenASR\build_venv\Scripts\python.exe" (
    SET VENV=F:\AIStudio\QwenASR\build_venv
) ELSE (
    SET VENV=F:\AIStudio\QwenASR\venv
)
SET PYTHON=%VENV%\Scripts\python.exe
SET SRC=F:\AIStudio\QwenASR

echo === Step 1: Ensure PyInstaller + pywebview ===
%PYTHON% -m pip install pyinstaller pywebview --quiet

echo.
echo === Step 2: Locate dependency paths ===
FOR /F "delims=" %%i IN ('%PYTHON% -c "import opencc, os; print(os.path.dirname(opencc.__file__))"') DO SET OPENCC_DIR=%%i
FOR /F "delims=" %%i IN ('%PYTHON% -c "import openvino, os; print(os.path.dirname(openvino.__file__))"') DO SET OV_PKG=%%i
FOR /F "delims=" %%i IN ('%PYTHON% -c "import kaldi_native_fbank, os; print(os.path.dirname(kaldi_native_fbank.__file__))"') DO SET KNF_DIR=%%i

echo opencc            : %OPENCC_DIR%
echo openvino          : %OV_PKG%
echo kaldi_native_fbank: %KNF_DIR%

IF "%OPENCC_DIR%"=="" GOTO :dep_failed
IF "%OV_PKG%"==""     GOTO :dep_failed
IF "%KNF_DIR%"==""    GOTO :dep_failed

echo.
echo === Step 2b: Ensure silero_vad_v4.onnx present ===
IF NOT EXIST "%SRC%\ov_models\silero_vad_v4.onnx" (
    echo   downloading silero_vad_v4.onnx...
    %PYTHON% -c "from downloader import _download_file, _VAD_URL; from pathlib import Path; p=Path(r'%SRC%\ov_models'); p.mkdir(exist_ok=True); _download_file(_VAD_URL, p/'silero_vad_v4.onnx')"
)

echo.
echo === Step 3: Build WebView edition (onedir) ===
IF EXIST "%SRC%\dist2\QwenASR-WebView\QwenASR-WebView.exe" DEL /Q "%SRC%\dist2\QwenASR-WebView\QwenASR-WebView.exe"

%PYTHON% -m PyInstaller ^
    --onedir ^
    --windowed ^
    --name "QwenASR-WebView" ^
    --distpath "%SRC%\dist2" ^
    --icon NONE ^
    --add-data "%OPENCC_DIR%;opencc" ^
    --add-data "%OV_PKG%;openvino" ^
    --add-data "%KNF_DIR%;kaldi_native_fbank" ^
    --add-data "%SRC%\webview;webview_assets" ^
    --add-data "%SRC%\prompt_template.json;." ^
    --add-data "%SRC%\ov_models\mel_filters.npy;ov_models" ^
    --add-data "%SRC%\ov_models\silero_vad_v4.onnx;ov_models" ^
    --add-data "%SRC%\app.py;." ^
    --add-data "%SRC%\webview_backend.py;." ^
    --add-data "%SRC%\webview_server.py;." ^
    --add-data "%SRC%\batch_tab.py;." ^
    --add-data "%SRC%\ffmpeg_utils.py;." ^
    --add-data "%SRC%\subtitle_editor.py;." ^
    --add-data "%SRC%\setting.py;." ^
    --add-data "%SRC%\model_tab.py;." ^
    --add-data "%SRC%\audio_io.py;." ^
    --add-data "%SRC%\version.py;." ^
    --add-data "%SRC%\updater.py;." ^
    --add-data "%SRC%\fa_aligner.py;." ^
    --add-data "%SRC%\api_server.py;." ^
    --add-data "%SRC%\endpoint_tab.py;." ^
    --add-data "%SRC%\crisp_engine.py;." ^
    --add-data "%SRC%\subtitle_lines.py;." ^
    --add-data "%SRC%\chatllm_engine.py;." ^
    --add-data "%SRC%\diarize.py;." ^
    --add-data "%SRC%\processor_numpy.py;." ^
    --runtime-hook "%SRC%\runtime_hook_utf8.py" ^
    --collect-all webview ^
    --collect-data certifi ^
    --collect-all tokenizers ^
    --hidden-import certifi ^
    --hidden-import webview ^
    --hidden-import webview.platforms.edgechromium ^
    --hidden-import clr_loader ^
    --hidden-import proxy_tools ^
    --hidden-import bottle ^
    --hidden-import app ^
    --hidden-import webview_backend ^
    --hidden-import webview_server ^
    --hidden-import openvino ^
    --hidden-import openvino.runtime ^
    --hidden-import onnxruntime ^
    --hidden-import opencc ^
    --hidden-import customtkinter ^
    --hidden-import sounddevice ^
    --hidden-import soxr ^
    --hidden-import soundfile ^
    --hidden-import kaldi_native_fbank ^
    --hidden-import scipy ^
    --hidden-import scipy.cluster ^
    --hidden-import scipy.cluster.hierarchy ^
    --hidden-import scipy.spatial ^
    --hidden-import scipy.spatial.distance ^
    --hidden-import scipy._lib.messagestream ^
    --hidden-import crisp_engine ^
    --hidden-import subtitle_lines ^
    --hidden-import version ^
    --hidden-import updater ^
    --hidden-import fa_aligner ^
    --hidden-import api_server ^
    --hidden-import endpoint_tab ^
    --hidden-import segno ^
    --exclude-module torch ^
    --exclude-module torchvision ^
    --exclude-module torchaudio ^
    --exclude-module transformers ^
    --exclude-module qwen_asr ^
    --exclude-module triton ^
    --exclude-module bitsandbytes ^
    --noconfirm ^
    %SRC%\app_webview.py

IF ERRORLEVEL 1 GOTO :build_failed
IF NOT EXIST "%SRC%\dist2\QwenASR-WebView\QwenASR-WebView.exe" GOTO :build_failed

echo.
echo ===================================================
echo  WebView build SUCCESS - copying chatllm + ffmpeg
echo ===================================================
IF NOT EXIST "%SRC%\chatllm\libchatllm.dll" GOTO :after_chatllm
xcopy "%SRC%\chatllm\*" "%SRC%\dist2\QwenASR-WebView\chatllm\" /E /I /Y /Q
echo  chatllm/ copied
:after_chatllm
IF NOT EXIST "%SRC%\ffmpeg\ffmpeg.exe" GOTO :after_ffmpeg
IF NOT EXIST "%SRC%\dist2\QwenASR-WebView\ffmpeg\" mkdir "%SRC%\dist2\QwenASR-WebView\ffmpeg\"
xcopy "%SRC%\ffmpeg\ffmpeg.exe" "%SRC%\dist2\QwenASR-WebView\ffmpeg\" /Y /Q
echo  ffmpeg/ copied
:after_ffmpeg

echo.
echo  Done: dist2\QwenASR-WebView\QwenASR-WebView.exe
GOTO :eof

:dep_failed
echo.
echo  ERROR: a dependency path is empty (import failed). Aborting.
EXIT /B 1

:build_failed
echo.
echo  ERROR: PyInstaller build FAILED.
EXIT /B 1
