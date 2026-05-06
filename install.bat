@echo off
echo ============================================================
echo Voice Interaction System - Windows Setup
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo [1/6] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Upgrade pip
echo [2/6] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 12.4
echo [3/6] Installing PyTorch with CUDA support...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

REM Install llama-cpp-python with CUDA
echo [4/6] Installing llama-cpp-python with CUDA...
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python

REM If the above fails, try prebuilt wheel:
REM pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

REM Install other dependencies
echo [5/6] Installing other dependencies...
pip install faster-whisper kokoro-onnx onnxruntime-gpu TTS jiwer huggingface_hub sounddevice soundfile numpy gradio

REM Download models
echo [6/6] Downloading models...
python download_models.py

echo.
echo ============================================================
echo Setup complete! Activate the venv and run:
echo   venv\Scripts\activate
echo   python app.py
echo ============================================================
pause
