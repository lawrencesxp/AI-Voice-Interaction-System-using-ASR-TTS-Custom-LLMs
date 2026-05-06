# Voice Interaction System

**EECE7398 Deep Learning Embedded Systems — Project 2**

A voice-based Q&A and translation system using:
- **ASR**: Whisper (tiny & base) via faster-whisper
- **LLM**: Qwen2.5 (0.5B & 1.5B) via llama.cpp, Q4_K_M quantization
- **TTS**: Kokoro-82M + XTTS-v2 (with zero-shot voice cloning)
- **UI**: Gradio web interface

Deployed on Windows 10 with NVIDIA RTX 2060 (6GB VRAM).

---

## Setup (Step-by-Step for Windows)

### Prerequisites
- **Python 3.10** (3.10.x recommended, 3.11 may also work)
- **Git** (for some model downloads)
- **Visual Studio Build Tools** (needed for llama-cpp-python CUDA build)
  - Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
  - Install "Desktop development with C++" workload
  - This gives you the C++ compiler needed to build llama-cpp-python

### Step 1: Open a terminal in the project folder

```powershell
cd path\to\voice_interaction_system
```

### Step 2: Create and activate a virtual environment

```powershell
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install PyTorch with CUDA

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA works:
```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Should print `True` and `NVIDIA GeForce RTX 2060`.

### Step 4: Install llama-cpp-python with CUDA support

**Option A — Build from source (recommended):**
```powershell
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --no-cache-dir
```

If this fails (missing compiler), make sure Visual Studio Build Tools are installed (see Prerequisites).

**Option B — Prebuilt wheel (if Option A fails):**
```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

**Option C — CPU only (fallback):**
```powershell
pip install llama-cpp-python
```
This works but will be slower. Change `LLM_GPU_LAYERS = 0` in `config.py`.

### Step 5: Install remaining dependencies

```powershell
pip install faster-whisper kokoro-onnx onnxruntime-gpu TTS jiwer huggingface_hub sounddevice soundfile numpy gradio
```

If `onnxruntime-gpu` fails, use `onnxruntime` instead (Kokoro will run on CPU, still fast).

### Step 6: Download models

```powershell
python download_models.py
```

This downloads:
- Whisper tiny & base (~225MB total)
- Qwen2.5-0.5B & 1.5B GGUF (~1.4GB total)
- Kokoro-82M ONNX (~300MB)
- XTTS-v2 (~1.8GB)

Total: ~3.7GB. At 5 Mbps this takes about 1-1.5 hours.

---

## Running

### Section 1: Individual Benchmarks

```powershell
# ASR benchmark (Whisper-tiny vs Whisper-base)
python benchmark_asr.py

# LLM benchmark (Qwen2.5-0.5B vs Qwen2.5-1.5B)
python benchmark_llm.py

# TTS benchmark (Kokoro vs XTTS-v2) — run record_voice.py first for voice cloning
python record_voice.py
python benchmark_tts.py
```

Results are saved as JSON in `outputs/` and printed to terminal.

### Section 2: Integrated System

**Command-line demo:**
```powershell
python pipeline.py
```

**Web UI (Gradio):**
```powershell
python app.py
```
Opens at http://localhost:7860

### Voice Cloning

1. Run `python record_voice.py` to record a 10-second clip of your voice
2. The file is saved as `voice_reference.wav`
3. In `benchmark_tts.py`, this is automatically used for the XTTS-v2 cloning test
4. In the Gradio UI, select "XTTS-v2 (Voice Cloning)" and upload/record a reference clip

---

## Project Structure

```
voice_interaction_system/
├── config.py              # All configuration and settings
├── download_models.py     # Download all models
├── benchmark_asr.py       # ASR: Whisper-tiny vs Whisper-base
├── benchmark_llm.py       # LLM: Qwen2.5-0.5B vs 1.5B
├── benchmark_tts.py       # TTS: Kokoro vs XTTS-v2 + voice cloning
├── record_voice.py        # Record voice reference clip
├── pipeline.py            # Integrated ASR→LLM→TTS pipeline
├── app.py                 # Gradio web UI
├── requirements.txt       # Python dependencies
├── install.bat            # Automated Windows setup
├── README.md              # This file
├── models/                # Downloaded model files
│   ├── kokoro/            # Kokoro ONNX files
│   └── *.gguf             # Qwen GGUF files
└── outputs/               # Benchmark results and audio files
    ├── asr_benchmark/
    ├── llm_benchmark/
    └── tts_benchmark/
```

---

## Troubleshooting

**"CUDA out of memory"**: Close other GPU-heavy apps (games, browsers with hardware acceleration). Check `nvidia-smi` for current usage.

**llama-cpp-python build fails**: Make sure Visual Studio Build Tools "Desktop development with C++" is installed. Restart terminal after installing.

**Kokoro not found**: Re-run `python download_models.py`. Check that `models/kokoro/kokoro-v0_19.onnx` exists.

**Microphone not working**: Run `python -c "import sounddevice; print(sounddevice.query_devices())"` to check devices. Set `MIC_DEVICE` in `config.py` if needed.

**Slow LLM responses**: Make sure GPU layers are being used. Check the llama.cpp loading output for "offloaded X/Y layers to GPU".
