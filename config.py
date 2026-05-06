"""
Configuration for Voice Interaction System
EECE7398 Deep Learning Embedded Systems - Project 2
"""
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VOICE_REF_PATH = os.path.join(BASE_DIR, "voice_reference.wav")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# ASR MODELS (faster-whisper)
# ============================================================
ASR_MODELS = {
    "whisper-tiny": "tiny",     # ~39M params, ~75MB download
    "whisper-base": "base",     # ~74M params, ~150MB download
}

# ============================================================
# LLM MODELS (llama.cpp, Q4_K_M quantization)
# ============================================================
LLM_MODELS = {
    "qwen2.5-0.5b": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
    },
    "qwen2.5-1.5b": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
    },
}

# ============================================================
# TTS MODELS
# ============================================================
# Kokoro: lightweight, preset voices, no cloning
# XTTS-v2: heavier, supports zero-shot voice cloning
TTS_XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
KOKORO_MODEL_PATH = os.path.join(BASE_DIR, "models", "kokoro", "kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = os.path.join(BASE_DIR, "models", "kokoro", "voices-v1.0.bin")

# ============================================================
# AUDIO SETTINGS
# ============================================================
SAMPLE_RATE = 16000          # For ASR input
TTS_SAMPLE_RATE = 24000      # Typical TTS output rate
RECORD_SECONDS = 10          # Default recording duration
MIC_DEVICE = None            # None = system default

# ============================================================
# LLM SETTINGS
# ============================================================
LLM_GPU_LAYERS = -1          # -1 = offload all layers to GPU
LLM_CONTEXT_SIZE = 2048
LLM_MAX_TOKENS = 256

# ============================================================
# SYSTEM PROMPTS
# ============================================================
QA_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Give concise, clear answers "
    "suitable for being read aloud. Keep responses under 3 sentences "
    "unless the question requires more detail."
)

TRANSLATION_SYSTEM_PROMPT = (
    "You are a translation assistant. Translate the following text to {target_lang}. "
    "Output ONLY the translation, nothing else. No explanations."
)

# ============================================================
# TEST DATA
# ============================================================
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming how we interact with technology.",
    "Deep learning models require significant computational resources for training.",
    "Voice assistants have become an integral part of modern smart devices.",
    "Neural networks can learn complex patterns from large amounts of data.",
]

LLM_TEST_PROMPTS = [
    "What is deep learning in one sentence?",
    "Explain the difference between CPU and GPU in simple terms.",
    "What are the main components of a voice interaction system?",
]