"""
Download all required models.
Run this first: python download_models.py
"""
import os
import sys

def download_llm_models():
    """Download GGUF files for LLM models."""
    from huggingface_hub import hf_hub_download
    from config import LLM_MODELS, MODEL_DIR

    for name, info in LLM_MODELS.items():
        dest = os.path.join(MODEL_DIR, info["file"])
        if os.path.exists(dest):
            print(f"[OK] {name} already downloaded: {dest}")
            continue
        print(f"\n[DOWNLOADING] {name} from {info['repo']}...")
        print(f"    File: {info['file']}")
        path = hf_hub_download(
            repo_id=info["repo"],
            filename=info["file"],
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
        )
        print(f"[OK] {name} saved to: {path}")


def download_asr_models():
    """Pre-download faster-whisper models."""
    from faster_whisper import WhisperModel
    from config import ASR_MODELS

    for name, size in ASR_MODELS.items():
        print(f"\n[DOWNLOADING] {name} (faster-whisper {size})...")
        try:
            model = WhisperModel(size, device="cpu", compute_type="int8")
            del model
            print(f"[OK] {name} ready.")
        except Exception as e:
            print(f"[WARN] {name}: {e}")


def download_kokoro():
    """Download Kokoro ONNX model files."""
    from huggingface_hub import hf_hub_download

    repo = "hexgrad/Kokoro-82M"
    files = ["kokoro-v0_19.onnx", "voices.bin"]
    kokoro_dir = os.path.join(os.path.dirname(__file__), "models", "kokoro")
    os.makedirs(kokoro_dir, exist_ok=True)

    for fname in files:
        dest = os.path.join(kokoro_dir, fname)
        if os.path.exists(dest):
            print(f"[OK] Kokoro {fname} already downloaded.")
            continue
        print(f"\n[DOWNLOADING] Kokoro {fname}...")
        try:
            hf_hub_download(
                repo_id=repo,
                filename=fname,
                local_dir=kokoro_dir,
                local_dir_use_symlinks=False,
            )
            print(f"[OK] Kokoro {fname} ready.")
        except Exception as e:
            print(f"[WARN] Kokoro {fname}: {e}")
            print("    You may need to download manually from https://huggingface.co/hexgrad/Kokoro-82M")


def download_xtts():
    """Trigger XTTS-v2 model download via TTS library."""
    print("\n[DOWNLOADING] XTTS-v2 (this may take a while, ~1.8GB)...")
    try:
        from TTS.api import TTS
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)
        del tts
        print("[OK] XTTS-v2 ready.")
    except Exception as e:
        print(f"[WARN] XTTS-v2: {e}")
        print("    Will attempt download on first use.")


def main():
    print("=" * 60)
    print("Voice Interaction System - Model Download")
    print("=" * 60)

    steps = [
        ("1/4 LLM Models (GGUF)", download_llm_models),
        ("2/4 ASR Models (Whisper)", download_asr_models),
        ("3/4 TTS - Kokoro", download_kokoro),
        ("4/4 TTS - XTTS-v2", download_xtts),
    ]

    for label, func in steps:
        print(f"\n{'='*60}")
        print(f"  Step {label}")
        print(f"{'='*60}")
        try:
            func()
        except Exception as e:
            print(f"[ERROR] {label}: {e}")
            print("  Continuing with remaining downloads...")

    print("\n" + "=" * 60)
    print("Download complete! You can now run the benchmarks.")
    print("=" * 60)


if __name__ == "__main__":
    main()
