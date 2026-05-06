"""
Record a voice reference clip for voice cloning.
Usage: python record_voice.py
"""
import sounddevice as sd
import soundfile as sf
import numpy as np
from config import VOICE_REF_PATH, SAMPLE_RATE


def list_devices():
    """Show available audio devices."""
    print("\nAvailable audio devices:")
    print(sd.query_devices())
    print(f"\nDefault input: {sd.default.device[0]}")


def record_reference(duration=10, sr=24000):
    """Record a voice reference clip."""
    print("=" * 60)
    print("Voice Reference Recording")
    print("=" * 60)
    print(f"\nThis will record {duration} seconds of your voice.")
    print("For best results:")
    print("  - Speak clearly and naturally")
    print("  - Read a sentence or paragraph aloud")
    print("  - Minimize background noise")
    print(f"\nSuggested text to read:")
    print('  "The quick brown fox jumps over the lazy dog.')
    print('   Artificial intelligence is rapidly transforming')
    print('   the way we live and work in the modern world."')

    input(f"\nPress Enter to start recording ({duration} seconds)...")

    print(f"\n>>> RECORDING... Speak now! ({duration}s) <<<")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print(">>> Recording complete! <<<")

    # Normalize
    audio = audio.flatten()
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    # Check if audio has signal
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 0.01:
        print("\n[WARNING] Audio seems very quiet. Check your microphone.")

    sf.write(VOICE_REF_PATH, audio, sr)
    print(f"\nSaved to: {VOICE_REF_PATH}")
    print(f"Duration: {len(audio)/sr:.1f}s | Sample rate: {sr}Hz")

    return VOICE_REF_PATH


if __name__ == "__main__":
    list_devices()
    record_reference()
