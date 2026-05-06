"""
ASR Benchmark: Whisper-tiny vs Whisper-base
Measures Word Error Rate (WER) and inference latency.
Usage: python benchmark_asr.py
"""
import time
import json
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from jiwer import wer
from faster_whisper import WhisperModel
from config import ASR_MODELS, OUTPUT_DIR, SAMPLE_RATE, TEST_SENTENCES


def record_audio(duration=5, sr=SAMPLE_RATE):
    """Record audio from microphone."""
    print(f"  Recording for {duration} seconds... SPEAK NOW!")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("  Recording done.")
    return audio.flatten()


def generate_test_audio_via_tts(sentences, output_dir):
    """Generate test audio files using Kokoro TTS for ASR benchmarking."""
    audio_files = []
    try:
        from config import KOKORO_MODEL_PATH, KOKORO_VOICES_PATH

        if not os.path.exists(KOKORO_MODEL_PATH):
            print("  Kokoro model not found. Falling back to mic recording.")
            return None

        from kokoro_onnx import Kokoro
        kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)

        for i, text in enumerate(sentences):
            print(f"  Generating audio for sentence {i+1}/{len(sentences)}...")
            samples, sr = kokoro.create(text, voice="af_bella", speed=1.0, lang="en-us")
            path = os.path.join(output_dir, f"test_sentence_{i}.wav")
            sf.write(path, samples, sr)
            audio_files.append((path, text, sr))

        del kokoro
        return audio_files

    except Exception as e:
        print(f"  TTS generation failed: {e}")
        return None


def benchmark_model(model_name, model_size, test_data, device="cpu", compute_type="int8"):
    """Benchmark a single ASR model."""
    print(f"\n{'─'*50}")
    print(f"Testing: {model_name} (device={device}, compute={compute_type})")
    print(f"{'─'*50}")

    # Load model
    t0 = time.time()
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.2f}s (device={device})")

    results = []
    total_audio_duration = 0
    total_inference_time = 0

    for audio_path, reference_text, sr in test_data:
        # Transcribe
        t0 = time.time()
        segments, info = model.transcribe(audio_path, beam_size=5)
        transcript = " ".join([s.text.strip() for s in segments])
        inference_time = time.time() - t0

        # Get audio duration
        audio_info = sf.info(audio_path)
        audio_duration = audio_info.duration
        total_audio_duration += audio_duration
        total_inference_time += inference_time

        # Calculate WER
        error_rate = wer(reference_text.lower(), transcript.lower())

        results.append({
            "reference": reference_text,
            "transcript": transcript,
            "wer": error_rate,
            "inference_time": inference_time,
            "audio_duration": audio_duration,
            "rtf": inference_time / audio_duration,  # Real-time factor
        })

        print(f"\n  Reference:  {reference_text}")
        print(f"  Transcript: {transcript}")
        print(f"  WER: {error_rate:.2%} | Time: {inference_time:.2f}s | RTF: {inference_time/audio_duration:.3f}")

    # Summary
    avg_wer = np.mean([r["wer"] for r in results])
    avg_rtf = total_inference_time / total_audio_duration if total_audio_duration > 0 else 0

    summary = {
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "load_time_s": load_time,
        "avg_wer": avg_wer,
        "avg_rtf": avg_rtf,
        "total_inference_time_s": total_inference_time,
        "total_audio_duration_s": total_audio_duration,
        "num_samples": len(results),
        "details": results,
    }

    print(f"\n  === SUMMARY: {model_name} ===")
    print(f"  Avg WER: {avg_wer:.2%}")
    print(f"  Avg RTF: {avg_rtf:.3f} (lower = faster, <1 means real-time)")
    print(f"  Load time: {load_time:.2f}s")

    del model
    return summary


def main():
    print("=" * 60)
    print("ASR Benchmark: Whisper-tiny vs Whisper-base")
    print("=" * 60)

    asr_output = os.path.join(OUTPUT_DIR, "asr_benchmark")
    os.makedirs(asr_output, exist_ok=True)

    # Generate or record test audio
    print("\nPreparing test audio...")
    test_data = generate_test_audio_via_tts(TEST_SENTENCES, asr_output)

    if test_data is None:
        print("\nFalling back to microphone recording.")
        print("You will read 3 sentences aloud.\n")
        test_data = []
        for i, sentence in enumerate(TEST_SENTENCES[:3]):
            print(f"\nSentence {i+1}: \"{sentence}\"")
            input("  Press Enter when ready to record (5 seconds)...")
            audio = record_audio(duration=5)
            path = os.path.join(asr_output, f"mic_test_{i}.wav")
            sf.write(path, audio, SAMPLE_RATE)
            test_data.append((path, sentence, SAMPLE_RATE))

    # Benchmark each model
    all_results = {}
    for model_name, model_size in ASR_MODELS.items():
        result = benchmark_model(model_name, model_size, test_data)
        all_results[model_name] = result

    # Save results
    results_path = os.path.join(asr_output, "asr_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Avg WER':<12} {'Avg RTF':<12} {'Load Time':<12}")
    print("-" * 56)
    for name, r in all_results.items():
        print(f"{name:<20} {r['avg_wer']:<12.2%} {r['avg_rtf']:<12.3f} {r['load_time_s']:<12.2f}s")


if __name__ == "__main__":
    main()