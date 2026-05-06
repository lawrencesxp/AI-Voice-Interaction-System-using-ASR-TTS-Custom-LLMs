"""
TTS Benchmark: Kokoro vs XTTS-v2
Measures synthesis latency, audio quality (subjective), and voice cloning.
Usage: python benchmark_tts.py
"""
import time
import json
import os
import numpy as np
import soundfile as sf
from config import OUTPUT_DIR, TEST_SENTENCES, VOICE_REF_PATH, KOKORO_MODEL_PATH, KOKORO_VOICES_PATH


def benchmark_kokoro(sentences, output_dir):
    """Benchmark Kokoro TTS."""
    print(f"\n{'='*60}")
    print("Benchmarking: Kokoro-82M")
    print(f"{'='*60}")

    if not os.path.exists(KOKORO_MODEL_PATH):
        print("[ERROR] Kokoro model not found. Run download_models.py first.")
        return None

    from kokoro_onnx import Kokoro
    t0 = time.time()
    kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    results = []
    for i, text in enumerate(sentences):
        print(f"\n  Sentence {i+1}/{len(sentences)}: \"{text[:60]}...\"")

        t0 = time.time()
        samples, sr = kokoro.create(text, voice="af_bella", speed=1.0, lang="en-us")
        synth_time = time.time() - t0

        audio_duration = len(samples) / sr
        rtf = synth_time / audio_duration

        # Save audio
        path = os.path.join(output_dir, f"kokoro_sample_{i}.wav")
        sf.write(path, samples, sr)

        results.append({
            "text": text,
            "synth_time_s": synth_time,
            "audio_duration_s": audio_duration,
            "rtf": rtf,
            "output_file": path,
            "sample_rate": sr,
        })

        print(f"  Time: {synth_time:.2f}s | Duration: {audio_duration:.2f}s | RTF: {rtf:.3f}")

    avg_rtf = np.mean([r["rtf"] for r in results])
    summary = {
        "model": "Kokoro-82M",
        "load_time_s": load_time,
        "avg_rtf": avg_rtf,
        "voice_cloning": False,
        "num_samples": len(results),
        "details": results,
    }

    print(f"\n  === SUMMARY: Kokoro-82M ===")
    print(f"  Avg RTF: {avg_rtf:.3f}")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Voice cloning: Not supported (preset voices only)")

    del kokoro
    return summary


def benchmark_xtts(sentences, output_dir, voice_ref_path=None):
    """Benchmark XTTS-v2 with optional voice cloning."""
    print(f"\n{'='*60}")
    print("Benchmarking: XTTS-v2")
    print(f"{'='*60}")

    from TTS.api import TTS
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading XTTS-v2 on {device}...")

    t0 = time.time()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.2f}s")

    # --- Default voice synthesis ---
    print("\n  --- Default Voice ---")
    results_default = []
    for i, text in enumerate(sentences):
        print(f"\n  Sentence {i+1}/{len(sentences)}: \"{text[:60]}...\"")

        out_path = os.path.join(output_dir, f"xtts_default_{i}.wav")
        t0 = time.time()
        tts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker="Ana Florence",
            language="en",
        )
        synth_time = time.time() - t0

        info = sf.info(out_path)
        audio_duration = info.duration
        rtf = synth_time / audio_duration if audio_duration > 0 else 0

        results_default.append({
            "text": text,
            "synth_time_s": synth_time,
            "audio_duration_s": audio_duration,
            "rtf": rtf,
            "output_file": out_path,
            "voice_type": "default",
        })

        print(f"  Time: {synth_time:.2f}s | Duration: {audio_duration:.2f}s | RTF: {rtf:.3f}")

    # --- Voice cloning (zero-shot) ---
    results_cloned = []
    if voice_ref_path and os.path.exists(voice_ref_path):
        print(f"\n  --- Zero-Shot Voice Cloning ---")
        print(f"  Reference audio: {voice_ref_path}")

        for i, text in enumerate(sentences[:2]):  # Just 2 sentences for cloning demo
            print(f"\n  Cloned sentence {i+1}: \"{text[:60]}...\"")

            out_path = os.path.join(output_dir, f"xtts_cloned_{i}.wav")
            t0 = time.time()
            tts.tts_to_file(
                text=text,
                file_path=out_path,
                speaker_wav=voice_ref_path,
                language="en",
            )
            synth_time = time.time() - t0

            info = sf.info(out_path)
            audio_duration = info.duration
            rtf = synth_time / audio_duration if audio_duration > 0 else 0

            results_cloned.append({
                "text": text,
                "synth_time_s": synth_time,
                "audio_duration_s": audio_duration,
                "rtf": rtf,
                "output_file": out_path,
                "voice_type": "zero_shot_clone",
            })

            print(f"  Time: {synth_time:.2f}s | Duration: {audio_duration:.2f}s | RTF: {rtf:.3f}")
    else:
        print(f"\n  [SKIP] Voice cloning - no reference audio found at {voice_ref_path}")
        print(f"  Run record_voice.py first to create a reference recording.")

    all_results = results_default + results_cloned
    avg_rtf_default = np.mean([r["rtf"] for r in results_default]) if results_default else 0
    avg_rtf_cloned = np.mean([r["rtf"] for r in results_cloned]) if results_cloned else 0

    summary = {
        "model": "XTTS-v2",
        "device": device,
        "load_time_s": load_time,
        "avg_rtf_default": avg_rtf_default,
        "avg_rtf_cloned": avg_rtf_cloned,
        "voice_cloning": True,
        "cloning_type": "zero-shot",
        "num_default_samples": len(results_default),
        "num_cloned_samples": len(results_cloned),
        "details": all_results,
    }

    print(f"\n  === SUMMARY: XTTS-v2 ===")
    print(f"  Avg RTF (default): {avg_rtf_default:.3f}")
    if results_cloned:
        print(f"  Avg RTF (cloned):  {avg_rtf_cloned:.3f}")
    print(f"  Load time: {load_time:.2f}s")
    print(f"  Voice cloning: Zero-shot supported")

    del tts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main():
    print("=" * 60)
    print("TTS Benchmark: Kokoro-82M vs XTTS-v2")
    print("=" * 60)

    tts_output = os.path.join(OUTPUT_DIR, "tts_benchmark")
    os.makedirs(tts_output, exist_ok=True)

    # Use first 3 sentences for benchmarking
    sentences = TEST_SENTENCES[:3]

    all_results = {}

    # Kokoro
    try:
        result = benchmark_kokoro(sentences, tts_output)
        if result:
            all_results["kokoro"] = result
    except Exception as e:
        print(f"[ERROR] Kokoro: {e}")

    # XTTS-v2
    try:
        result = benchmark_xtts(sentences, tts_output, voice_ref_path=VOICE_REF_PATH)
        if result:
            all_results["xtts-v2"] = result
    except Exception as e:
        print(f"[ERROR] XTTS-v2: {e}")

    # Save results
    results_path = os.path.join(tts_output, "tts_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Model':<15} {'Avg RTF':<12} {'Load Time':<12} {'Voice Clone':<12}")
    print("-" * 51)
    for name, r in all_results.items():
        rtf = r.get("avg_rtf", r.get("avg_rtf_default", 0))
        clone = "Yes" if r.get("voice_cloning") else "No"
        print(f"{name:<15} {rtf:<12.3f} {r['load_time_s']:<12.2f}s {clone:<12}")

    print(f"\nAudio samples saved in: {tts_output}")
    print("Listen to them to subjectively compare quality!")


if __name__ == "__main__":
    main()