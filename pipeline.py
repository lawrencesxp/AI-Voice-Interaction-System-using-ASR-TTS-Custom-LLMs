"""
Integrated Voice Interaction Pipeline: ASR -> LLM -> TTS
Supports Q&A mode and Translation mode.
Usage: python pipeline.py
"""
import time
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
from config import (
    MODEL_DIR, OUTPUT_DIR, VOICE_REF_PATH, SAMPLE_RATE,
    LLM_MODELS, LLM_GPU_LAYERS, LLM_CONTEXT_SIZE, LLM_MAX_TOKENS,
    QA_SYSTEM_PROMPT, TRANSLATION_SYSTEM_PROMPT,
    KOKORO_MODEL_PATH, KOKORO_VOICES_PATH,
)


class VoiceInteractionPipeline:
    """End-to-end voice interaction system."""

    def __init__(
        self,
        asr_model_size="base",
        llm_model_name="qwen2.5-1.5b",
        tts_engine="kokoro",
        use_voice_cloning=False,
        voice_ref_path=None,
    ):
        self.tts_engine = tts_engine
        self.use_voice_cloning = use_voice_cloning
        self.voice_ref_path = voice_ref_path or VOICE_REF_PATH
        self.timings = {}

        print("=" * 60)
        print("Initializing Voice Interaction Pipeline")
        print("=" * 60)

        # Load ASR
        self._load_asr(asr_model_size)

        # Load LLM
        self._load_llm(llm_model_name)

        # Load TTS
        self._load_tts(tts_engine)

        print("\n[READY] Pipeline initialized successfully!")

    def _load_asr(self, model_size):
        print(f"\n[1/3] Loading ASR: Whisper-{model_size}...")
        from faster_whisper import WhisperModel
        t0 = time.time()
        self.asr = WhisperModel(model_size, device="cpu", compute_type="int8")
        self.asr_device = "cpu"
        self.timings["asr_load"] = time.time() - t0
        print(f"  ASR loaded in {self.timings['asr_load']:.2f}s ({self.asr_device})")

    def _load_llm(self, model_name):
        print(f"\n[2/3] Loading LLM: {model_name} (Q4_K_M)...")
        from llama_cpp import Llama
        model_info = LLM_MODELS[model_name]
        model_path = os.path.join(MODEL_DIR, model_info["file"])
        t0 = time.time()
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=LLM_GPU_LAYERS,
            n_ctx=LLM_CONTEXT_SIZE,
            verbose=False,
        )
        self.timings["llm_load"] = time.time() - t0
        print(f"  LLM loaded in {self.timings['llm_load']:.2f}s")

    def _load_tts(self, engine):
        t0 = time.time()
        if engine == "kokoro":
            print(f"\n[3/3] Loading TTS: Kokoro-82M...")
            from kokoro_onnx import Kokoro
            self.tts = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
            self.tts_type = "kokoro"
        elif engine == "xtts":
            print(f"\n[3/3] Loading TTS: XTTS-v2...")
            from TTS.api import TTS
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
            self.tts_type = "xtts"
        else:
            raise ValueError(f"Unknown TTS engine: {engine}")
        self.timings["tts_load"] = time.time() - t0
        print(f"  TTS loaded in {self.timings['tts_load']:.2f}s")

    def transcribe(self, audio_path):
        """ASR: Audio file -> text."""
        t0 = time.time()
        segments, info = self.asr.transcribe(audio_path, beam_size=5)
        text = " ".join([s.text.strip() for s in segments])
        elapsed = time.time() - t0
        return text, elapsed

    def generate_response(self, text, mode="qa", target_lang="Spanish"):
        """LLM: Generate response text."""
        if mode == "qa":
            system = QA_SYSTEM_PROMPT
        elif mode == "translate":
            system = TRANSLATION_SYSTEM_PROMPT.format(target_lang=target_lang)
        else:
            system = QA_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ]

        t0 = time.time()
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=LLM_MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
        )
        elapsed = time.time() - t0

        reply = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {}).get("completion_tokens", 0)
        return reply, elapsed, tokens

    def synthesize_speech(self, text, output_path=None):
        """TTS: Text -> audio."""
        if output_path is None:
            output_path = os.path.join(OUTPUT_DIR, "pipeline_output.wav")

        t0 = time.time()
        if self.tts_type == "kokoro":
            samples, sr = self.tts.create(text, voice="af_bella", speed=1.0, lang="en-us")
            sf.write(output_path, samples, sr)
        elif self.tts_type == "xtts":
            if self.use_voice_cloning and os.path.exists(self.voice_ref_path):
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker_wav=self.voice_ref_path,
                    language="en",
                )
            else:
                self.tts.tts_to_file(
                    text=text,
                    file_path=output_path,
                    speaker="Ana Florence",
                    language="en",
                )
        elapsed = time.time() - t0
        return output_path, elapsed

    def run(self, audio_path, mode="qa", target_lang="Spanish"):
        """Full pipeline: audio -> response audio."""
        print(f"\n{'─'*50}")
        print(f"Pipeline run (mode={mode})")
        print(f"{'─'*50}")

        total_start = time.time()

        # ASR
        print("\n  [ASR] Transcribing...")
        transcript, asr_time = self.transcribe(audio_path)
        print(f"  Transcript: \"{transcript}\"")
        print(f"  ASR time: {asr_time:.2f}s")

        # LLM
        print("\n  [LLM] Generating response...")
        response, llm_time, tokens = self.generate_response(transcript, mode, target_lang)
        print(f"  Response: \"{response[:200]}\"")
        print(f"  LLM time: {llm_time:.2f}s ({tokens} tokens)")

        # TTS
        print("\n  [TTS] Synthesizing speech...")
        output_path, tts_time = self.synthesize_speech(response)
        print(f"  Output: {output_path}")
        print(f"  TTS time: {tts_time:.2f}s")

        total_time = time.time() - total_start

        result = {
            "input_audio": audio_path,
            "transcript": transcript,
            "response_text": response,
            "output_audio": output_path,
            "mode": mode,
            "timings": {
                "asr_s": asr_time,
                "llm_s": llm_time,
                "tts_s": tts_time,
                "total_s": total_time,
            },
            "llm_tokens": tokens,
        }

        print(f"\n  === TIMING BREAKDOWN ===")
        print(f"  ASR:   {asr_time:.2f}s")
        print(f"  LLM:   {llm_time:.2f}s")
        print(f"  TTS:   {tts_time:.2f}s")
        print(f"  TOTAL: {total_time:.2f}s")

        return result

    def run_from_mic(self, duration=5, mode="qa", target_lang="Spanish"):
        """Record from mic and run pipeline."""
        print(f"\nRecording for {duration} seconds... SPEAK NOW!")
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        print("Recording done.")

        temp_path = os.path.join(OUTPUT_DIR, "mic_input.wav")
        sf.write(temp_path, audio.flatten(), SAMPLE_RATE)

        return self.run(temp_path, mode, target_lang)


def play_audio(path):
    """Play a WAV file."""
    data, sr = sf.read(path)
    sd.play(data, sr)
    sd.wait()


def main():
    """Interactive demo."""
    print("=" * 60)
    print("Voice Interaction System - Interactive Demo")
    print("=" * 60)

    # Initialize with best models from benchmarks
    pipeline = VoiceInteractionPipeline(
        asr_model_size="base",
        llm_model_name="qwen2.5-1.5b",
        tts_engine="kokoro",        # Fast; switch to "xtts" for voice cloning
        use_voice_cloning=False,
    )

    while True:
        print("\n" + "=" * 60)
        print("Options:")
        print("  1. Q&A (speak a question)")
        print("  2. Translate (speak in English -> translated)")
        print("  3. Change TTS engine")
        print("  4. Quit")
        choice = input("\nChoice: ").strip()

        if choice == "1":
            input("Press Enter to start recording (5 seconds)...")
            result = pipeline.run_from_mic(duration=5, mode="qa")
            print("\nPlaying response...")
            play_audio(result["output_audio"])

        elif choice == "2":
            lang = input("Target language (e.g., Spanish, French, Chinese): ").strip() or "Spanish"
            input("Press Enter to start recording (5 seconds)...")
            result = pipeline.run_from_mic(duration=5, mode="translate", target_lang=lang)
            print("\nPlaying response...")
            play_audio(result["output_audio"])

        elif choice == "3":
            eng = input("TTS engine (kokoro/xtts): ").strip()
            if eng in ("kokoro", "xtts"):
                pipeline._load_tts(eng)
                if eng == "xtts":
                    clone = input("Use voice cloning? (y/n): ").strip().lower()
                    pipeline.use_voice_cloning = (clone == "y")

        elif choice == "4":
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()