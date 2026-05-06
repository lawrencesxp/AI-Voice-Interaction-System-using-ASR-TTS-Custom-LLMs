"""
Gradio Web UI for Voice Interaction System
Usage: python app.py
"""
import os
import time
import tempfile
import gradio as gr
import soundfile as sf
import numpy as np

from config import (
    MODEL_DIR, OUTPUT_DIR, VOICE_REF_PATH,
    LLM_MODELS, LLM_GPU_LAYERS, LLM_CONTEXT_SIZE, LLM_MAX_TOKENS,
    QA_SYSTEM_PROMPT, TRANSLATION_SYSTEM_PROMPT,
    KOKORO_MODEL_PATH, KOKORO_VOICES_PATH,
)

# ============================================================
# GLOBALS (loaded once at startup)
# ============================================================
asr_model = None
llm_model = None
tts_kokoro = None
tts_xtts = None


def load_models():
    """Load all models at startup."""
    global asr_model, llm_model, tts_kokoro

    # ASR
    print("[1/3] Loading ASR (Whisper-base)...")
    from faster_whisper import WhisperModel
    asr_model = WhisperModel("base", device="cpu", compute_type="int8")
    print("  ASR ready.")

    # LLM
    print("[2/3] Loading LLM (Qwen2.5-1.5B Q4_K_M)...")
    from llama_cpp import Llama
    model_path = os.path.join(MODEL_DIR, LLM_MODELS["qwen2.5-1.5b"]["file"])
    llm_model = Llama(
        model_path=model_path,
        n_gpu_layers=LLM_GPU_LAYERS,
        n_ctx=LLM_CONTEXT_SIZE,
        verbose=False,
    )
    print("  LLM ready.")

    # TTS - Kokoro (default, fast)
    print("[3/3] Loading TTS (Kokoro)...")
    try:
        from kokoro_onnx import Kokoro
        tts_kokoro = Kokoro(KOKORO_MODEL_PATH, KOKORO_VOICES_PATH)
        print("  Kokoro TTS ready.")
    except Exception as e:
        print(f"  Kokoro failed: {e}")

    print("\nAll models loaded! Starting UI...\n")


def load_xtts_if_needed():
    """Lazy-load XTTS-v2 only when voice cloning is requested."""
    global tts_xtts
    if tts_xtts is None:
        print("Loading XTTS-v2 for voice cloning (one-time)...")
        from TTS.api import TTS
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_xtts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        print("  XTTS-v2 ready.")
    return tts_xtts


def transcribe_audio(audio_path):
    """ASR step."""
    if audio_path is None:
        return "", 0
    segments, _ = asr_model.transcribe(audio_path, beam_size=5)
    text = " ".join([s.text.strip() for s in segments])
    return text


def generate_llm_response(text, mode, target_lang):
    """LLM step."""
    if mode == "translate":
        system = TRANSLATION_SYSTEM_PROMPT.format(target_lang=target_lang)
    else:
        system = QA_SYSTEM_PROMPT

    response = llm_model.create_chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.7,
    )
    return response["choices"][0]["message"]["content"]


def synthesize_tts(text, tts_choice, voice_ref):
    """TTS step."""
    out_path = os.path.join(OUTPUT_DIR, "ui_output.wav")

    if tts_choice == "XTTS-v2 (Voice Cloning)" and voice_ref is not None:
        xtts = load_xtts_if_needed()
        xtts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker_wav=voice_ref,
            language="en",
        )
    elif tts_choice == "XTTS-v2 (Default Voice)":
        xtts = load_xtts_if_needed()
        xtts.tts_to_file(
            text=text,
            file_path=out_path,
            speaker="Ana Florence",
            language="en",
        )
    else:
        # Kokoro (default)
        if tts_kokoro is None:
            return None
        samples, sr = tts_kokoro.create(text, voice="af_bella", speed=1.0, lang="en-us")
        sf.write(out_path, samples, sr)

    return out_path


def process_voice(audio, mode, target_lang, tts_choice, voice_ref):
    """Full pipeline: audio -> transcription -> LLM -> TTS."""
    if audio is None:
        return "No audio received.", "", None, ""

    t_start = time.time()

    # Step 1: ASR
    t0 = time.time()
    transcript = transcribe_audio(audio)
    asr_time = time.time() - t0

    if not transcript.strip():
        return "Could not transcribe audio. Try speaking louder.", "", None, ""

    # Step 2: LLM
    t0 = time.time()
    response = generate_llm_response(transcript, mode, target_lang)
    llm_time = time.time() - t0

    # Step 3: TTS
    t0 = time.time()
    audio_out = synthesize_tts(response, tts_choice, voice_ref)
    tts_time = time.time() - t0

    total = time.time() - t_start

    timing_str = (
        f"ASR: {asr_time:.2f}s | LLM: {llm_time:.2f}s | "
        f"TTS: {tts_time:.2f}s | Total: {total:.2f}s"
    )

    return transcript, response, audio_out, timing_str


# ============================================================
# GRADIO INTERFACE
# ============================================================
def build_ui():
    with gr.Blocks(title="Voice Interaction System", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Voice Interaction System")
        gr.Markdown("**EECE7398 Deep Learning Embedded Systems - Project 2**")
        gr.Markdown("Speak into your microphone → ASR (Whisper) → LLM (Qwen2.5) → TTS (Kokoro/XTTS-v2)")

        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record your voice",
                )
                mode = gr.Radio(
                    choices=["qa", "translate"],
                    value="qa",
                    label="Mode",
                    info="Q&A: Ask a question | Translate: Speak in English to translate",
                )
                target_lang = gr.Dropdown(
                    choices=["Spanish", "French", "German", "Chinese", "Japanese",
                             "Korean", "Italian", "Portuguese", "Arabic", "Hindi"],
                    value="Spanish",
                    label="Target Language (for translation mode)",
                    visible=True,
                )
                tts_choice = gr.Radio(
                    choices=["Kokoro (Fast)", "XTTS-v2 (Default Voice)", "XTTS-v2 (Voice Cloning)"],
                    value="Kokoro (Fast)",
                    label="TTS Engine",
                )
                voice_ref = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Voice Reference (for cloning, ~10s clip)",
                    visible=True,
                )
                submit_btn = gr.Button("Process", variant="primary", size="lg")

            with gr.Column(scale=1):
                transcript_out = gr.Textbox(label="Transcription (ASR output)", lines=3)
                response_out = gr.Textbox(label="LLM Response", lines=5)
                audio_output = gr.Audio(label="Spoken Response (TTS output)", type="filepath")
                timing_out = gr.Textbox(label="Timing Breakdown", lines=1)

        submit_btn.click(
            fn=process_voice,
            inputs=[audio_input, mode, target_lang, tts_choice, voice_ref],
            outputs=[transcript_out, response_out, audio_output, timing_out],
        )

    return demo


if __name__ == "__main__":
    load_models()
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )