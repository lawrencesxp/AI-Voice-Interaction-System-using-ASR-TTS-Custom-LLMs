# Voice Interaction System: Technical Report
## EECE7398 Deep Learning Embedded Systems — Project 2

---

## REPORT OUTLINE (fill in with your results)

### 1. Introduction (~0.5 page)
- Motivation for voice interaction systems
- Overview of the three core components: ASR, LLM, TTS
- Deployment target: RTX 2060, Windows 10, real-time constraint

### 2. System Architecture (~1 page)
- Pipeline diagram: Microphone → ASR → LLM → TTS → Speaker
- Data flow: audio waveform → text → text → audio waveform
- Design decisions: model selection rationale, quantization strategy

### 3. Component Selection and Evaluation (~3-4 pages)

#### 3.1 Automatic Speech Recognition (ASR)
- Models tested: Whisper-tiny (39M params), Whisper-base (74M params)
- Framework: faster-whisper (CTranslate2 backend)
- Metrics: Word Error Rate (WER), Real-Time Factor (RTF), load time
- Results table (from benchmark_asr.py output)
- Analysis: accuracy vs speed tradeoff

#### 3.2 Large Language Model (LLM)
- Models tested: Qwen2.5-0.5B-Instruct, Qwen2.5-1.5B-Instruct
- Deployment: llama.cpp with Q4_K_M quantization
- GPU offloading: all layers on RTX 2060
- Metrics: tokens/sec, response latency, qualitative quality
- Results table (from benchmark_llm.py output)
- Analysis: model size vs quality vs speed

#### 3.3 Text-to-Speech (TTS)
- Models tested: Kokoro-82M (ONNX), XTTS-v2
- Metrics: Real-Time Factor (RTF), load time, subjective quality
- Voice cloning: zero-shot with XTTS-v2
  - Method: 10-second reference clip, no fine-tuning
  - Results: cloning quality assessment
  - Discussion: LoRA-based fine-tuning as future work
    (would require 3-5 min reference audio, ~30-60 min training)
- Results table (from benchmark_tts.py output)

### 4. System Integration (~2-3 pages)

#### 4.1 Pipeline Design
- How ASR, LLM, TTS are connected
- Memory management: VRAM allocation across models
- Prompt engineering for Q&A and translation modes

#### 4.2 End-to-End Performance
- Total latency breakdown: ASR time + LLM time + TTS time
- Bottleneck analysis: which component dominates?
- Comparison across configurations (e.g., Kokoro vs XTTS-v2 in pipeline)

#### 4.3 User Interface
- Gradio-based web interface
- Features: mic recording, mode selection, TTS engine selection, voice cloning
- Screenshot(s)

#### 4.4 Prompt Engineering
- System prompts for Q&A mode (concise, spoken-friendly answers)
- System prompts for translation mode (output-only translation)
- How prompt design affects response quality and latency

### 5. Discussion (~1 page)
- Key findings and tradeoffs
- Limitations: VRAM constraints, model size, real-time factor
- Future work: streaming ASR, interrupt handling, LoRA voice cloning

### 6. Conclusion (~0.5 page)

### References

---

## TABLES TO FILL IN

### Table 1: ASR Comparison
| Model        | Params | WER (%) | RTF   | Load Time (s) | Device |
|-------------|--------|---------|-------|---------------|--------|
| Whisper-tiny | 39M    |         |       |               |        |
| Whisper-base | 74M    |         |       |               |        |

### Table 2: LLM Comparison
| Model            | Quant  | Tok/s | Avg Latency (s) | Load Time (s) | VRAM (MB) |
|-----------------|--------|-------|-----------------|---------------|-----------|
| Qwen2.5-0.5B   | Q4_K_M |       |                 |               |           |
| Qwen2.5-1.5B   | Q4_K_M |       |                 |               |           |

### Table 3: TTS Comparison
| Model    | RTF (default) | RTF (cloned) | Load Time (s) | Voice Cloning |
|----------|--------------|-------------|---------------|---------------|
| Kokoro   |              | N/A         |               | No            |
| XTTS-v2  |              |             |               | Yes (zero-shot)|

### Table 4: End-to-End Pipeline
| Configuration              | ASR (s) | LLM (s) | TTS (s) | Total (s) |
|---------------------------|---------|---------|---------|-----------|
| Whisper-base + Qwen-1.5B + Kokoro  |   |    |    |           |
| Whisper-base + Qwen-1.5B + XTTS-v2 |   |    |    |           |
