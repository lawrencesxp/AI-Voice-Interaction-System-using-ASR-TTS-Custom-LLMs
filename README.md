# AI-Voice-Interaction-System-using-ASR-TTS-Custom-LLMs

A voice interaction system combining Whisper ASR, Qwen2.5 LLM (via llama.cpp with Q4_K_M 
quantization), and Kokoro/XTTS-v2 TTS was successfully designed, deployed, and evaluated on 
an RTX 2060 workstation. The system supports both question-answering and translation modes 
through a Gradio web interface. Component benchmarking revealed consistent accuracy-speed 
tradeoffs, and the integrated pipeline achieved total latencies of 4.6 seconds for translation and 
25.7 seconds for Q&A with longer responses. TTS synthesis was identified as the primary 
bottleneck for extended responses, suggesting streaming synthesis as the most impactful 
optimization for future work.
