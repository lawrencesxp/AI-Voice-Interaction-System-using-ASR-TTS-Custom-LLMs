"""
LLM Benchmark: Qwen2.5-0.5B vs Qwen2.5-1.5B (Q4_K_M via llama.cpp)
Measures tokens/sec, time-to-first-token, and response quality.
Usage: python benchmark_llm.py
"""
import time
import json
import os
from llama_cpp import Llama
from config import LLM_MODELS, MODEL_DIR, OUTPUT_DIR, LLM_GPU_LAYERS, LLM_CONTEXT_SIZE, LLM_MAX_TOKENS, QA_SYSTEM_PROMPT, LLM_TEST_PROMPTS


def load_model(model_name, model_info):
    """Load a GGUF model with llama.cpp."""
    model_path = os.path.join(MODEL_DIR, model_info["file"])
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun download_models.py first.")

    print(f"  Loading {model_name} from {model_path}...")
    t0 = time.time()
    model = Llama(
        model_path=model_path,
        n_gpu_layers=LLM_GPU_LAYERS,
        n_ctx=LLM_CONTEXT_SIZE,
        verbose=False,
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.2f}s")
    return model, load_time


def benchmark_prompt(model, prompt, system_prompt=QA_SYSTEM_PROMPT, max_tokens=LLM_MAX_TOKENS):
    """Run a single prompt and measure performance."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    t0 = time.time()
    response = model.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )
    total_time = time.time() - t0

    reply = response["choices"][0]["message"]["content"]
    usage = response.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    tokens_per_sec = completion_tokens / total_time if total_time > 0 else 0

    return {
        "prompt": prompt,
        "response": reply,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time_s": total_time,
        "tokens_per_sec": tokens_per_sec,
    }


def benchmark_model(model_name, model_info, prompts):
    """Full benchmark for one model."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")

    model, load_time = load_model(model_name, model_info)

    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n  Prompt {i+1}/{len(prompts)}: \"{prompt}\"")
        result = benchmark_prompt(model, prompt)
        results.append(result)

        print(f"  Response: {result['response'][:200]}...")
        print(f"  Tokens: {result['completion_tokens']} | "
              f"Time: {result['total_time_s']:.2f}s | "
              f"Speed: {result['tokens_per_sec']:.1f} tok/s")

    # Summary stats
    avg_tps = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_time = sum(r["total_time_s"] for r in results) / len(results)
    avg_tokens = sum(r["completion_tokens"] for r in results) / len(results)

    summary = {
        "model": model_name,
        "quantization": "Q4_K_M",
        "repo": model_info["repo"],
        "load_time_s": load_time,
        "avg_tokens_per_sec": avg_tps,
        "avg_response_time_s": avg_time,
        "avg_completion_tokens": avg_tokens,
        "gpu_layers": LLM_GPU_LAYERS,
        "context_size": LLM_CONTEXT_SIZE,
        "details": results,
    }

    print(f"\n  === SUMMARY: {model_name} ===")
    print(f"  Avg Speed: {avg_tps:.1f} tokens/sec")
    print(f"  Avg Response Time: {avg_time:.2f}s")
    print(f"  Avg Completion Tokens: {avg_tokens:.0f}")
    print(f"  Load Time: {load_time:.2f}s")

    del model
    return summary


def main():
    print("=" * 60)
    print("LLM Benchmark: Qwen2.5-0.5B vs Qwen2.5-1.5B (Q4_K_M)")
    print("=" * 60)

    llm_output = os.path.join(OUTPUT_DIR, "llm_benchmark")
    os.makedirs(llm_output, exist_ok=True)

    all_results = {}
    for model_name, model_info in LLM_MODELS.items():
        try:
            result = benchmark_model(model_name, model_info, LLM_TEST_PROMPTS)
            all_results[model_name] = result
        except Exception as e:
            print(f"\n[ERROR] {model_name}: {e}")
            continue

    # Save results
    results_path = os.path.join(llm_output, "llm_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")

    # Comparison table
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Model':<20} {'Tok/s':<10} {'Avg Time':<12} {'Load Time':<12}")
    print("-" * 54)
    for name, r in all_results.items():
        print(f"{name:<20} {r['avg_tokens_per_sec']:<10.1f} "
              f"{r['avg_response_time_s']:<12.2f}s {r['load_time_s']:<12.2f}s")

    # Show quality comparison
    print(f"\n{'='*60}")
    print("RESPONSE QUALITY COMPARISON")
    print(f"{'='*60}")
    for prompt in LLM_TEST_PROMPTS:
        print(f"\nQ: {prompt}")
        for name, r in all_results.items():
            for d in r["details"]:
                if d["prompt"] == prompt:
                    print(f"  [{name}]: {d['response'][:300]}")
                    break


if __name__ == "__main__":
    main()
