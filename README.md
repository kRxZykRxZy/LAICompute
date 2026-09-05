# LAICompute

LAICompute v0.1 is a cache-aware CPU compute foundation for local AI inference.

## v0.1 milestones

- **M1 — FP16 tensor/storage layer:** IEEE-754 binary16 storage with float conversion; FP16 tensors use 2 bytes/element.
- **M2 — tiled matrix multiplication:** cache-derived tile sizing, FP32 and FP16-storage/FP32-accumulation kernels, multithreading.
- **M3 — CPU SIMD:** AVX kernel when the binary is compiled for AVX; runtime CPU feature discovery reports AVX/F16C/AVX2/FMA.
- **M4 — cache-aware pipeline:** current/next/future tile prefetching. Hardware caches remain hardware-managed; LAICompute uses locality, blocking and software prefetch.
- **M5 — benchmarking:** CPU/cache topology, working-set targets, throughput and checksums.
- **M6 — GGUF loader:** reads GGUF v2/v3 metadata and tensor tables and supports F32/F16 tensor values plus Q4_0/Q8_0 value decoding.
- **M7 — token generation:** GPT-2 GGUF tokenizer plus Llama-family RMSNorm, RoPE, GQA, SwiGLU, KV cache and greedy/temperature/top-k autoregressive generation.

## M6/M7 model support

The first runtime target is **Llama-architecture GGUF models using the GGUF GPT-2 tokenizer**. This matches SmolLM-135M's published GGUF architecture and tokenizer metadata. The inference path currently executes F32/F16, Q4_0 and Q8_0 tensor values; other GGML quantization formats are detected by the loader but intentionally rejected by inference until their dequantizers are implemented.

For a full-precision SmolLM-135M GGUF, run:

```bash
cmake -S . -B build -DLAIC_ENABLE_NATIVE=ON
cmake --build build -j
./build/laic_generate SmolLM-135M.gguf "Hello, how are you?" 32 0
```

Arguments are `MODEL PROMPT MAX_TOKENS TEMPERATURE TOP_K`. Temperature `0` selects greedy decoding.

## Runtime cache detection

Cache capacities are **not hardcoded**. LAICompute reads CPU cache information from x86 CPUID and, on Linux, `/sys/devices/system/cpu/.../cache`. Working-set budgets are derived from detected cache capacities.

## Build and test

```bash
cmake -S . -B build -DLAIC_ENABLE_NATIVE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/laic_bench
```

GitHub Actions automatically builds and tests pushes and pull requests with GCC and Clang.
