# LAICompute

LAICompute v0.1 is a cache-aware CPU compute foundation for local AI inference.

## v0.1 milestones

- **M1 — FP16 tensor/storage layer:** IEEE-754 binary16 storage with float conversion; FP16 tensors use 2 bytes/element.
- **M2 — tiled matrix multiplication:** cache-derived tile sizing, FP32 and FP16-storage/FP32-accumulation kernels, multithreading.
- **M3 — CPU SIMD:** AVX kernel when the binary is compiled for AVX; runtime CPU feature discovery reports AVX/F16C/AVX2/FMA. No AVX2 instructions are required by the baseline kernel.
- **M4 — cache-aware pipeline:** current/next/future tile prefetching. Hardware caches remain hardware-managed; LAICompute uses locality, blocking and software prefetch rather than pretending it can assign data directly to L1/L2/L3.
- **M5 — benchmarking:** reports CPU/cache topology, working-set targets, FP32/FP16-storage throughput and checksums. Linux users can pair it with `perf stat` for hardware cache counters.

## Runtime cache detection

Cache capacities are **not hardcoded**. LAICompute reads CPU cache information from x86 CPUID and, on Linux, `/sys/devices/system/cpu/.../cache`. Working-set budgets are derived from the detected L1D/L2/L3 capacities for the machine running the program.

## Build

```bash
cmake -S . -B build -DLAIC_ENABLE_NATIVE=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/laic_bench
```

For Linux cache counters:

```bash
perf stat -e cycles,instructions,cache-references,cache-misses,LLC-load-misses ./build/laic_bench
```

`-march=native` is optional and should only be used when the resulting binary will run on a compatible CPU. LAICompute's AVX path is compile-time selected; runtime feature reporting prevents the project from confusing AVX with AVX2.
