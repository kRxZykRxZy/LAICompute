#include "laic/matmul.hpp"
#include "laic/cache.hpp"
#include "laic/pipeline.hpp"
#include <algorithm>
#include <thread>
#include <vector>
#if defined(__AVX__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

namespace laic { namespace {

MatmulConfig auto_cfg() {
    CachePlan p = CachePlan::detect();
    MatmulConfig c;
    // No cache-size constant is assumed. If the platform cannot report L1D,
    // use a small algorithmic tile rather than pretending to know its cache.
    size_t budget = p.l1_tile_bytes;
    size_t n = 8;
    if (budget) {
        while (n * n * 3 * sizeof(float) < budget && n < 128) n *= 2;
    }
    c.tile_m = std::max<size_t>(4, n / 2);
    c.tile_n = std::max<size_t>(4, n / 2);
    c.tile_k = std::max<size_t>(8, n);
    c.threads = std::max(1u, p.cpu.logical_cpus);
    return c;
}

MatmulConfig norm(MatmulConfig c) {
    auto a = auto_cfg();
    if (!c.tile_m) c.tile_m = a.tile_m;
    if (!c.tile_n) c.tile_n = a.tile_n;
    if (!c.tile_k) c.tile_k = a.tile_k;
    if (!c.threads) c.threads = a.threads;
    return c;
}

void scalar(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
            size_t i0, size_t i1, const MatmulConfig& c, const TilePipeline& pipeline) {
    (void)M;
    for (size_t i = i0; i < i1; i += c.tile_m) {
        for (size_t k0 = 0; k0 < K; k0 += c.tile_k) {
            size_t im = std::min(i + c.tile_m, i1);
            size_t km = std::min(k0 + c.tile_k, K);
            for (size_t j0 = 0; j0 < N; j0 += c.tile_n) {
                size_t jm = std::min(j0 + c.tile_n, N);
                const size_t next_j = j0 + c.tile_n;
                const size_t future_k = k0 + c.tile_k;
                const void* next = next_j < N ? static_cast<const void*>(B + k0 * N + next_j) : nullptr;
                const void* future = future_k < K ? static_cast<const void*>(B + future_k * N + j0) : nullptr;
                pipeline.advance(B + k0 * N + j0,
                                 (km - k0) * (jm - j0) * sizeof(float),
                                 next,
                                 next_j < N ? (jm - j0) * sizeof(float) : 0,
                                 future,
                                 future_k < K ? (std::min(c.tile_k, K - future_k) * (jm - j0) * sizeof(float)) : 0);
                for (size_t ii = i; ii < im; ++ii) {
                    float* d = C + ii * N;
                    for (size_t k = k0; k < km; ++k) {
                        float a = A[ii * K + k];
                        const float* b = B + k * N;
                        for (size_t j = j0; j < jm; ++j) d[j] += a * b[j];
                    }
                }
            }
        }
    }
}

#if defined(__AVX__) && (defined(__x86_64__) || defined(__i386__))
void avx(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
         size_t i0, size_t i1, const MatmulConfig& c, const TilePipeline& pipeline) {
    (void)M;
    for (size_t i = i0; i < i1; i += c.tile_m) {
        for (size_t k0 = 0; k0 < K; k0 += c.tile_k) {
            size_t im = std::min(i + c.tile_m, i1);
            size_t km = std::min(k0 + c.tile_k, K);
            for (size_t j0 = 0; j0 < N; j0 += c.tile_n) {
                size_t jm = std::min(j0 + c.tile_n, N);
                const size_t next_j = j0 + c.tile_n;
                const size_t future_k = k0 + c.tile_k;
                const void* next = next_j < N ? static_cast<const void*>(B + k0 * N + next_j) : nullptr;
                const void* future = future_k < K ? static_cast<const void*>(B + future_k * N + j0) : nullptr;
                pipeline.advance(B + k0 * N + j0,
                                 (km - k0) * (jm - j0) * sizeof(float),
                                 next,
                                 next_j < N ? (jm - j0) * sizeof(float) : 0,
                                 future,
                                 future_k < K ? (std::min(c.tile_k, K - future_k) * (jm - j0) * sizeof(float)) : 0);
                for (size_t ii = i; ii < im; ++ii) {
                    float* d = C + ii * N;
                    for (size_t k = k0; k < km; ++k) {
                        __m256 av = _mm256_set1_ps(A[ii * K + k]);
                        const float* b = B + k * N;
                        size_t j = j0;
                        for (; j + 8 <= jm; j += 8) {
                            __m256 cv = _mm256_loadu_ps(d + j);
                            _mm256_storeu_ps(d + j, _mm256_add_ps(cv, _mm256_mul_ps(av, _mm256_loadu_ps(b + j))));
                        }
                        for (; j < jm; ++j) d[j] += A[ii * K + k] * b[j];
                    }
                }
            }
        }
    }
}
#endif

} // namespace

void matmul_tiled(const float* A, const float* B, float* C, size_t M, size_t N, size_t K,
                  const MatmulConfig& cfg) {
    auto c = norm(cfg);
    std::fill(C, C + M * N, 0.0f);
    CachePlan plan = CachePlan::detect();
    TilePipeline pipeline(plan);
    unsigned t = std::max(1u, std::min<unsigned>(c.threads, static_cast<unsigned>(M)));
    std::vector<std::thread> workers;
    workers.reserve(t);
    for (unsigned x = 0; x < t; ++x) {
        size_t a = M * x / t;
        size_t b = M * (x + 1) / t;
        workers.emplace_back([=, &pipeline]() {
#if defined(__AVX__) && (defined(__x86_64__) || defined(__i386__))
            avx(A, B, C, M, N, K, a, b, c, pipeline);
#else
            scalar(A, B, C, M, N, K, a, b, c, pipeline);
#endif
        });
    }
    for (auto& worker : workers) worker.join();
}

void matmul_fp16(const Half* A, const Half* B, float* C, size_t M, size_t N, size_t K,
                 const MatmulConfig& cfg) {
    auto c = norm(cfg);
    std::fill(C, C + M * N, 0.0f);
    CachePlan plan = CachePlan::detect();
    TilePipeline pipeline(plan);
    unsigned t = std::max(1u, std::min<unsigned>(c.threads, static_cast<unsigned>(M)));
    std::vector<std::thread> workers;
    workers.reserve(t);
    for (unsigned x = 0; x < t; ++x) {
        size_t i0 = M * x / t, i1 = M * (x + 1) / t;
        workers.emplace_back([=, &pipeline]() {
            for (size_t i = i0; i < i1; i += c.tile_m) {
                for (size_t k0 = 0; k0 < K; k0 += c.tile_k) {
                    size_t im = std::min(i + c.tile_m, i1);
                    size_t km = std::min(k0 + c.tile_k, K);
                    for (size_t j0 = 0; j0 < N; j0 += c.tile_n) {
                        size_t jm = std::min(j0 + c.tile_n, N);
                        size_t next_j = j0 + c.tile_n;
                        size_t future_k = k0 + c.tile_k;
                        const void* next = next_j < N ? static_cast<const void*>(B + k0 * N + next_j) : nullptr;
                        const void* future = future_k < K ? static_cast<const void*>(B + future_k * N + j0) : nullptr;
                        pipeline.advance(B + k0 * N + j0,
                                         (km - k0) * (jm - j0) * sizeof(Half),
                                         next,
                                         next_j < N ? (jm - j0) * sizeof(Half) : 0,
                                         future,
                                         future_k < K ? (std::min(c.tile_k, K - future_k) * (jm - j0) * sizeof(Half)) : 0);
                        for (size_t ii = i; ii < im; ++ii) {
                            float* d = C + ii * N;
                            for (size_t kk = k0; kk < km; ++kk) {
                                float a = A[ii * K + kk].to_float();
                                const Half* b = B + kk * N;
                                for (size_t j = j0; j < jm; ++j) d[j] += a * b[j].to_float();
                            }
                        }
                    }
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
}

} // namespace laic
