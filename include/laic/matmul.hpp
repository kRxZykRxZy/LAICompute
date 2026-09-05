#pragma once
#include "laic/half.hpp"
#include <cstddef>

namespace laic {

struct MatmulConfig {
    size_t tile_m = 0; // 0 = auto-tune from detected L1D working set
    size_t tile_n = 0;
    size_t tile_k = 0;
    unsigned threads = 0; // 0 = physical-core count
};

void matmul_tiled(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K,
                  const MatmulConfig& cfg = {});

void matmul_fp16(const Half* A, const Half* B, float* C,
                 size_t M, size_t N, size_t K,
                 const MatmulConfig& cfg = {});

} // namespace laic
