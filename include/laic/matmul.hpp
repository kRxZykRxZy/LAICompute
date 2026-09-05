#pragma once
#include <cstddef>

namespace laic {

struct MatmulConfig {
    size_t tile_m = 32;
    size_t tile_n = 32;
    size_t tile_k = 64;
};

void matmul_tiled(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K,
                  const MatmulConfig& cfg = {});

} // namespace laic
