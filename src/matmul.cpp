#include "laic/matmul.hpp"
#include <algorithm>
#include <cstring>

namespace laic {

void matmul_tiled(const float* A, const float* B, float* C,
                  size_t M, size_t N, size_t K, const MatmulConfig& cfg) {
    std::fill(C, C + M * N, 0.0f);
    for (size_t i0 = 0; i0 < M; i0 += cfg.tile_m)
        for (size_t k0 = 0; k0 < K; k0 += cfg.tile_k)
            for (size_t j0 = 0; j0 < N; j0 += cfg.tile_n) {
                const size_t imax = std::min(M, i0 + cfg.tile_m);
                const size_t kmax = std::min(K, k0 + cfg.tile_k);
                const size_t jmax = std::min(N, j0 + cfg.tile_n);
                for (size_t i = i0; i < imax; ++i) {
                    for (size_t k = k0; k < kmax; ++k) {
                        const float a = A[i * K + k];
                        const float* brow = B + k * N;
                        float* crow = C + i * N;
                        for (size_t j = j0; j < jmax; ++j)
                            crow[j] += a * brow[j];
                    }
                }
            }
}

} // namespace laic
