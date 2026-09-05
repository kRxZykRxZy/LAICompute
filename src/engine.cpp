#include "laic/engine.hpp"

namespace laic {

Engine::Engine(CachePlan plan) : cache_(plan) {}

void Engine::matmul(const float* A, const float* B, float* C,
                    size_t M, size_t N, size_t K, const MatmulConfig& cfg) {
    matmul_tiled(A, B, C, M, N, K, cfg);
}

} // namespace laic
