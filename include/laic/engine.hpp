#pragma once
#include "laic/cache.hpp"
#include "laic/matmul.hpp"

namespace laic {

class Engine {
public:
    explicit Engine(CachePlan plan = {});
    void matmul(const float* A, const float* B, float* C,
                size_t M, size_t N, size_t K,
                const MatmulConfig& cfg = {});
private:
    CacheScheduler cache_;
};

} // namespace laic
