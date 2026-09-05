#pragma once
#include "laic/cache.hpp"
#include "laic/matmul.hpp"
#include "laic/pipeline.hpp"
namespace laic {
class Engine {
public:
    explicit Engine(CachePlan plan = CachePlan::detect());
    void matmul(const float* A,const float* B,float* C,size_t M,size_t N,size_t K,const MatmulConfig& cfg={});
    void matmul_fp16(const Half* A,const Half* B,float* C,size_t M,size_t N,size_t K,const MatmulConfig& cfg={});
    const CachePlan& cache_plan() const noexcept { return cache_.plan(); }
private:
    CacheScheduler cache_;
    TilePipeline pipeline_;
};
} // namespace laic
