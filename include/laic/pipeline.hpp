#pragma once
#include "laic/cache.hpp"
#include <cstddef>
#include <utility>
namespace laic {
class TilePipeline {
public:
    explicit TilePipeline(CachePlan plan = CachePlan::detect()) : plan_(std::move(plan)) {}
    void prime(const void* next,size_t next_bytes,const void* future,size_t future_bytes) const noexcept;
    void advance(const void* current,size_t current_bytes,const void* next,size_t next_bytes,const void* future,size_t future_bytes) const noexcept;
    const CachePlan& plan() const noexcept{return plan_;}
private:
    CachePlan plan_;
};
} // namespace laic
