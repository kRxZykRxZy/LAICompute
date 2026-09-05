#pragma once
#include <cstddef>

namespace laic {

struct CachePlan {
    size_t l1_tile_bytes = 32 * 1024;
    size_t l2_window_bytes = 256 * 1024;
    size_t l3_window_bytes = 3 * 1024 * 1024;
    size_t cache_line_bytes = 64;
};

class CacheScheduler {
public:
    explicit CacheScheduler(CachePlan plan = {});
    void prepare(const void* current, size_t current_bytes,
                 const void* next, size_t next_bytes,
                 const void* future, size_t future_bytes) const noexcept;
    const CachePlan& plan() const noexcept { return plan_; }
private:
    CachePlan plan_;
};

} // namespace laic
