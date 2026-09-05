#include "laic/cache.hpp"
#include "laic/prefetch.hpp"

namespace laic {

CacheScheduler::CacheScheduler(CachePlan plan) : plan_(plan) {}

void CacheScheduler::prepare(const void* current, size_t current_bytes,
                             const void* next, size_t next_bytes,
                             const void* future, size_t future_bytes) const noexcept {
    // Caches remain hardware-managed. We encourage locality by prefetching the
    // next and future working sets; current data is assumed to be hot already.
    (void)current;
    (void)current_bytes;
    const size_t next_n = next_bytes < plan_.l2_window_bytes ? next_bytes : plan_.l2_window_bytes;
    const size_t future_n = future_bytes < plan_.l3_window_bytes ? future_bytes : plan_.l3_window_bytes;
    prefetch_range(next, next_n, plan_.cache_line_bytes);
    prefetch_range(future, future_n, plan_.cache_line_bytes);
}

} // namespace laic
