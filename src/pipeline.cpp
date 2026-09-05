#include "laic/pipeline.hpp"
#include "laic/prefetch.hpp"
#include <algorithm>

namespace laic {
void TilePipeline::prime(const void* next,size_t next_bytes,const void* future,size_t future_bytes) const noexcept {
    if(next) prefetch_range(next,plan_.l2_window_bytes?std::min(next_bytes,plan_.l2_window_bytes):next_bytes,plan_.cache_line_bytes);
    if(future) prefetch_range(future,plan_.l3_window_bytes?std::min(future_bytes,plan_.l3_window_bytes):future_bytes,plan_.cache_line_bytes);
}
void TilePipeline::advance(const void* current,size_t current_bytes,const void* next,size_t next_bytes,const void* future,size_t future_bytes) const noexcept {
    (void)current;(void)current_bytes;prime(next,next_bytes,future,future_bytes);
}
} // namespace laic
