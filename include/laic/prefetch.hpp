#pragma once
#include <cstddef>

namespace laic {

inline void prefetch_read(const void* ptr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    __builtin_prefetch(ptr, 0, 3);
#else
    (void)ptr;
#endif
}

void prefetch_range(const void* ptr, size_t bytes, size_t line_size = 64) noexcept;

} // namespace laic
