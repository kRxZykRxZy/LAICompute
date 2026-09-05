#include "laic/prefetch.hpp"

namespace laic {

void prefetch_range(const void* ptr, size_t bytes, size_t line_size) noexcept {
    const auto* p = static_cast<const unsigned char*>(ptr);
    if (line_size == 0) return;
    for (size_t i = 0; i < bytes; i += line_size) prefetch_read(p + i);
}

} // namespace laic
