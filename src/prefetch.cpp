#include "laic/prefetch.hpp"

namespace laic {

void prefetch_range(const void* ptr, size_t bytes, size_t line_size) noexcept {
    if (!ptr || bytes == 0 || line_size == 0) return;
    const auto* p = static_cast<const unsigned char*>(ptr);
    for (size_t i = 0; i < bytes; i += line_size) prefetch_read(p + i);
}

} // namespace laic
