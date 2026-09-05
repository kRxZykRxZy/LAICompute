#pragma once
#include <cstddef>
#include <cstdint>

namespace laic::compute {
float dot_f16(const uint8_t * weights, const float * x, size_t n) noexcept;
float dot_f32(const uint8_t * weights, const float * x, size_t n) noexcept;
void axpy_f32(float * dst, const float * src, float scale, size_t n) noexcept;
}
