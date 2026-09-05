#pragma once
#include <cstdint>

namespace laic {

// IEEE-754 binary16 storage with explicit float conversion. Arithmetic is
// performed in float because older AVX CPUs do not have native FP16 math.
class Half {
public:
    Half() noexcept : bits_(0) {}
    explicit Half(float value) noexcept;
    float to_float() const noexcept;
    uint16_t bits() const noexcept { return bits_; }
    static Half from_bits(uint16_t bits) noexcept { Half h; h.bits_ = bits; return h; }
private:
    uint16_t bits_;
};

} // namespace laic
