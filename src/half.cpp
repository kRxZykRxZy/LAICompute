#include "laic/half.hpp"
#include <cstring>

namespace laic {
namespace {

uint16_t float_to_half(float f) noexcept {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int exp = static_cast<int>((x >> 23) & 0xffu) - 127;
    uint32_t mant = x & 0x7fffffu;
    if (exp > 15) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp < -14) {
        if (exp < -24) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int shift = -exp - 14;
        uint32_t hm = mant >> (shift + 13);
        const uint32_t round_bit = (mant >> (shift + 12)) & 1u;
        if (round_bit) ++hm;
        return static_cast<uint16_t>(sign | hm);
    }
    uint32_t he = static_cast<uint32_t>(exp + 15) << 10;
    uint32_t hm = mant >> 13;
    if (mant & 0x1000u) ++hm;
    if (hm == 0x400u) { hm = 0; he += 0x400u; }
    return static_cast<uint16_t>(sign | he | hm);
}

float half_to_float(uint16_t h) noexcept {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            out = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

} // namespace

Half::Half(float value) noexcept : bits_(float_to_half(value)) {}
float Half::to_float() const noexcept { return half_to_float(bits_); }

} // namespace laic
