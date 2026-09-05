#include "laic/qpu_kernel.hpp"

#include <cstring>

namespace laic::qpu {
namespace {

static float half_to_float(std::uint16_t h) noexcept {
    const std::uint32_t s = (h & 0x8000u) << 16;
    const std::uint32_t e = (h >> 10) & 0x1fu;
    const std::uint32_t f = h & 0x03ffu;
    std::uint32_t bits;
    if (e == 0) {
        if (f == 0) bits = s;
        else {
            std::uint32_t mant = f;
            int exp = -14;
            while ((mant & 0x400u) == 0) { mant <<= 1; --exp; }
            mant &= 0x3ffu;
            bits = s | (static_cast<std::uint32_t>(exp + 127) << 23) | (mant << 13);
        }
    } else if (e == 31) {
        bits = s | 0x7f800000u | (f << 13);
    } else {
        bits = s | ((e + 112u) << 23) | (f << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

} // namespace

KernelResult MatvecKernel::f32(const float* weights, const float* input,
                               float* output, std::size_t rows,
                               std::size_t cols) noexcept {
    if (!weights || !input || !output || !rows || !cols)
        return {};

    // Portable reference kernel for the Phase 3 ABI. Phase 4 supplies the
    // actual VC4CL transport while retaining this exact row-major contract.
    for (std::size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        const float* w = weights + r * cols;
        for (std::size_t c = 0; c < cols; ++c)
            sum += w[c] * input[c];
        output[r] = sum;
    }
    return {false, rows, cols, "reference"};
}

KernelResult MatvecKernel::f16(const std::uint16_t* weights, const float* input,
                               float* output, std::size_t rows,
                               std::size_t cols) noexcept {
    if (!weights || !input || !output || !rows || !cols)
        return {};

    for (std::size_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        const std::uint16_t* w = weights + r * cols;
        for (std::size_t c = 0; c < cols; ++c)
            sum += half_to_float(w[c]) * input[c];
        output[r] = sum;
    }
    return {false, rows, cols, "reference-f16"};
}

} // namespace laic::qpu
