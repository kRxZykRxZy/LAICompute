#pragma once

#include <cstddef>
#include <cstdint>

namespace laic::qpu {

struct KernelResult {
    bool executed = false;
    std::size_t rows = 0;
    std::size_t cols = 0;
    const char* backend = "none";
};

// Phase 3 kernel ABI. The implementation deliberately keeps the host API
// independent of the QPU transport so Phase 4 can add OpenCL/VC4CL without
// changing inference code.
class MatvecKernel {
public:
    static KernelResult f32(const float* weights,
                            const float* input,
                            float* output,
                            std::size_t rows,
                            std::size_t cols) noexcept;

    static KernelResult f16(const std::uint16_t* weights,
                            const float* input,
                            float* output,
                            std::size_t rows,
                            std::size_t cols) noexcept;
};

} // namespace laic::qpu
