#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace laic::qpu {

class OpenCLBackend {
public:
    OpenCLBackend();
    ~OpenCLBackend();
    OpenCLBackend(const OpenCLBackend&) = delete;
    OpenCLBackend& operator=(const OpenCLBackend&) = delete;

    bool initialize(std::string* error = nullptr);
    void shutdown() noexcept;
    bool available() const noexcept { return ready_; }
    const std::string& device_name() const noexcept { return device_name_; }
    bool matvec_f32(const float*, const float*, float*, std::size_t, std::size_t, std::string* = nullptr);
    bool matvec_f16(const std::uint16_t*, const float*, float*, std::size_t, std::size_t, std::string* = nullptr);

    // Kept public for the tiny dynamic OpenCL ABI implementation. No OpenCL
    // headers are required on CPU-only systems.
    struct Api;
    Api* api_ = nullptr;
    void* library_ = nullptr;
    void* context_ = nullptr;
    void* queue_ = nullptr;
    void* program_ = nullptr;
    void* kernel_f32_ = nullptr;
    void* kernel_f16_ = nullptr;
    void* weights_buf_ = nullptr;
    void* input_buf_ = nullptr;
    void* output_buf_ = nullptr;
    std::size_t weights_capacity_ = 0, input_capacity_ = 0, output_capacity_ = 0;
    bool ready_ = false;
    std::string device_name_;
};

} // namespace laic::qpu
