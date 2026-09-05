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

    bool matvec_f32(const float* weights, const float* input, float* output,
                    std::size_t rows, std::size_t cols, std::string* error = nullptr);
    bool matvec_f16(const std::uint16_t* weights, const float* input, float* output,
                    std::size_t rows, std::size_t cols, std::string* error = nullptr);

private:
    struct Api;
    bool run_matvec(void* kernel, const void* weights, std::size_t weights_bytes,
                    const float* input, std::size_t input_bytes, float* output,
                    std::size_t output_bytes, std::size_t rows, std::size_t cols,
                    std::string* error);

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
    std::size_t weights_capacity_ = 0;
    std::size_t input_capacity_ = 0;
    std::size_t output_capacity_ = 0;
    bool ready_ = false;
    std::string device_name_;
};

} // namespace laic::qpu
