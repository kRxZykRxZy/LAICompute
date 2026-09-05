#pragma once
#include "laic/inference.hpp"
#include "laic/qpu_inference.hpp"
#include <memory>
#include <string>

namespace laic {
enum class ExecutionMode { CPU, GPU, BOTH, AUTO };
const char* execution_mode_name(ExecutionMode mode) noexcept;
ExecutionMode execution_mode_from_string(const std::string& s) noexcept;

class ExecutionRuntime {
public:
    ExecutionRuntime();
    ~ExecutionRuntime();
    ExecutionRuntime(const ExecutionRuntime&) = delete;
    ExecutionRuntime& operator=(const ExecutionRuntime&) = delete;
    void load(const std::string& path);
    std::vector<uint32_t> generate_ids(const std::string& prompt,const GenerationConfig& cfg,const LlamaRuntime::TokenCallback& cb);
    void request_stop() noexcept;
    bool stop_requested() const noexcept;
    ExecutionMode mode() const noexcept { return mode_; }
    void set_mode(ExecutionMode m) noexcept { mode_=m; }
    bool gpu_available() const noexcept;
    std::string gpu_device() const;
    size_t qpu_matvecs() const noexcept;
    size_t cpu_matvecs() const noexcept;
    const Gpt2Tokenizer& tokenizer() const;
private:
    ExecutionMode mode_=ExecutionMode::AUTO;
    std::unique_ptr<LlamaRuntime> cpu_;
    std::unique_ptr<qpu::QpuLlamaRuntime> gpu_;
};
} // namespace laic
