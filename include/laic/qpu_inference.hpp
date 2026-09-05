#pragma once
#include "laic/inference.hpp"
#include "laic/qpu_opencl.hpp"
#include <string>

namespace laic::qpu {

class QpuLlamaRuntime {
public:
    using TokenCallback = LlamaRuntime::TokenCallback;
    QpuLlamaRuntime();
    ~QpuLlamaRuntime();
    QpuLlamaRuntime(const QpuLlamaRuntime&) = delete;
    QpuLlamaRuntime& operator=(const QpuLlamaRuntime&) = delete;

    void load(const std::string& path);
    std::vector<uint32_t> generate_ids(const std::string& prompt, const GenerationConfig& cfg = {});
    std::vector<uint32_t> generate_ids(const std::string& prompt, const GenerationConfig& cfg, const TokenCallback& cb);
    std::string generate(const std::string& prompt, const GenerationConfig& cfg = {});
    void request_stop() noexcept { stop_.store(true, std::memory_order_relaxed); }
    bool stop_requested() const noexcept { return stop_.load(std::memory_order_relaxed); }
    bool gpu_ready() const noexcept { return gpu_.available(); }
    const std::string& device_name() const noexcept { return gpu_.device_name(); }
    const GgufModel& model() const noexcept { return model_; }
    const Gpt2Tokenizer& tokenizer() const noexcept { return tokenizer_; }
    size_t qpu_matvecs() const noexcept { return qpu_matvecs_; }
    size_t cpu_matvecs() const noexcept { return cpu_matvecs_; }

private:
    OpenCLBackend gpu_;
    GgufModel model_;
    Gpt2Tokenizer tokenizer_;
    std::vector<std::vector<float>> kcache_, vcache_;
    CachePlan cache_plan_;
    std::atomic<bool> stop_{false};
    size_t hidden_=0,layers_=0,heads_=0,kv_heads_=0,head_dim_=0,ffn_=0,ctx_=0,rope_dim_=0,pos_=0;
    float eps_=1e-5f, theta_=10000.f;
    size_t qpu_matvecs_=0, cpu_matvecs_=0;

    std::vector<float> embedding(uint32_t id) const;
    std::vector<float> matvec(const GgufTensor& w, const std::vector<float>& x, size_t out, size_t in);
    void norm(std::vector<float>& x, const GgufTensor& w) const;
    std::vector<float> step(uint32_t token);
    uint32_t sample(const std::vector<float>& logits, const GenerationConfig& cfg, const std::vector<uint32_t>& history) const;
};

} // namespace laic::qpu
