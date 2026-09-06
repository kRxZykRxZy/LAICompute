#pragma once
#include "laic/gguf.hpp"
#include "laic/tokenizer.hpp"
#include "laic/cache.hpp"
#include "laic/videocore.hpp"
#include "laic/gpu_engine.hpp"
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace laic {
struct GenerationConfig { size_t max_tokens=32; float temperature=0.0f; size_t top_k=0; float top_p=1.0f; float repeat_penalty=1.0f; uint64_t seed=0; };
class LlamaRuntime {
public:
    using TokenCallback=std::function<bool(uint32_t)>;
    void load(const std::string& path);
    void set_backend(videocore::Backend backend) noexcept { backend_=backend; videocore::set_requested_backend(backend); }
    videocore::Backend backend() const noexcept { return backend_; }
    bool gpu_available() const noexcept { return gpu_ && gpu_->available(); }
    const gpu::Stats* gpu_stats() const noexcept { return gpu_ ? &gpu_->stats() : nullptr; }
    std::vector<uint32_t> generate_ids(const std::string& prompt,const GenerationConfig& cfg={});
    std::vector<uint32_t> generate_ids(const std::string& prompt,const GenerationConfig& cfg,const TokenCallback& callback);
    std::string generate(const std::string& prompt,const GenerationConfig& cfg={});
    void request_stop() noexcept { stop_requested_.store(true,std::memory_order_relaxed); }
    bool stop_requested()const noexcept{return stop_requested_.load(std::memory_order_relaxed);}
    const GgufModel& model()const noexcept{return model_;}
    const Gpt2Tokenizer& tokenizer()const noexcept{return tokenizer_;}
private:
    struct LayerCache{std::vector<float> k,v;};
    GgufModel model_; Gpt2Tokenizer tokenizer_; std::vector<LayerCache> cache_; CachePlan cache_plan_;
    std::unique_ptr<gpu::Engine> gpu_;
    std::atomic<bool> stop_requested_{false};
    videocore::Backend backend_=videocore::requested_backend();
    size_t hidden_=0,layers_=0,heads_=0,kv_heads_=0,head_dim_=0,ffn_=0,ctx_=0,rope_dim_=0,pos_=0;
    float eps_=1e-5f,theta_=10000.f;
    std::vector<float> embedding(uint32_t id)const;
    std::vector<float> matvec(const GgufTensor&w,const std::vector<float>&x,size_t out,size_t in)const;
    void norm(std::vector<float>&x,const GgufTensor&w)const;
    std::vector<float> step(uint32_t token);
    uint32_t sample(const std::vector<float>&logits,const GenerationConfig&cfg,const std::vector<uint32_t>&history)const;
};
}
