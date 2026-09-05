#pragma once
#include "laic/gguf.hpp"
#include "laic/tokenizer.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace laic {
struct GenerationConfig { size_t max_tokens=32; float temperature=0.0f; size_t top_k=0; };
class LlamaRuntime {
public:
    void load(const std::string& path);
    std::vector<uint32_t> generate_ids(const std::string& prompt,const GenerationConfig& cfg={});
    std::string generate(const std::string& prompt,const GenerationConfig& cfg={});
    const GgufModel& model() const noexcept{return model_;}
    const Gpt2Tokenizer& tokenizer() const noexcept{return tokenizer_;}
private:
    struct LayerCache { std::vector<float> k,v; };
    GgufModel model_; Gpt2Tokenizer tokenizer_; std::vector<LayerCache> cache_;
    size_t hidden_=0,layers_=0,heads_=0,kv_heads_=0,head_dim_=0,ffn_=0,ctx_=0,rope_dim_=0,pos_=0; float eps_=1e-5f,theta_=10000.f;
    std::vector<float> embedding(uint32_t id) const;
    std::vector<float> matvec(const GgufTensor& w,const std::vector<float>& x,size_t out,size_t in) const;
    void norm(std::vector<float>& x,const GgufTensor& w) const;
    std::vector<float> step(uint32_t token);
    uint32_t sample(const std::vector<float>& logits,const GenerationConfig& cfg) const;
};
} // namespace laic
