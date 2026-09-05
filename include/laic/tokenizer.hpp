#pragma once
#include "laic/gguf.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace laic {
class Gpt2Tokenizer {
public:
    void load(const GgufModel& model);
    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(uint32_t id) const;
    uint32_t bos_id() const noexcept { return bos_; }
    uint32_t eos_id() const noexcept { return eos_; }
    bool add_bos() const noexcept { return add_bos_; }
    size_t vocab_size() const noexcept { return tokens_.size(); }
private:
    std::vector<std::string> tokens_;
    std::unordered_map<std::string,uint32_t> ids_;
    std::unordered_map<std::string,uint32_t> ranks_;
    uint32_t bos_=0,eos_=0,unk_=0;
    bool add_bos_=false;
    static std::string byte_encode(const std::string& s);
    static std::string byte_decode(const std::string& s);
};
} // namespace laic
