#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <variant>

namespace laic {

enum class GgmlType : uint32_t { F32=0, F16=1, Q4_0=2, Q4_1=3, Q5_0=6, Q5_1=7, Q8_0=8 };

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> shape;
    GgmlType type = GgmlType::F32;
    uint64_t offset = 0;
    const uint8_t* data = nullptr;

    size_t elements() const noexcept;
    size_t bytes() const noexcept;
    float value(size_t index) const;
};

using GgufValue = std::variant<uint64_t, int64_t, double, bool, std::string,
                               std::vector<uint64_t>, std::vector<std::string>>;

class GgufModel {
public:
    void load(const std::string& path);
    const GgufTensor& tensor(const std::string& name) const;
    bool has_tensor(const std::string& name) const noexcept;
    const GgufValue* metadata(const std::string& key) const noexcept;
    uint64_t u64(const std::string& key, uint64_t fallback=0) const;
    double f64(const std::string& key, double fallback=0.0) const;
    std::string str(const std::string& key, const std::string& fallback={}) const;
    const std::vector<std::string>& strings(const std::string& key) const;
    const std::unordered_map<std::string,GgufTensor>& tensors() const noexcept { return tensors_; }
    const std::string& path() const noexcept { return path_; }
private:
    std::string path_;
    std::vector<uint8_t> file_;
    std::unordered_map<std::string,GgufValue> meta_;
    std::unordered_map<std::string,GgufTensor> tensors_;
    std::vector<std::string> empty_strings_;
};

} // namespace laic
