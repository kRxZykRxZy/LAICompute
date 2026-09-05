#pragma once
#include <cstddef>
#include <vector>

namespace laic {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<size_t> shape);
    Tensor(std::vector<size_t> shape, std::vector<float> data);

    size_t size() const noexcept;
    size_t ndim() const noexcept;
    const std::vector<size_t>& shape() const noexcept;
    float* data() noexcept;
    const float* data() const noexcept;

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;
};

} // namespace laic
