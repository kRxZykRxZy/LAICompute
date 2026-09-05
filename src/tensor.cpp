#include "laic/tensor.hpp"
#include <numeric>
#include <stdexcept>

namespace laic {

Tensor::Tensor(std::vector<size_t> shape) : shape_(std::move(shape)) {
    size_t n = 1;
    for (size_t d : shape_) n *= d;
    data_.resize(n);
}

Tensor::Tensor(std::vector<size_t> shape, std::vector<float> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
    size_t n = 1;
    for (size_t d : shape_) n *= d;
    if (n != data_.size()) throw std::invalid_argument("Tensor data size does not match shape");
}

size_t Tensor::size() const noexcept { return data_.size(); }
size_t Tensor::ndim() const noexcept { return shape_.size(); }
const std::vector<size_t>& Tensor::shape() const noexcept { return shape_; }
float* Tensor::data() noexcept { return data_.data(); }
const float* Tensor::data() const noexcept { return data_.data(); }

} // namespace laic
