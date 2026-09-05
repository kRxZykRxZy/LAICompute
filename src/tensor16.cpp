#include "laic/tensor16.hpp"
#include <stdexcept>

namespace laic {

static size_t element_count(const std::vector<size_t>& shape) {
    size_t n = 1;
    for (size_t d : shape) n *= d;
    return n;
}

Tensor16::Tensor16(std::vector<size_t> shape) : shape_(std::move(shape)), data_(element_count(shape_)) {}
Tensor16::Tensor16(std::vector<size_t> shape, std::vector<Half> data)
    : shape_(std::move(shape)), data_(std::move(data)) {
    if (element_count(shape_) != data_.size()) throw std::invalid_argument("Tensor16 data size does not match shape");
}
size_t Tensor16::size() const noexcept { return data_.size(); }
size_t Tensor16::ndim() const noexcept { return shape_.size(); }

} // namespace laic
