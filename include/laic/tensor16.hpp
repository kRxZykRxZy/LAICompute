#pragma once
#include "laic/half.hpp"
#include <cstddef>
#include <vector>

namespace laic {

class Tensor16 {
public:
    Tensor16() = default;
    explicit Tensor16(std::vector<size_t> shape);
    Tensor16(std::vector<size_t> shape, std::vector<Half> data);
    size_t size() const noexcept;
    size_t ndim() const noexcept;
    size_t bytes() const noexcept { return data_.size() * sizeof(Half); }
    const std::vector<size_t>& shape() const noexcept { return shape_; }
    Half* data() noexcept { return data_.data(); }
    const Half* data() const noexcept { return data_.data(); }
    Half& operator[](size_t i) noexcept { return data_[i]; }
    const Half& operator[](size_t i) const noexcept { return data_[i]; }
private:
    std::vector<size_t> shape_;
    std::vector<Half> data_;
};

} // namespace laic
