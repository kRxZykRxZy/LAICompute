#pragma once
#include "laic/gguf.hpp"
#include "laic/videocore.hpp"
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace laic::gpu {

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool available() const noexcept;
    const std::string& detail() const noexcept;
    videocore::Generation generation() const noexcept;

    // Lazy persistent model buffers. The host GGUF mapping remains the owner of
    // the bytes; GPU buffers are created once per tensor and reused for every token.
    bool prepare(const GgufModel& model);

    bool matvec(const GgufTensor& w, const float* x, float* y,
                std::size_t out, std::size_t in);
    bool matmul(const float* a, const float* b, float* c,
                std::size_t m, std::size_t n, std::size_t k);

    // Split output rows between GPU and CPU. GPU and CPU work overlap when both
    // are enabled; the GPU owns the first gpu_rows rows.
    bool partitioned_matvec(const GgufTensor& w, const float* x, float* y,
                            std::size_t out, std::size_t in,
                            std::size_t gpu_rows);

    void clear() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace laic::gpu
