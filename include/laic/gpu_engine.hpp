#pragma once
#include "laic/gguf.hpp"
#include "laic/videocore.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace laic::gpu {

struct Stats {
    uint64_t matvec_calls=0;
    uint64_t matvec_gpu_calls=0;
    uint64_t matvec_fallbacks=0;
    uint64_t gpu_ns=0;
    uint64_t gpu_flops=0;
    uint64_t bytes_read=0;
    uint64_t bytes_written=0;
    std::size_t max_work_group_size=0;
    std::size_t preferred_work_group_multiple=0;
    std::size_t selected_work_group_size=0;
    std::size_t compute_units=0;
    std::size_t clock_mhz=0;
};

class Engine {
public:
    Engine();
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    bool available() const noexcept;
    const std::string& detail() const noexcept;
    videocore::Generation generation() const noexcept;
    const Stats& stats() const noexcept;
    void reset_stats() noexcept;

    bool prepare(const GgufModel& model);
    bool matvec(const GgufTensor& w, const float* x, float* y,
                std::size_t out, std::size_t in);
    bool matmul(const float* a, const float* b, float* c,
                std::size_t m, std::size_t n, std::size_t k);
    bool partitioned_matvec(const GgufTensor& w, const float* x, float* y,
                            std::size_t out, std::size_t in,
                            std::size_t gpu_rows);
    void clear() noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace laic::gpu
