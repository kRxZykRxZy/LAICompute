#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace laic::qpu {

struct QpuInfo {
    bool available = false;
    bool videocore_iv = false;
    bool vc4cl = false;
    bool legacy_qpu = false;
    unsigned qpus = 0;
    unsigned simd_width = 16;
    std::string device_name;
    std::string reason;
};

class Engine {
public:
    Engine();
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    static QpuInfo detect() noexcept;
    const QpuInfo& info() const noexcept { return info_; }
    bool available() const noexcept { return info_.available; }

    // Phase 1/2 intentionally expose detection and a backend-neutral
    // lifecycle only. Compute kernels are added in Phase 3.
    bool initialize();
    void shutdown() noexcept;

private:
    QpuInfo info_;
    bool initialized_ = false;
};

} // namespace laic::qpu
