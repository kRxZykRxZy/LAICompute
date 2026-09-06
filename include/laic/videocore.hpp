#pragma once
#include <cstddef>
#include <string>

namespace laic::videocore {

enum class Backend { CPU, GPU, Both };
enum class Generation { Unknown, IV, VI, VII };

struct DeviceInfo {
    bool present = false;
    Generation generation = Generation::Unknown;
    std::string name;
    std::string runtime;
    unsigned qpus = 0;
    unsigned clock_mhz = 0;
    double theoretical_gflops = 0.0;
    bool compute_available = false;
};

// Parses a backend name and records it as the process-wide requested backend.
// The server uses this path, so inference sees UI backend changes without a
// second server/runtime coupling layer.
Backend backend_from_string(const std::string& value) noexcept;
void set_requested_backend(Backend backend) noexcept;
Backend requested_backend() noexcept;
const char* backend_name(Backend backend) noexcept;
DeviceInfo detect();
std::string generation_name(Generation generation) noexcept;
double theoretical_peak_gflops(const DeviceInfo& device) noexcept;

} // namespace laic::videocore
