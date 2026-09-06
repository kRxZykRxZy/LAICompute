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

Backend backend_from_string(const std::string& value) noexcept;
const char* backend_name(Backend backend) noexcept;
DeviceInfo detect();
std::string generation_name(Generation generation) noexcept;

// Returns a conservative theoretical FP32 peak based on the detected QPU topology.
// This is a hardware peak, not a claim about end-to-end transformer inference speed.
double theoretical_peak_gflops(const DeviceInfo& device) noexcept;

} // namespace laic::videocore
