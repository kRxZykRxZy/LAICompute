#pragma once
#include "laic/videocore.hpp"
#include <string>
#include <vector>

namespace laic::videocore {

struct RuntimeInfo {
    bool available = false;
    Generation generation = Generation::Unknown;
    std::string api;
    std::string device;
    std::string detail;
    std::vector<std::string> all_devices;
    std::string all_device_versions;
};

// Probes the native compute stack without linking LAICompute to optional GPU SDKs.
// IV: VC4CL/OpenCL. VI/VII: Mesa V3DV/Vulkan.
RuntimeInfo probe_runtime(const DeviceInfo& device) noexcept;

// Returns true when the selected VideoCore generation has a usable native runtime.
bool runtime_available(const DeviceInfo& device) noexcept;

} // namespace laic::videocore
