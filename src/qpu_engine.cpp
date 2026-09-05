#include "laic/qpu_engine.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

#if defined(__linux__)
#include <sys/utsname.h>
#endif

namespace laic::qpu {
namespace {

static bool file_contains(const char* path, const char* needle) noexcept {
    std::ifstream f(path);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str().find(needle) != std::string::npos;
}

static bool has_path(const char* path) noexcept {
    std::ifstream f(path);
    return static_cast<bool>(f);
}

static unsigned detect_qpus_from_vc4() noexcept {
    // VideoCore IV exposes 12 QPUs on Raspberry Pi 2/3.
    // Keep this as a hardware-family value; Phase 3 will query the
    // actual execution backend before launching kernels.
    return 12;
}

} // namespace

QpuInfo Engine::detect() noexcept {
    QpuInfo out;
#if defined(__linux__)
    // BCM2836 is the Raspberry Pi 2 SoC. Device-tree compatible strings
    // are preferable to uname because ARM machines can report generic names.
    const bool bcm2836 = file_contains("/proc/device-tree/compatible", "brcm,bcm2836");
    const bool bcm2837 = file_contains("/proc/device-tree/compatible", "brcm,bcm2837");
    const bool vc4 = has_path("/sys/firmware/devicetree/base/soc/v3d@7ec00000") ||
                     has_path("/sys/firmware/devicetree/base/soc/gpu@7ec000000");

    if (bcm2836 || bcm2837 || vc4) {
        out.videocore_iv = true;
        out.qpus = detect_qpus_from_vc4();
        out.simd_width = 16;
        out.device_name = "VideoCore IV";

        // Phase 2 backend discovery. VC4CL can expose OpenCL through a
        // platform/device pair; do not claim it is usable merely because
        // the Raspberry Pi GPU exists.
        if (std::getenv("LAIC_QPU_DISABLE") == nullptr) {
            const bool loader = has_path("/usr/lib/libOpenCL.so") ||
                                has_path("/usr/lib/arm-linux-gnueabihf/libOpenCL.so") ||
                                has_path("/usr/lib/aarch64-linux-gnu/libOpenCL.so") ||
                                has_path("/usr/local/lib/libOpenCL.so");
            out.vc4cl = loader && (std::getenv("LAIC_QPU_VC4CL") != nullptr);
        }

        out.legacy_qpu = has_path("/dev/vcio") &&
                         (std::getenv("LAIC_QPU_LEGACY") != nullptr);
        out.available = out.vc4cl || out.legacy_qpu;
        if (out.available)
            out.reason = out.vc4cl ? "VC4CL OpenCL backend available" : "legacy QPU mailbox backend available";
        else
            out.reason = "VideoCore IV detected; no enabled QPU execution backend found";
    } else {
        out.reason = "VideoCore IV not detected";
    }
#else
    out.reason = "QPU detection currently supports Linux Raspberry Pi systems";
#endif
    return out;
}

Engine::Engine() : info_(detect()) {}
Engine::~Engine() { shutdown(); }

bool Engine::initialize() {
    info_ = detect();
    initialized_ = info_.available;
    return initialized_;
}

void Engine::shutdown() noexcept {
    initialized_ = false;
}

} // namespace laic::qpu
