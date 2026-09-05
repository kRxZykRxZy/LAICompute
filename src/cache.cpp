#include "laic/cache.hpp"
#include "laic/prefetch.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace laic {
namespace {

size_t parse_size(const std::string& s) {
    if (s.empty()) return 0;
    size_t value = 0;
    try { value = std::stoull(s); } catch (...) { return 0; }
    if (s.back() == 'K' || s.back() == 'k') return value * 1024;
    if (s.back() == 'M' || s.back() == 'm') return value * 1024 * 1024;
    return value;
}

#if defined(__x86_64__) || defined(__i386__)
void detect_x86(CpuCacheInfo& out) {
    unsigned eax, ebx, ecx, edx;
    if (!__get_cpuid_max(0, nullptr)) return;

    char vendor[13] = {};
    __cpuid(0, eax, ebx, ecx, edx);
    *reinterpret_cast<unsigned*>(vendor + 0) = ebx;
    *reinterpret_cast<unsigned*>(vendor + 4) = edx;
    *reinterpret_cast<unsigned*>(vendor + 8) = ecx;
    out.cpu_name = vendor;

    unsigned max_leaf = __get_cpuid_max(0, nullptr);
    if (max_leaf < 4) return;

    size_t l1d = 0, l1i = 0, l2 = 0, l3 = 0;
    for (unsigned subleaf = 0; subleaf < 32; ++subleaf) {
        __cpuid_count(4, subleaf, eax, ebx, ecx, edx);
        unsigned type = eax & 0x1f;
        if (type == 0) break;
        unsigned level = (eax >> 5) & 0x7;
        size_t line = (ebx & 0xfff) + 1ULL;
        size_t partitions = ((ebx >> 12) & 0x3ff) + 1ULL;
        size_t ways = ((ebx >> 22) & 0x3ff) + 1ULL;
        size_t sets = static_cast<size_t>(ecx) + 1ULL;
        size_t bytes = line * partitions * ways * sets;
        out.line_bytes = std::max(out.line_bytes, line);

        if (level == 1) {
            if (type == 1) l1d = std::max(l1d, bytes);
            if (type == 2) l1i = std::max(l1i, bytes);
        } else if (level == 2) {
            l2 = std::max(l2, bytes);
        } else if (level == 3) {
            l3 = std::max(l3, bytes);
        }
    }
    out.l1_data_bytes = l1d;
    out.l1_instruction_bytes = l1i;
    out.l2_bytes = l2;
    out.l3_bytes = l3;
}
#endif

#if defined(__linux__)
void detect_linux(CpuCacheInfo& out) {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.rfind("model name", 0) == 0 || line.rfind("Hardware", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) out.cpu_name = line.substr(pos + 2);
            break;
        }
    }

    out.logical_cpus = std::max(1u, std::thread::hardware_concurrency());

    // Linux exposes the actual cache topology per logical CPU. Read CPU0;
    // heterogeneous systems are handled by the x86 CPUID path where possible.
    for (int index = 0; index < 16; ++index) {
        std::string base = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(index) + "/";
        std::ifstream level_file(base + "level");
        std::ifstream type_file(base + "type");
        std::ifstream size_file(base + "size");
        std::string level_s, type_s, size_s;
        if (!std::getline(level_file, level_s) || !std::getline(type_file, type_s) || !std::getline(size_file, size_s)) continue;
        int level = 0;
        try { level = std::stoi(level_s); } catch (...) { continue; }
        size_t bytes = parse_size(size_s);
        if (level == 1) {
            if (type_s == "Data") out.l1_data_bytes = std::max(out.l1_data_bytes, bytes);
            else if (type_s == "Instruction") out.l1_instruction_bytes = std::max(out.l1_instruction_bytes, bytes);
        } else if (level == 2) out.l2_bytes = std::max(out.l2_bytes, bytes);
        else if (level == 3) out.l3_bytes = std::max(out.l3_bytes, bytes);

        std::ifstream line_file(base + "coherency_line_size");
        size_t line_size = 0;
        if (line_file >> line_size) out.line_bytes = std::max(out.line_bytes, line_size);
    }
}
#endif

} // namespace

CpuCacheInfo CpuCacheInfo::detect() {
    CpuCacheInfo out;
    out.logical_cpus = std::max(1u, std::thread::hardware_concurrency());
#if defined(__x86_64__) || defined(__i386__)
    detect_x86(out);
#endif
#if defined(__linux__)
    detect_linux(out);
#endif
    if (out.line_bytes == 0) out.line_bytes = 64;
    if (out.cpu_name.empty()) out.cpu_name = "unknown";
    return out;
}

CachePlan CachePlan::detect() {
    CachePlan p;
    p.cpu = CpuCacheInfo::detect();
    p.cache_line_bytes = p.cpu.line_bytes;

    // Use fractions of detected capacity as working-set targets. These are
    // deliberately not cache-size constants: different CPUs produce different
    // budgets at runtime.
    p.l1_tile_bytes = p.cpu.l1_data_bytes ? p.cpu.l1_data_bytes / 2 : 0;
    p.l2_window_bytes = p.cpu.l2_bytes ? p.cpu.l2_bytes / 2 : 0;
    p.l3_window_bytes = p.cpu.l3_bytes ? p.cpu.l3_bytes / 2 : 0;
    return p;
}

CacheScheduler::CacheScheduler(CachePlan plan) : plan_(plan) {}

void CacheScheduler::prepare(const void* current, size_t current_bytes,
                             const void* next, size_t next_bytes,
                             const void* future, size_t future_bytes) const noexcept {
    (void)current;
    (void)current_bytes;
    const size_t next_n = plan_.l2_window_bytes ? std::min(next_bytes, plan_.l2_window_bytes) : next_bytes;
    const size_t future_n = plan_.l3_window_bytes ? std::min(future_bytes, plan_.l3_window_bytes) : future_bytes;
    prefetch_range(next, next_n, plan_.cache_line_bytes);
    prefetch_range(future, future_n, plan_.cache_line_bytes);
}

} // namespace laic
