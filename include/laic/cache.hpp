#pragma once
#include <cstddef>
#include <string>

namespace laic {

struct CpuCacheInfo {
    size_t l1_data_bytes = 0;
    size_t l1_instruction_bytes = 0;
    size_t l2_bytes = 0;
    size_t l3_bytes = 0;
    size_t line_bytes = 0;
    unsigned logical_cpus = 1;
    std::string cpu_name;

    static CpuCacheInfo detect();
};

struct CachePlan {
    CpuCacheInfo cpu;
    size_t l1_tile_bytes = 0;
    size_t l2_window_bytes = 0;
    size_t l3_window_bytes = 0;
    size_t cache_line_bytes = 0;

    static CachePlan detect();
};

class CacheScheduler {
public:
    explicit CacheScheduler(CachePlan plan = CachePlan::detect());
    void prepare(const void* current, size_t current_bytes,
                 const void* next, size_t next_bytes,
                 const void* future, size_t future_bytes) const noexcept;
    const CachePlan& plan() const noexcept { return plan_; }
private:
    CachePlan plan_;
};

} // namespace laic
