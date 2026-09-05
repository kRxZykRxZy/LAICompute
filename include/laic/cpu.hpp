#pragma once
#include <cstddef>
#include <string>

namespace laic {

struct CpuFeatures {
    bool avx = false;
    bool f16c = false;
    bool avx2 = false;
    bool fma = false;
};

struct CpuInfo {
    std::string name;
    unsigned logical_cpus = 1;
    unsigned physical_cores = 1;
    size_t cache_line = 64;
    size_t l1d = 0;
    size_t l1i = 0;
    size_t l2 = 0;
    size_t l3 = 0;
    CpuFeatures features;
    static CpuInfo detect();
};

} // namespace laic
