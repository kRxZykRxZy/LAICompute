#include "laic/cpu.hpp"
#include <algorithm>
#include <fstream>
#include <thread>
#include <string>
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace laic {
namespace {
size_t parse_size(std::string s) {
    if (s.empty()) return 0;
    char u = s.back();
    if (u=='K'||u=='k'||u=='M'||u=='m') s.pop_back();
    size_t v=0; try { v=std::stoull(s); } catch (...) { return 0; }
    if (u=='K'||u=='k') v*=1024;
    if (u=='M'||u=='m') v*=1024*1024;
    return v;
}
#if defined(__x86_64__) || defined(__i386__)
void x86(CpuInfo& c) {
    unsigned a,b,d,e;
    if (__get_cpuid_max(0,nullptr)==0) return;
    __cpuid(0,a,b,d,e);
    unsigned maxleaf=a;
    char vendor[13]{}; *reinterpret_cast<unsigned*>(vendor)=b; *reinterpret_cast<unsigned*>(vendor+4)=d; *reinterpret_cast<unsigned*>(vendor+8)=e;
    c.name=vendor;
    unsigned r1,r2,r3,r4; __cpuid(1,r1,r2,r3,r4);
    c.features.avx=(r3&(1u<<28)); c.features.f16c=(r3&(1u<<29)); c.features.fma=(r3&(1u<<12));
    if (maxleaf>=7) { unsigned ebx; __cpuid_count(7,0,r1,ebx,r3,r4); c.features.avx2=ebx&(1u<<5); }
    if (maxleaf>=4) for(unsigned s=0;s<32;s++) {
        __cpuid_count(4,s,r1,r2,r3,r4); unsigned type=r1&31; if(!type) break;
        unsigned level=(r1>>5)&7; size_t line=(r2&0xfff)+1ULL, parts=((r2>>12)&0x3ff)+1ULL, ways=((r2>>22)&0x3ff)+1ULL, sets=static_cast<size_t>(r3)+1;
        size_t bytes=line*parts*ways*sets; c.cache_line=std::max(c.cache_line,line);
        if(level==1 && type==1) c.l1d=std::max(c.l1d,bytes); else if(level==1 && type==2) c.l1i=std::max(c.l1i,bytes); else if(level==2) c.l2=std::max(c.l2,bytes); else if(level==3) c.l3=std::max(c.l3,bytes);
    }
}
#endif
#if defined(__linux__)
void linux_sysfs(CpuInfo& c) {
    c.logical_cpus=std::max(1u,std::thread::hardware_concurrency());
    std::ifstream f("/proc/cpuinfo"); std::string line;
    while(std::getline(f,line)) if(line.rfind("model name",0)==0 || line.rfind("Hardware",0)==0) { auto p=line.find(':'); if(p!=std::string::npos)c.name=line.substr(p+2); break; }
    for(int i=0;i<16;i++) { std::string base="/sys/devices/system/cpu/cpu0/cache/index"+std::to_string(i)+"/"; std::ifstream lf(base+"level"),tf(base+"type"),sf(base+"size"),cf(base+"coherency_line_size"); int level; std::string type,size; size_t line_size;
        if(!(lf>>level)||!(tf>>type)||!(sf>>size)) continue; size_t bytes=parse_size(size); if(level==1&&type=="Data")c.l1d=std::max(c.l1d,bytes); else if(level==1&&type=="Instruction")c.l1i=std::max(c.l1i,bytes); else if(level==2)c.l2=std::max(c.l2,bytes); else if(level==3)c.l3=std::max(c.l3,bytes); if(cf>>line_size)c.cache_line=std::max(c.cache_line,line_size);
    }
    // Count physical package/core pairs where Linux exposes them.
    std::ifstream p("/proc/cpuinfo"); std::string l; std::string package,core,last; unsigned count=0; while(std::getline(p,l)) { if(l.rfind("physical id",0)==0) package=l; if(l.rfind("core id",0)==0) core=l; if(l.empty() && (!package.empty()||!core.empty())) { std::string key=package+"/"+core; if(key!=last){++count;last=key;} package.clear();core.clear(); } } if(count)c.physical_cores=count;
}
#endif
}
CpuInfo CpuInfo::detect() { CpuInfo c;
#if defined(__x86_64__) || defined(__i386__)
 x86(c);
#endif
#if defined(__linux__)
 linux_sysfs(c);
#endif
 if(!c.logical_cpus)c.logical_cpus=1; if(!c.physical_cores)c.physical_cores=std::min(2u,c.logical_cpus); if(!c.name.size())c.name="unknown"; return c;
}
} // namespace laic
