#include "laic/videocore.hpp"
#include "laic/videocore_runtime.hpp"
#include <atomic>
#include <cctype>
#include <fstream>
#include <sstream>

namespace laic::videocore {
namespace {
std::string lower(std::string s){for(char&c:s)c=char(std::tolower(static_cast<unsigned char>(c)));return s;}
std::string read_file(const char* path){std::ifstream f(path);std::ostringstream s;s<<f.rdbuf();return s.str();}
std::string cpuinfo(){return read_file("/proc/cpuinfo");}
std::atomic<Backend> g_requested{Backend::CPU};
Generation detect_generation(const std::string& text){
    const auto t=lower(text);
    if(t.find("2712")!=std::string::npos||t.find("bcm2712")!=std::string::npos||t.find("videocore vii")!=std::string::npos)return Generation::VII;
    if(t.find("2711")!=std::string::npos||t.find("bcm2711")!=std::string::npos||t.find("videocore vi")!=std::string::npos)return Generation::VI;
    if(t.find("2835")!=std::string::npos||t.find("2836")!=std::string::npos||t.find("2837")!=std::string::npos||t.find("bcm2835")!=std::string::npos||t.find("bcm2836")!=std::string::npos||t.find("bcm2837")!=std::string::npos||t.find("videocore iv")!=std::string::npos)return Generation::IV;
    return Generation::Unknown;
}
}
Backend backend_from_string(const std::string& value) noexcept{const auto v=lower(value);Backend b=v=="gpu"||v=="videocore"?Backend::GPU:(v=="both"||v=="hybrid"?Backend::Both:Backend::CPU);g_requested.store(b,std::memory_order_relaxed);return b;}
void set_requested_backend(Backend b) noexcept{g_requested.store(b,std::memory_order_relaxed);}
Backend requested_backend() noexcept{return g_requested.load(std::memory_order_relaxed);}
const char* backend_name(Backend b) noexcept{return b==Backend::GPU?"GPU":b==Backend::Both?"Both":"CPU";}
std::string generation_name(Generation g) noexcept{return g==Generation::IV?"VideoCore IV":g==Generation::VI?"VideoCore VI":g==Generation::VII?"VideoCore VII":"Unknown";}
DeviceInfo detect(){
    DeviceInfo d;std::string all=cpuinfo()+read_file("/sys/firmware/devicetree/base/compatible")+read_file("/sys/firmware/devicetree/base/model");d.generation=detect_generation(all);if(d.generation==Generation::Unknown)return d;d.present=true;d.name=generation_name(d.generation);
    if(d.generation==Generation::IV){d.qpus=12;d.clock_mhz=250;d.runtime="VC4CL / OpenCL";}
    else if(d.generation==Generation::VI){d.qpus=8;d.clock_mhz=500;d.runtime="Mesa V3DV / Vulkan";}
    else {d.qpus=12;d.clock_mhz=1000;d.runtime="Mesa V3DV / Vulkan";}
    d.theoretical_gflops=theoretical_peak_gflops(d);d.compute_available=runtime_available(d);return d;
}
double theoretical_peak_gflops(const DeviceInfo& d) noexcept{return double(d.qpus)*4.0*2.0*double(d.clock_mhz)/1000.0;}
} // namespace laic::videocore
