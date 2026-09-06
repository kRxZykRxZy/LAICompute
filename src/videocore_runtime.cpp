#include "laic/videocore_runtime.hpp"
#include <dlfcn.h>
#include <cstring>

namespace laic::videocore {
namespace {
void* open_any(const char* const* names) noexcept {
    for (size_t i=0; names[i]; ++i) {
        if (void* h=dlopen(names[i], RTLD_LAZY|RTLD_LOCAL)) return h;
    }
    return nullptr;
}

// Minimal OpenCL ABI declarations. We intentionally avoid a hard build-time
// dependency on OpenCL headers/libraries so CPU-only builds remain portable.
using cl_int=int;
using cl_uint=unsigned;
using cl_platform_id=void*;
using cl_device_id=void*;
using cl_device_type=unsigned long long;
using clGetPlatformIDs_fn=cl_int(*)(cl_uint,cl_platform_id*,cl_uint*);
using clGetDeviceIDs_fn=cl_int(*)(cl_platform_id,cl_device_type,cl_uint,cl_device_id*,cl_uint*);
using clGetDeviceInfo_fn=cl_int(*)(cl_device_id,cl_uint,size_t,void*,size_t*);
constexpr cl_int CL_SUCCESS=0;
constexpr cl_device_type CL_DEVICE_TYPE_GPU=(1ull<<2);
constexpr cl_uint CL_DEVICE_NAME=0x102B;
constexpr cl_uint CL_DEVICE_VERSION=0x102F;

RuntimeInfo probe_iv() noexcept {
    RuntimeInfo r; r.generation=Generation::IV; r.api="OpenCL / VC4CL";
    const char* libs[]={"libOpenCL.so.1","libOpenCL.so",nullptr};
    void* h=open_any(libs);
    if(!h){r.detail="OpenCL loader not found";return r;}
    auto platforms=(clGetPlatformIDs_fn)dlsym(h,"clGetPlatformIDs");
    auto devices=(clGetDeviceIDs_fn)dlsym(h,"clGetDeviceIDs");
    auto info=(clGetDeviceInfo_fn)dlsym(h,"clGetDeviceInfo");
    if(!platforms||!devices||!info){r.detail="OpenCL entry points unavailable";dlclose(h);return r;}
    cl_uint np=0;
    if(platforms(0,nullptr,&np)!=CL_SUCCESS||np==0){r.detail="No OpenCL platform";dlclose(h);return r;}
    cl_platform_id p[8]{}; if(np>8)np=8;
    if(platforms(np,p,nullptr)!=CL_SUCCESS){r.detail="OpenCL platform query failed";dlclose(h);return r;}
    for(cl_uint i=0;i<np;i++){
        cl_uint nd=0; if(devices(p[i],CL_DEVICE_TYPE_GPU,0,nullptr,&nd)!=CL_SUCCESS||!nd)continue;
        cl_device_id d[8]{}; if(nd>8)nd=8;
        if(devices(p[i],CL_DEVICE_TYPE_GPU,nd,d,nullptr)!=CL_SUCCESS)continue;
        char name[256]{}; char version[256]{};
        info(d[0],CL_DEVICE_NAME,sizeof(name),name,nullptr);
        info(d[0],CL_DEVICE_VERSION,sizeof(version),version,nullptr);
        r.device=name;
        r.detail=std::string(version);
        // VC4CL identifies itself as VideoCore IV GPU. Do not treat an
        // unrelated OpenCL GPU as a VideoCore IV runtime.
        std::string s=name; for(char& c:s) if(c>='A'&&c<='Z') c=char(c-'A'+'a');
        if(s.find("videocore")!=std::string::npos || s.find("vc4")!=std::string::npos){r.available=true;break;}
    }
    dlclose(h); return r;
}

// Vulkan loader ABI probe. The full Vulkan headers are deliberately optional;
// the runtime only needs vkGetInstanceProcAddr to establish that a Vulkan
// loader exists. Driver/device validation is completed by the inference
// backend when Vulkan support is compiled in.
RuntimeInfo probe_vulkan(Generation g) noexcept {
    RuntimeInfo r; r.generation=g; r.api="Vulkan / Mesa V3DV";
    const char* libs[]={"libvulkan.so.1","libvulkan.so",nullptr};
    void* h=open_any(libs);
    if(!h){r.detail="Vulkan loader not found";return r;}
    void* p=dlsym(h,"vkGetInstanceProcAddr");
    if(!p){r.detail="Vulkan entry point unavailable";dlclose(h);return r;}
    r.available=true;
    r.device=(g==Generation::VI)?"VideoCore VI":"VideoCore VII";
    r.detail="Vulkan loader present; V3DV device probing deferred to Vulkan backend";
    dlclose(h); return r;
}
}

RuntimeInfo probe_runtime(const DeviceInfo& device) noexcept {
    if(!device.present) return {};
    if(device.generation==Generation::IV) return probe_iv();
    if(device.generation==Generation::VI||device.generation==Generation::VII) return probe_vulkan(device.generation);
    return {};
}

bool runtime_available(const DeviceInfo& device) noexcept { return probe_runtime(device).available; }

} // namespace laic::videocore
