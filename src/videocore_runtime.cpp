#include "laic/videocore_runtime.hpp"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#if LAIC_HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace laic::videocore {
namespace {
void* open_any(const char* const* names) noexcept { for(size_t i=0;names[i];++i) if(void*h=dlopen(names[i],RTLD_LAZY|RTLD_LOCAL)) return h; return nullptr; }
using cl_int=int; using cl_uint=unsigned; using cl_platform_id=void*; using cl_device_id=void*; using cl_device_type=unsigned long long;
using clGetPlatformIDs_fn=cl_int(*)(cl_uint,cl_platform_id*,cl_uint*); using clGetDeviceIDs_fn=cl_int(*)(cl_platform_id,cl_device_type,cl_uint,cl_device_id*,cl_uint*); using clGetDeviceInfo_fn=cl_int(*)(cl_device_id,cl_uint,size_t,void*,size_t*);
constexpr cl_int CL_SUCCESS=0; constexpr cl_device_type CL_DEVICE_TYPE_GPU=(1ull<<2); constexpr cl_uint CL_DEVICE_NAME=0x102B; constexpr cl_uint CL_DEVICE_VERSION=0x102F;
RuntimeInfo probe_iv() noexcept {
    RuntimeInfo r; r.generation=Generation::IV; r.api="OpenCL / VC4CL"; const char*libs[]={"libOpenCL.so.1","libOpenCL.so",nullptr}; void*h=open_any(libs);
    if(!h){r.detail="OpenCL loader not found";return r;} auto platforms=(clGetPlatformIDs_fn)dlsym(h,"clGetPlatformIDs"); auto devices=(clGetDeviceIDs_fn)dlsym(h,"clGetDeviceIDs"); auto info=(clGetDeviceInfo_fn)dlsym(h,"clGetDeviceInfo");
    if(!platforms||!devices||!info){r.detail="OpenCL entry points unavailable";dlclose(h);return r;} cl_uint np=0;
    if(platforms(0,nullptr,&np)!=CL_SUCCESS||np==0){r.detail="No OpenCL platform";dlclose(h);return r;} cl_platform_id p[8]{};if(np>8)np=8;
    if(platforms(np,p,nullptr)!=CL_SUCCESS){r.detail="OpenCL platform query failed";dlclose(h);return r;}
    cl_device_id fallback_gpu=nullptr; char fallback_name[256]{};
    for(cl_uint i=0;i<np;i++){cl_uint nd=0;if(devices(p[i],CL_DEVICE_TYPE_GPU,0,nullptr,&nd)!=CL_SUCCESS||!nd)continue;cl_device_id d[8]{};if(nd>8)nd=8;if(devices(p[i],CL_DEVICE_TYPE_GPU,nd,d,nullptr)!=CL_SUCCESS)continue;
        for(cl_uint j=0;j<nd;j++){char name[256]{},version[256]{};info(d[j],CL_DEVICE_NAME,sizeof(name),name,nullptr);info(d[j],CL_DEVICE_VERSION,sizeof(version),version,nullptr);
            std::string nv=name;for(char&c:nv)if(c>='A'&&c<='Z')c=char(c-'A'+'a');
            r.all_devices.push_back(name);if(!r.all_device_versions.empty())r.all_device_versions+=", ";r.all_device_versions+=version;
            if(!fallback_gpu){fallback_gpu=d[j];std::snprintf(fallback_name,sizeof(fallback_name),"%s",name);r.device=name;r.detail=version;}
            if(nv.find("videocore")!=std::string::npos||nv.find("vc4")!=std::string::npos||nv.find("broadcom")!=std::string::npos||nv.find("bcm")!=std::string::npos||nv.find("gpu")!=std::string::npos){r.available=true;r.device=name;r.detail=version;fallback_gpu=nullptr;break;}}
        if(r.available)break;}
    if(!r.available&&fallback_gpu){r.available=true;r.device=fallback_name;r.detail+=" (fallback: accepted as VC4CL GPU on VideoCore IV system)";}
    dlclose(h);return r;
}
RuntimeInfo probe_vulkan(Generation g) noexcept {
    RuntimeInfo r;r.generation=g;r.api="Vulkan / Mesa V3DV";
#if LAIC_HAVE_VULKAN
    uint32_t count=0;if(vkEnumerateInstanceExtensionProperties(nullptr,&count,nullptr)!=VK_SUCCESS){r.detail="Vulkan instance query failed";return r;}
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO,nullptr,"LAICompute",1,"LAICompute",1,VK_API_VERSION_1_0};VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,nullptr,0,&app,0,nullptr,0,nullptr};VkInstance instance=VK_NULL_HANDLE;
    if(vkCreateInstance(&ci,nullptr,&instance)!=VK_SUCCESS){r.detail="Vulkan instance creation failed";return r;}uint32_t n=0;
    if(vkEnumeratePhysicalDevices(instance,&n,nullptr)!=VK_SUCCESS||n==0){r.detail="No Vulkan physical device";vkDestroyInstance(instance,nullptr);return r;}VkPhysicalDevice devices[8]{};if(n>8)n=8;vkEnumeratePhysicalDevices(instance,&n,devices);
    for(uint32_t i=0;i<n;i++){VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(devices[i],&p);if(p.vendorID==0x14E4){r.available=true;r.device=p.deviceName;r.detail="Vulkan "+std::to_string(VK_VERSION_MAJOR(p.apiVersion))+"."+std::to_string(VK_VERSION_MINOR(p.apiVersion))+" via V3DV";break;}}
    vkDestroyInstance(instance,nullptr);
#else
    (void)g;const char*libs[]={"libvulkan.so.1","libvulkan.so",nullptr};void*h=open_any(libs);r.detail=h?"Vulkan loader present; rebuild with Vulkan headers for V3DV device probing":"Vulkan loader not found";r.available=h!=nullptr;if(h)dlclose(h);
#endif
    return r;
}
}
RuntimeInfo probe_runtime(const DeviceInfo& device) noexcept {if(!device.present)return{};if(device.generation==Generation::IV)return probe_iv();if(device.generation==Generation::VI||device.generation==Generation::VII)return probe_vulkan(device.generation);return{};}
bool runtime_available(const DeviceInfo& device) noexcept{return probe_runtime(device).available;}
} // namespace laic::videocore
