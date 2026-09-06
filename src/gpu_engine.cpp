#include "laic/gpu_engine.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#if LAIC_HAVE_OPENCL
#include <CL/cl.h>
#endif
namespace laic::gpu {
#if LAIC_HAVE_OPENCL
static const char* kSource=R"CLC(
inline float h2f(ushort h){uint s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;if(e==0){if(!m)return as_float(s<<31);int sh=0;while((m&1024)==0){m<<=1;sh++;}m&=1023;u=(s<<31)|((127-14-sh)<<23)|(m<<13);}else if(e==31)u=(s<<31)|0x7f800000|(m<<13);else u=(s<<31)|((e+112)<<23)|(m<<13);return as_float(u);}
__kernel void matvec_f32(__global const float*w,__global const float*x,__global float*y,uint in,uint out){uint r=get_global_id(0);if(r>=out)return;float s=0;for(uint j=0;j<in;j++)s+=w[r*in+j]*x[j];y[r]=s;}
__kernel void matvec_f16(__global const ushort*w,__global const float*x,__global float*y,uint in,uint out){uint r=get_global_id(0);if(r>=out)return;float s=0;for(uint j=0;j<in;j++)s+=h2f(w[r*in+j])*x[j];y[r]=s;}
__kernel void matvec_q4(__global const uchar*w,__global const float*x,__global float*y,uint in,uint out){uint r=get_global_id(0);if(r>=out)return;float s=0;uint blocks=(in+31)/32;for(uint j=0;j<in;j++){uint b=j>>5,o=j&31;__global const uchar*q=w+r*blocks*18+b*18;uchar z=q[2+(o>>1)];int n=(o&1)?(z>>4):(z&15);s+=h2f((ushort)q[0]|((ushort)q[1]<<8))*(float)(n-8)*x[j];}y[r]=s;}
__kernel void matvec_q8(__global const uchar*w,__global const float*x,__global float*y,uint in,uint out){uint r=get_global_id(0);if(r>=out)return;float s=0;uint blocks=(in+31)/32;for(uint j=0;j<in;j++){uint b=j>>5,o=j&31;__global const uchar*q=w+r*blocks*34+b*34;s+=h2f((ushort)q[0]|((ushort)q[1]<<8))*(float)((char)q[2+o])*x[j];}y[r]=s;}
)CLC";
#endif
struct Engine::Impl{bool ok=false;videocore::Generation gen=videocore::Generation::Unknown;std::string detail="GPU runtime unavailable";Stats stats;std::mutex mu;
#if LAIC_HAVE_OPENCL
cl_context context=nullptr;cl_command_queue queue=nullptr;cl_program program=nullptr;cl_kernel kf32=nullptr,kf16=nullptr,kq4=nullptr,kq8=nullptr;std::unordered_map<const GgufTensor*,cl_mem>buffers;cl_mem xb=nullptr,yb=nullptr;size_t xcap=0,ycap=0;
#endif
Impl(){auto d=videocore::detect();gen=d.generation;
#if LAIC_HAVE_OPENCL
if(!d.compute_available||d.generation!=videocore::Generation::IV){detail=d.runtime.empty()?"VideoCore IV OpenCL not selected":d.runtime;return;}cl_uint np=0;if(clGetPlatformIDs(0,nullptr,&np)!=CL_SUCCESS||!np){detail="OpenCL platform missing";return;}std::vector<cl_platform_id>ps(np);clGetPlatformIDs(np,ps.data(),nullptr);cl_uint nd=0;if(clGetDeviceIDs(ps[0],CL_DEVICE_TYPE_GPU,0,nullptr,&nd)!=CL_SUCCESS||!nd){detail="VideoCore GPU device missing";return;}std::vector<cl_device_id>ds(nd);clGetDeviceIDs(ps[0],CL_DEVICE_TYPE_GPU,nd,ds.data(),nullptr);cl_device_id dev=ds[0];size_t mw=0,pref=1;cl_uint cu=0,clk=0;clGetDeviceInfo(dev,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(mw),&mw,nullptr);clGetDeviceInfo(dev,CL_DEVICE_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,sizeof(pref),&pref,nullptr);clGetDeviceInfo(dev,CL_DEVICE_MAX_COMPUTE_UNITS,sizeof(cu),&cu,nullptr);clGetDeviceInfo(dev,CL_DEVICE_MAX_CLOCK_FREQUENCY,sizeof(clk),&clk,nullptr);stats.max_work_group_size=mw;stats.preferred_work_group_multiple=pref;stats.selected_work_group_size=std::max<size_t>(1,std::min<size_t>(mw,12));stats.compute_units=cu;stats.clock_mhz=clk;cl_int e=0;context=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&e);if(!context||e){detail="clCreateContext failed";return;}queue=clCreateCommandQueue(context,dev,0,&e);program=clCreateProgramWithSource(context,1,&kSource,nullptr,&e);if(!program||e){detail="OpenCL program creation failed";return;}e=clBuildProgram(program,1,&dev,"",nullptr,nullptr);if(e){detail="VideoCore OpenCL kernel build failed";return;}kf32=clCreateKernel(program,"matvec_f32",&e);kf16=clCreateKernel(program,"matvec_f16",&e);kq4=clCreateKernel(program,"matvec_q4",&e);kq8=clCreateKernel(program,"matvec_q8",&e);if(e){detail="kernel creation failed";return;}ok=true;detail="VC4CL F32/F16/Q4_0/Q8_0 QPU kernels";
#endif
}
~Impl(){clear();}void clear()noexcept{
#if LAIC_HAVE_OPENCL
for(auto&p:buffers)if(p.second)clReleaseMemObject(p.second);buffers.clear();if(xb)clReleaseMemObject(xb);if(yb)clReleaseMemObject(yb);if(kf32)clReleaseKernel(kf32);if(kf16)clReleaseKernel(kf16);if(kq4)clReleaseKernel(kq4);if(kq8)clReleaseKernel(kq8);if(program)clReleaseProgram(program);if(queue)clReleaseCommandQueue(queue);if(context)clReleaseContext(context);kf32=kf16=kq4=kq8=nullptr;program=nullptr;queue=nullptr;context=nullptr;xb=yb=nullptr;xcap=ycap=0;
#endif
ok=false;}};
Engine::Engine():impl_(std::make_unique<Impl>()){}Engine::~Engine()=default;bool Engine::available()const noexcept{return impl_&&impl_->ok;}const std::string&Engine::detail()const noexcept{return impl_->detail;}videocore::Generation Engine::generation()const noexcept{return impl_?impl_->gen:videocore::Generation::Unknown;}const Stats&Engine::stats()const noexcept{return impl_->stats;}void Engine::reset_stats()noexcept{if(impl_){std::lock_guard<std::mutex>g(impl_->mu);impl_->stats={};}}
bool Engine::prepare(const GgufModel&){return available();}
#if LAIC_HAVE_OPENCL
static cl_mem buf(cl_context c,cl_mem&b,size_t&cap,size_t bytes,cl_mem_flags f){if(b&&cap>=bytes)return b;if(b)clReleaseMemObject(b);cl_int e=0;b=clCreateBuffer(c,f,bytes,nullptr,&e);if(e){b=nullptr;cap=0;return nullptr;}cap=bytes;return b;}
#endif
bool Engine::matvec(const GgufTensor&w,const float*x,float*y,size_t out,size_t in){
#if LAIC_HAVE_OPENCL
if(!available()){impl_->stats.matvec_fallbacks++;return false;}std::lock_guard<std::mutex>g(impl_->mu);impl_->stats.matvec_calls++;auto it=impl_->buffers.find(&w);if(it==impl_->buffers.end()){cl_int e=0;auto m=clCreateBuffer(impl_->context,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,w.bytes(),const_cast<uint8_t*>(w.data),&e);if(e){impl_->stats.matvec_fallbacks++;return false;}it=impl_->buffers.emplace(&w,m).first;}auto xmem=buf(impl_->context,impl_->xb,impl_->xcap,in*4,CL_MEM_READ_ONLY);auto ymem=buf(impl_->context,impl_->yb,impl_->ycap,out*4,CL_MEM_WRITE_ONLY);if(!xmem||!ymem){impl_->stats.matvec_fallbacks++;return false;}if(clEnqueueWriteBuffer(impl_->queue,xmem,CL_TRUE,0,in*4,x,0,nullptr,nullptr)!=CL_SUCCESS){impl_->stats.matvec_fallbacks++;return false;}cl_kernel k=w.type==GgmlType::F32?impl_->kf32:w.type==GgmlType::F16?impl_->kf16:w.type==GgmlType::Q4_0?impl_->kq4:w.type==GgmlType::Q8_0?impl_->kq8:nullptr;if(!k){impl_->stats.matvec_fallbacks++;return false;}cl_uint I=cl_uint(in),O=cl_uint(out);if(clSetKernelArg(k,0,sizeof(cl_mem),&it->second)||clSetKernelArg(k,1,sizeof(cl_mem),&xmem)||clSetKernelArg(k,2,sizeof(cl_mem),&ymem)||clSetKernelArg(k,3,sizeof(I),&I)||clSetKernelArg(k,4,sizeof(O),&O)){impl_->stats.matvec_fallbacks++;return false;}size_t local=impl_->stats.selected_work_group_size,global=((out+local-1)/local)*local;auto t=std::chrono::steady_clock::now();if(clEnqueueNDRangeKernel(impl_->queue,k,1,nullptr,&global,&local,0,nullptr,nullptr)!=CL_SUCCESS||clFinish(impl_->queue)!=CL_SUCCESS||clEnqueueReadBuffer(impl_->queue,ymem,CL_TRUE,0,out*4,y,0,nullptr,nullptr)!=CL_SUCCESS){impl_->stats.matvec_fallbacks++;return false;}impl_->stats.gpu_ns+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-t).count());impl_->stats.gpu_flops+=uint64_t(2ull*out*in);impl_->stats.bytes_read+=w.bytes()+in*4;impl_->stats.bytes_written+=out*4;impl_->stats.matvec_gpu_calls++;return true;
#else
(void)w;(void)x;(void)y;(void)out;(void)in;impl_->stats.matvec_fallbacks++;return false;
#endif
}
bool Engine::matmul(const float*,const float*,float*,size_t,size_t,size_t){return false;}
bool Engine::partitioned_matvec(const GgufTensor&w,const float*x,float*y,size_t out,size_t in,size_t gpu_rows){gpu_rows=std::min(gpu_rows,out);if(!gpu_rows)return false;if(gpu_rows==out)return matvec(w,x,y,out,in);std::atomic<bool>ok{false};std::thread t([&]{ok=matvec(w,x,y,gpu_rows,in);});for(size_t r=gpu_rows;r<out;r++){float s=0;for(size_t j=0;j<in;j++)s+=w.value(r*in+j)*x[j];y[r]=s;}t.join();return ok;}
void Engine::clear()noexcept{if(impl_)impl_->clear();}
}
