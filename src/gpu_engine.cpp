#include "laic/gpu_engine.hpp"
#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>
#if LAIC_HAVE_OPENCL
#include <CL/cl.h>
#endif
namespace laic::gpu { namespace {
#if LAIC_HAVE_OPENCL
static const char* kSource=R"CLC(
#define TS 8
inline float h2f(ushort h){uint s=(h>>15)&1,e=(h>>10)&31,m=h&1023,u;if(e==0){if(m==0)u=s<<31;else{int sh=0;while((m&1024)==0){m<<=1;sh++;}m&=1023;u=(s<<31)|((127-14-sh)<<23)|(m<<13);}}else if(e==31)u=(s<<31)|0x7f800000|(m<<13);else u=(s<<31)|((e+112)<<23)|(m<<13);return as_float(u);}
__kernel void matvec_f32(__global const float*w,__global const float*x,__global float*y,uint in,uint row0,uint out){uint r=get_global_id(0)+row0;if(r>=row0+out)return;__local float xt[TS*8];float s=0.0f;for(uint base=0;base<in;base+=TS*8){uint lid=get_local_id(0);for(uint j=lid;j<TS*8;j+=TS){uint p=base+j;xt[j]=(p<in)?x[p]:0.0f;}barrier(CLK_LOCAL_MEM_FENCE);uint e=min(in,base+TS*8);for(uint j=base;j<e;j++)s+=w[r*in+j]*xt[j-base];barrier(CLK_LOCAL_MEM_FENCE);}y[r]=s;}
__kernel void matvec_q4(__global const uchar*w,__global const float*x,__global float*y,uint in,uint row0,uint out){uint r=get_global_id(0)+row0;if(r>=row0+out)return;__local float xt[TS*8];float s=0.0f;uint blocks=(in+31)/32;for(uint base=0;base<in;base+=TS*8){uint lid=get_local_id(0);for(uint j=lid;j<TS*8;j+=TS){uint p=base+j;xt[j]=(p<in)?x[p]:0.0f;}barrier(CLK_LOCAL_MEM_FENCE);uint e=min(in,base+TS*8);for(uint j=base;j<e;j++){uint b=j>>5,off=j&31;__global const uchar*q=w+r*blocks*18+b*18;uchar z=q[2+(off>>1)];int n=(off&1)?(z>>4):(z&15);float d=h2f((ushort)q[0]|((ushort)q[1]<<8));s+=d*(float)(n-8)*xt[j-base];}barrier(CLK_LOCAL_MEM_FENCE);}y[r]=s;}
__kernel void matvec_q8(__global const uchar*w,__global const float*x,__global float*y,uint in,uint row0,uint out){uint r=get_global_id(0)+row0;if(r>=row0+out)return;__local float xt[TS*8];float s=0.0f;uint blocks=(in+31)/32;for(uint base=0;base<in;base+=TS*8){uint lid=get_local_id(0);for(uint j=lid;j<TS*8;j+=TS){uint p=base+j;xt[j]=(p<in)?x[p]:0.0f;}barrier(CLK_LOCAL_MEM_FENCE);uint e=min(in,base+TS*8);for(uint j=base;j<e;j++){uint b=j>>5,off=j&31;__global const uchar*q=w+r*blocks*34+b*34;float d=h2f((ushort)q[0]|((ushort)q[1]<<8));s+=d*(float)((char)q[2+off])*xt[j-base];}barrier(CLK_LOCAL_MEM_FENCE);}y[r]=s;}
__kernel void matmul_f32(__global const float*a,__global const float*b,__global float*c,uint m,uint n,uint k){uint ly=get_local_id(0),lx=get_local_id(1),row=get_group_id(0)*TS+ly,col=get_group_id(1)*TS+lx;__local float A[TS][TS],B[TS][TS];float acc=0.0f;for(uint base=0;base<k;base+=TS){uint ak=base+lx,bk=base+ly;A[ly][lx]=(row<m&&ak<k)?a[row*k+ak]:0.0f;B[ly][lx]=(bk<k&&col<n)?b[bk*n+col]:0.0f;barrier(CLK_LOCAL_MEM_FENCE);for(uint j=0;j<TS;j++)acc+=A[ly][j]*B[j][lx];barrier(CLK_LOCAL_MEM_FENCE);}if(row<m&&col<n)c[row*n+col]=acc;}
)CLC";
#endif
}
struct Engine::Impl{bool ok=false;videocore::Generation gen=videocore::Generation::Unknown;std::string detail="GPU runtime unavailable";
#if LAIC_HAVE_OPENCL
cl_context context=nullptr;cl_command_queue queue=nullptr;cl_program program=nullptr;cl_kernel mv_f32=nullptr,mv_q4=nullptr,mv_q8=nullptr,mm_f32=nullptr;std::unordered_map<const GgufTensor*,cl_mem>buffers;cl_mem xbuf=nullptr,ybuf=nullptr,abuf=nullptr,bbuf=nullptr,cbuf=nullptr;std::size_t xcap=0,ycap=0,acap=0,bbcap=0,ccap=0;std::mutex mu;
#endif
Impl(){auto d=videocore::detect();gen=d.generation;
#if LAIC_HAVE_OPENCL
if(!d.compute_available||d.generation!=videocore::Generation::IV){detail=d.runtime.empty()?"VideoCore IV OpenCL not selected":d.runtime;return;}cl_int e=CL_SUCCESS;cl_uint np=0;if(clGetPlatformIDs(0,nullptr,&np)!=CL_SUCCESS||!np){detail="OpenCL platform missing";return;}std::vector<cl_platform_id>ps(np);clGetPlatformIDs(np,ps.data(),nullptr);cl_platform_id p=ps[0];cl_uint nd=0;if(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,0,nullptr,&nd)!=CL_SUCCESS||!nd){detail="VideoCore GPU device missing";return;}std::vector<cl_device_id>ds(nd);if(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,nd,ds.data(),nullptr)!=CL_SUCCESS){detail="cannot enumerate GPU";return;}cl_device_id dev=ds[0];context=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&e);if(!context||e!=CL_SUCCESS){detail="clCreateContext failed";return;}queue=clCreateCommandQueue(context,dev,0,&e);if(!queue||e!=CL_SUCCESS){detail="clCreateCommandQueue failed";return;}program=clCreateProgramWithSource(context,1,&kSource,nullptr,&e);if(!program||e!=CL_SUCCESS){detail="kernel program creation failed";return;}e=clBuildProgram(program,1,&dev,"",nullptr,nullptr);if(e!=CL_SUCCESS){detail="VideoCore OpenCL kernel build failed";return;}mv_f32=clCreateKernel(program,"matvec_f32",&e);mv_q4=clCreateKernel(program,"matvec_q4",&e);mv_q8=clCreateKernel(program,"matvec_q8",&e);mm_f32=clCreateKernel(program,"matmul_f32",&e);if(e!=CL_SUCCESS){detail="kernel creation failed";return;}ok=true;detail="VC4CL OpenCL tiled kernels with persistent model buffers";
#else
(void)d;
#endif
}
~Impl(){clear();}void clear()noexcept{
#if LAIC_HAVE_OPENCL
for(auto&kv:buffers)if(kv.second)clReleaseMemObject(kv.second);buffers.clear();if(xbuf)clReleaseMemObject(xbuf);if(ybuf)clReleaseMemObject(ybuf);if(abuf)clReleaseMemObject(abuf);if(bbuf)clReleaseMemObject(bbuf);if(cbuf)clReleaseMemObject(cbuf);if(mv_f32)clReleaseKernel(mv_f32);if(mv_q4)clReleaseKernel(mv_q4);if(mv_q8)clReleaseKernel(mv_q8);if(mm_f32)clReleaseKernel(mm_f32);if(program)clReleaseProgram(program);if(queue)clReleaseCommandQueue(queue);if(context)clReleaseContext(context);mv_f32=mv_q4=mv_q8=mm_f32=nullptr;program=nullptr;queue=nullptr;context=nullptr;xbuf=ybuf=abuf=bbuf=cbuf=nullptr;xcap=ycap=acap=bbcap=ccap=0;
#endif
ok=false;}};
Engine::Engine():impl_(std::make_unique<Impl>()){}Engine::~Engine()=default;bool Engine::available()const noexcept{return impl_&&impl_->ok;}const std::string&Engine::detail()const noexcept{return impl_->detail;}videocore::Generation Engine::generation()const noexcept{return impl_?impl_->gen:videocore::Generation::Unknown;}
bool Engine::prepare(const GgufModel&model){(void)model;return available();}
#if LAIC_HAVE_OPENCL
static cl_mem ensure_buf(cl_context c,cl_mem&b,std::size_t&cap,std::size_t bytes,cl_mem_flags flags){if(cap>=bytes&&b)return b;if(b)clReleaseMemObject(b);cl_int e=0;b=clCreateBuffer(c,flags,bytes,nullptr,&e);if(e!=CL_SUCCESS){b=nullptr;cap=0;return nullptr;}cap=bytes;return b;}
#endif
bool Engine::matvec(const GgufTensor&w,const float*x,float*y,std::size_t out,std::size_t in){
#if LAIC_HAVE_OPENCL
if(!available()||w.type==GgmlType::F16)return false;std::lock_guard<std::mutex>g(impl_->mu);auto it=impl_->buffers.find(&w);if(it==impl_->buffers.end()){cl_int e=0;cl_mem b=clCreateBuffer(impl_->context,CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR,w.bytes(),const_cast<uint8_t*>(w.data),&e);if(e!=CL_SUCCESS)return false;it=impl_->buffers.emplace(&w,b).first;}cl_mem xb=ensure_buf(impl_->context,impl_->xbuf,impl_->xcap,in*sizeof(float),CL_MEM_READ_ONLY),yb=ensure_buf(impl_->context,impl_->ybuf,impl_->ycap,out*sizeof(float),CL_MEM_WRITE_ONLY);if(!xb||!yb)return false;if(clEnqueueWriteBuffer(impl_->queue,xb,CL_FALSE,0,in*sizeof(float),x,0,nullptr,nullptr)!=CL_SUCCESS)return false;cl_kernel k=w.type==GgmlType::F32?impl_->mv_f32:(w.type==GgmlType::Q4_0?impl_->mv_q4:(w.type==GgmlType::Q8_0?impl_->mv_q8:nullptr));if(!k)return false;cl_uint ui=cl_uint(in),zero=0,uo=cl_uint(out);if(clSetKernelArg(k,0,sizeof(cl_mem),&it->second)||clSetKernelArg(k,1,sizeof(cl_mem),&xb)||clSetKernelArg(k,2,sizeof(cl_mem),&yb)||clSetKernelArg(k,3,sizeof(ui),&ui)||clSetKernelArg(k,4,sizeof(zero),&zero)||clSetKernelArg(k,5,sizeof(uo),&uo))return false;size_t local=8,global=((out+local-1)/local)*local;if(clEnqueueNDRangeKernel(impl_->queue,k,1,nullptr,&global,&local,0,nullptr,nullptr)!=CL_SUCCESS)return false;if(clFinish(impl_->queue)!=CL_SUCCESS)return false;return clEnqueueReadBuffer(impl_->queue,yb,CL_TRUE,0,out*sizeof(float),y,0,nullptr,nullptr)==CL_SUCCESS;
#else
(void)w;(void)x;(void)y;(void)out;(void)in;return false;
#endif
}
bool Engine::matmul(const float*a,const float*b,float*c,std::size_t m,std::size_t n,std::size_t k){
#if LAIC_HAVE_OPENCL
if(!available())return false;std::lock_guard<std::mutex>g(impl_->mu);auto A=ensure_buf(impl_->context,impl_->abuf,impl_->acap,m*k*sizeof(float),CL_MEM_READ_ONLY);auto B=ensure_buf(impl_->context,impl_->bbuf,impl_->bbcap,k*n*sizeof(float),CL_MEM_READ_ONLY);auto C=ensure_buf(impl_->context,impl_->cbuf,impl_->ccap,m*n*sizeof(float),CL_MEM_WRITE_ONLY);if(!A||!B||!C)return false;if(clEnqueueWriteBuffer(impl_->queue,A,CL_FALSE,0,m*k*sizeof(float),a,0,nullptr,nullptr)!=CL_SUCCESS||clEnqueueWriteBuffer(impl_->queue,B,CL_FALSE,0,k*n*sizeof(float),b,0,nullptr,nullptr)!=CL_SUCCESS)return false;cl_uint M=cl_uint(m),N=cl_uint(n),K=cl_uint(k);if(clSetKernelArg(impl_->mm_f32,0,sizeof(cl_mem),&A)||clSetKernelArg(impl_->mm_f32,1,sizeof(cl_mem),&B)||clSetKernelArg(impl_->mm_f32,2,sizeof(cl_mem),&C)||clSetKernelArg(impl_->mm_f32,3,sizeof(M),&M)||clSetKernelArg(impl_->mm_f32,4,sizeof(N),&N)||clSetKernelArg(impl_->mm_f32,5,sizeof(K),&K))return false;size_t local[2]={8,8},global[2]={((m+7)/8)*8,((n+7)/8)*8};if(clEnqueueNDRangeKernel(impl_->queue,impl_->mm_f32,2,nullptr,global,local,0,nullptr,nullptr)!=CL_SUCCESS)return false;if(clFinish(impl_->queue)!=CL_SUCCESS)return false;return clEnqueueReadBuffer(impl_->queue,C,CL_TRUE,0,m*n*sizeof(float),c,0,nullptr,nullptr)==CL_SUCCESS;
#else
(void)a;(void)b;(void)c;(void)m;(void)n;(void)k;return false;
#endif
}
bool Engine::partitioned_matvec(const GgufTensor&w,const float*x,float*y,std::size_t out,std::size_t in,std::size_t gpu_rows){gpu_rows=std::min(gpu_rows,out);if(!gpu_rows)return false;if(gpu_rows==out)return matvec(w,x,y,out,in);std::atomic<bool>gpu_ok{false};std::thread gt([&]{gpu_ok=matvec(w,x,y,gpu_rows,in);});for(std::size_t r=gpu_rows;r<out;r++){float s=0;for(std::size_t j=0;j<in;j++)s+=w.value(r*in+j)*x[j];y[r]=s;}gt.join();return gpu_ok.load();}
void Engine::clear()noexcept{if(impl_)impl_->clear();}
} // namespace laic::gpu
