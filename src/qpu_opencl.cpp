#include "laic/qpu_opencl.hpp"

#include <dlfcn.h>
#include <cstring>
#include <sstream>
#include <vector>

namespace laic::qpu {

using cl_int = int;
using cl_uint = unsigned int;
using cl_ulong = unsigned long;
using cl_size_t = std::size_t;
using cl_bool = cl_uint;
using cl_device_type = cl_ulong;
using cl_mem_flags = cl_ulong;
using cl_command_queue_properties = cl_ulong;
using cl_context = void*;
using cl_command_queue = void*;
using cl_program = void*;
using cl_kernel = void*;
using cl_mem = void*;
using cl_platform_id = void*;
using cl_device_id = void*;

constexpr cl_int CL_SUCCESS = 0;
constexpr cl_device_type CL_DEVICE_TYPE_ALL = static_cast<cl_device_type>(~cl_device_type(0));
constexpr cl_device_type CL_DEVICE_TYPE_GPU = static_cast<cl_device_type>(1ull << 2);
constexpr cl_mem_flags CL_MEM_READ_ONLY = static_cast<cl_mem_flags>(1ull << 2);
constexpr cl_mem_flags CL_MEM_WRITE_ONLY = static_cast<cl_mem_flags>(1ull << 1);
constexpr cl_mem_flags CL_MEM_READ_WRITE = static_cast<cl_mem_flags>(1ull << 0);
constexpr cl_uint CL_PLATFORM_VENDOR = 0x0903;
constexpr cl_uint CL_DEVICE_NAME = 0x102B;
constexpr cl_uint CL_DEVICE_VENDOR = 0x102C;
constexpr cl_uint CL_DEVICE_TYPE = 0x1000;
constexpr cl_uint CL_TRUE = 1;
constexpr cl_uint CL_FALSE = 0;

struct OpenCLBackend::Api {
    cl_int (*GetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*) = nullptr;
    cl_int (*GetPlatformInfo)(cl_platform_id, cl_uint, cl_size_t, void*, cl_size_t*) = nullptr;
    cl_int (*GetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*) = nullptr;
    cl_int (*GetDeviceInfo)(cl_device_id, cl_uint, cl_size_t, void*, cl_size_t*) = nullptr;
    cl_context (*CreateContext)(const void*, cl_uint, const cl_device_id*, void*, void*, cl_int*) = nullptr;
    cl_command_queue (*CreateCommandQueue)(cl_context, cl_device_id, cl_command_queue_properties, cl_int*) = nullptr;
    cl_program (*CreateProgramWithSource)(cl_context, cl_uint, const char**, const cl_size_t*, cl_int*) = nullptr;
    cl_int (*BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void*, void*) = nullptr;
    cl_kernel (*CreateKernel)(cl_program, const char*, cl_int*) = nullptr;
    cl_mem (*CreateBuffer)(cl_context, cl_mem_flags, cl_size_t, void*, cl_int*) = nullptr;
    cl_int (*EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, cl_size_t, cl_size_t, const void*, cl_uint, const void*, void*) = nullptr;
    cl_int (*SetKernelArg)(cl_kernel, cl_uint, cl_size_t, const void*) = nullptr;
    cl_int (*EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const std::size_t*, const std::size_t*, const std::size_t*, cl_uint, const void*, void*) = nullptr;
    cl_int (*EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, cl_size_t, cl_size_t, void*, cl_uint, const void*, void*) = nullptr;
    cl_int (*Finish)(cl_command_queue) = nullptr;
    cl_int (*ReleaseMemObject)(cl_mem) = nullptr;
    cl_int (*ReleaseKernel)(cl_kernel) = nullptr;
    cl_int (*ReleaseProgram)(cl_program) = nullptr;
    cl_int (*ReleaseCommandQueue)(cl_command_queue) = nullptr;
    cl_int (*ReleaseContext)(cl_context) = nullptr;
};

namespace {

static const char* kSource = R"CLC(
inline float half_to_float(ushort h) {
    uint s = ((uint)(h & 0x8000)) << 16;
    uint e = (h >> 10) & 31;
    uint f = h & 1023;
    uint bits;
    if (e == 0) {
        if (f == 0) bits = s;
        else {
            int exp = -14;
            uint mant = f;
            while ((mant & 1024u) == 0u) { mant <<= 1; --exp; }
            mant &= 1023u;
            bits = s | ((uint)(exp + 127) << 23) | (mant << 13);
        }
    } else if (e == 31) bits = s | 0x7f800000u | (f << 13);
    else bits = s | ((e + 112u) << 23) | (f << 13);
    return as_float(bits);
}

__kernel void laic_matvec_f32(__global const float* w, __global const float* x,
                               __global float* y, uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    float16 acc = (float16)(0.0f);
    uint c = 0;
    for (; c + 16 <= cols; c += 16) {
        float16 xv = vload16(0, x + c);
        float16 wv = vload16(0, w + r * cols + c);
        acc += xv * wv;
    }
    float sum = 0.0f;
    sum += acc.s0 + acc.s1 + acc.s2 + acc.s3;
    sum += acc.s4 + acc.s5 + acc.s6 + acc.s7;
    sum += acc.s8 + acc.s9 + acc.sa + acc.sb;
    sum += acc.sc + acc.sd + acc.se + acc.sf;
    for (; c < cols; ++c) sum += w[r * cols + c] * x[c];
    y[r] = sum;
}

__kernel void laic_matvec_f16(__global const ushort* w, __global const float* x,
                               __global float* y, uint rows, uint cols) {
    uint r = get_global_id(0);
    if (r >= rows) return;
    float16 acc = (float16)(0.0f);
    uint c = 0;
    for (; c + 16 <= cols; c += 16) {
        float16 xv = vload16(0, x + c);
        float16 wv = (float16)(
            half_to_float(w[r * cols + c]), half_to_float(w[r * cols + c + 1]),
            half_to_float(w[r * cols + c + 2]), half_to_float(w[r * cols + c + 3]),
            half_to_float(w[r * cols + c + 4]), half_to_float(w[r * cols + c + 5]),
            half_to_float(w[r * cols + c + 6]), half_to_float(w[r * cols + c + 7]),
            half_to_float(w[r * cols + c + 8]), half_to_float(w[r * cols + c + 9]),
            half_to_float(w[r * cols + c + 10]), half_to_float(w[r * cols + c + 11]),
            half_to_float(w[r * cols + c + 12]), half_to_float(w[r * cols + c + 13]),
            half_to_float(w[r * cols + c + 14]), half_to_float(w[r * cols + c + 15]));
        acc += xv * wv;
    }
    float sum = 0.0f;
    sum += acc.s0 + acc.s1 + acc.s2 + acc.s3;
    sum += acc.s4 + acc.s5 + acc.s6 + acc.s7;
    sum += acc.s8 + acc.s9 + acc.sa + acc.sb;
    sum += acc.sc + acc.sd + acc.se + acc.sf;
    for (; c < cols; ++c) sum += half_to_float(w[r * cols + c]) * x[c];
    y[r] = sum;
}
)CLC";

static void seterr(std::string* e, const char* s) { if (e) *e = s; }

} // namespace

OpenCLBackend::OpenCLBackend() = default;
OpenCLBackend::~OpenCLBackend() { shutdown(); }

bool OpenCLBackend::initialize(std::string* error) {
    shutdown();
    const char* names[] = {"libOpenCL.so", "libOpenCL.so.1"};
    for (const char* n : names) {
        library_ = dlopen(n, RTLD_NOW | RTLD_LOCAL);
        if (library_) break;
    }
    if (!library_) { seterr(error, "libOpenCL not found"); return false; }

    api_ = new Api();
    Api& a = *api_;

    // Keep the loader handle explicit here. The old macro lived in the
    // anonymous namespace and could not see initialize()'s local `lib`.
#define LOAD(api, handle, name) \
    do { *(void**)(&((api).name)) = dlsym((handle), "cl" #name); \
         if (!(api).name) { seterr(error, "missing OpenCL symbol: cl" #name); shutdown(); return false; } } while (0)

    LOAD(a, library_, GetPlatformIDs);
    LOAD(a, library_, GetPlatformInfo);
    LOAD(a, library_, GetDeviceIDs);
    LOAD(a, library_, GetDeviceInfo);
    LOAD(a, library_, CreateContext);
    LOAD(a, library_, CreateCommandQueue);
    LOAD(a, library_, CreateProgramWithSource);
    LOAD(a, library_, BuildProgram);
    LOAD(a, library_, CreateKernel);
    LOAD(a, library_, CreateBuffer);
    LOAD(a, library_, EnqueueWriteBuffer);
    LOAD(a, library_, SetKernelArg);
    LOAD(a, library_, EnqueueNDRangeKernel);
    LOAD(a, library_, EnqueueReadBuffer);
    LOAD(a, library_, Finish);
    LOAD(a, library_, ReleaseMemObject);
    LOAD(a, library_, ReleaseKernel);
    LOAD(a, library_, ReleaseProgram);
    LOAD(a, library_, ReleaseCommandQueue);
    LOAD(a, library_, ReleaseContext);

#undef LOAD

    cl_uint np = 0;
    if (a.GetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || !np) {
        seterr(error, "no OpenCL platforms");
        shutdown();
        return false;
    }

    std::vector<cl_platform_id> ps(np);
    a.GetPlatformIDs(np, ps.data(), nullptr);
    cl_platform_id chosen = nullptr;
    cl_device_id device = nullptr;
    std::size_t sz = 0;

    for (auto p : ps) {
        cl_uint nd = 0;
        if (a.GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nd) != CL_SUCCESS || !nd) continue;
        std::vector<cl_device_id> ds(nd);
        a.GetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
        char vendor[256] = {};
        sz = 0;
        a.GetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor)-1, vendor, &sz);
        std::string v(vendor);
        for (auto d : ds) {
            char name[256] = {};
            sz = 0;
            a.GetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name)-1, name, &sz);
            std::string dn(name);
            if (v.find("VC4CL") != std::string::npos ||
                dn.find("VideoCore") != std::string::npos ||
                dn.find("VC4") != std::string::npos) {
                chosen = p;
                device = d;
                break;
            }
        }
        if (device) break;
    }

    (void)chosen;
    if (!device) {
        seterr(error, "no VC4CL VideoCore GPU device found");
        shutdown();
        return false;
    }

    cl_int rc = CL_SUCCESS;
    context_ = a.CreateContext(nullptr, 1, &device, nullptr, nullptr, &rc);
    if (rc != CL_SUCCESS || !context_) { seterr(error, "clCreateContext failed"); shutdown(); return false; }
    queue_ = a.CreateCommandQueue(static_cast<cl_context>(context_), device, 0, &rc);
    if (rc != CL_SUCCESS || !queue_) { seterr(error, "clCreateCommandQueue failed"); shutdown(); return false; }
    const char* src = kSource;
    program_ = a.CreateProgramWithSource(static_cast<cl_context>(context_), 1, &src, nullptr, &rc);
    if (rc != CL_SUCCESS || !program_) { seterr(error, "clCreateProgramWithSource failed"); shutdown(); return false; }
    rc = a.BuildProgram(static_cast<cl_program>(program_), 1, &device, "-cl-std=CL1.2", nullptr, nullptr);
    if (rc != CL_SUCCESS) { seterr(error, "VC4CL failed to build QPU kernels"); shutdown(); return false; }
    kernel_f32_ = a.CreateKernel(static_cast<cl_program>(program_), "laic_matvec_f32", &rc);
    kernel_f16_ = a.CreateKernel(static_cast<cl_program>(program_), "laic_matvec_f16", &rc);
    if (rc != CL_SUCCESS || !kernel_f32_ || !kernel_f16_) { seterr(error, "failed to create QPU kernels"); shutdown(); return false; }
    char name[256] = {};
    sz = 0;
    a.GetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name)-1, name, &sz);
    device_name_ = name;
    ready_ = true;
    return true;
}

void OpenCLBackend::shutdown() noexcept {
    if (api_) {
        Api& a = *api_;
        if (weights_buf_) a.ReleaseMemObject(static_cast<cl_mem>(weights_buf_));
        if (input_buf_) a.ReleaseMemObject(static_cast<cl_mem>(input_buf_));
        if (output_buf_) a.ReleaseMemObject(static_cast<cl_mem>(output_buf_));
        if (kernel_f32_) a.ReleaseKernel(static_cast<cl_kernel>(kernel_f32_));
        if (kernel_f16_) a.ReleaseKernel(static_cast<cl_kernel>(kernel_f16_));
        if (program_) a.ReleaseProgram(static_cast<cl_program>(program_));
        if (queue_) a.ReleaseCommandQueue(static_cast<cl_command_queue>(queue_));
        if (context_) a.ReleaseContext(static_cast<cl_context>(context_));
    }
    weights_buf_=input_buf_=output_buf_=kernel_f32_=kernel_f16_=program_=queue_=context_=nullptr;
    if (library_) dlclose(library_);
    library_=nullptr;
    delete api_;
    api_=nullptr;
    ready_=false;
    weights_capacity_=input_capacity_=output_capacity_=0;
    device_name_.clear();
}

static bool run(OpenCLBackend* self, void* kernel, const void* weights, std::size_t wb,
                const float* x, std::size_t xb, float* y, std::size_t yb,
                std::size_t rows, std::size_t cols, std::string* error) {
    if (!self || !self->available()) { seterr(error, "QPU backend unavailable"); return false; }
    auto& a = *self->api_;
    cl_int rc = CL_SUCCESS;
    auto ensure = [&](void*& b, std::size_t& cap, cl_mem_flags flags, std::size_t need) {
        if (cap >= need && b) return true;
        if (b) a.ReleaseMemObject(static_cast<cl_mem>(b));
        b = a.CreateBuffer(static_cast<cl_context>(self->context_), flags, need, nullptr, &rc);
        if (rc != CL_SUCCESS || !b) return false;
        cap = need;
        return true;
    };
    if (!ensure(self->weights_buf_, self->weights_capacity_, CL_MEM_READ_ONLY, wb) ||
        !ensure(self->input_buf_, self->input_capacity_, CL_MEM_READ_ONLY, xb) ||
        !ensure(self->output_buf_, self->output_capacity_, CL_MEM_WRITE_ONLY, yb)) {
        seterr(error,"QPU buffer allocation failed");
        return false;
    }
    if (a.EnqueueWriteBuffer(static_cast<cl_command_queue>(self->queue_), static_cast<cl_mem>(self->weights_buf_), CL_TRUE, 0, wb, weights, 0,nullptr,nullptr) != CL_SUCCESS ||
        a.EnqueueWriteBuffer(static_cast<cl_command_queue>(self->queue_), static_cast<cl_mem>(self->input_buf_), CL_TRUE, 0, xb, x, 0,nullptr,nullptr) != CL_SUCCESS) {
        seterr(error,"QPU upload failed");
        return false;
    }
    if (a.SetKernelArg(static_cast<cl_kernel>(kernel), 0, sizeof(cl_mem), &self->weights_buf_) != CL_SUCCESS ||
        a.SetKernelArg(static_cast<cl_kernel>(kernel), 1, sizeof(cl_mem), &self->input_buf_) != CL_SUCCESS ||
        a.SetKernelArg(static_cast<cl_kernel>(kernel), 2, sizeof(cl_mem), &self->output_buf_) != CL_SUCCESS) {
        seterr(error,"QPU kernel argument setup failed");
        return false;
    }
    cl_uint ur = static_cast<cl_uint>(rows);
    cl_uint uc = static_cast<cl_uint>(cols);
    if (a.SetKernelArg(static_cast<cl_kernel>(kernel), 3, sizeof(ur), &ur) != CL_SUCCESS ||
        a.SetKernelArg(static_cast<cl_kernel>(kernel), 4, sizeof(uc), &uc) != CL_SUCCESS) {
        seterr(error,"QPU kernel dimension setup failed");
        return false;
    }
    std::size_t global = rows;
    if (a.EnqueueNDRangeKernel(static_cast<cl_command_queue>(self->queue_), static_cast<cl_kernel>(kernel), 1, nullptr, &global, nullptr, 0,nullptr,nullptr) != CL_SUCCESS ||
        a.Finish(static_cast<cl_command_queue>(self->queue_)) != CL_SUCCESS ||
        a.EnqueueReadBuffer(static_cast<cl_command_queue>(self->queue_), static_cast<cl_mem>(self->output_buf_), CL_TRUE, 0, yb, y, 0,nullptr,nullptr) != CL_SUCCESS) {
        seterr(error,"QPU execution failed");
        return false;
    }
    return true;
}

bool OpenCLBackend::matvec_f32(const float* weights, const float* x, float* y,
                               std::size_t rows, std::size_t cols, std::string* error) {
    if (!weights || !x || !y || rows == 0 || cols == 0) { seterr(error, "invalid F32 matvec arguments"); return false; }
    return run(this, kernel_f32_, weights, rows * cols * sizeof(float), x, cols * sizeof(float), y, rows * sizeof(float), rows, cols, error);
}

bool OpenCLBackend::matvec_f16(const std::uint16_t* weights, const float* x, float* y,
                               std::size_t rows, std::size_t cols, std::string* error) {
    if (!weights || !x || !y || rows == 0 || cols == 0) { seterr(error, "invalid F16 matvec arguments"); return false; }
    return run(this, kernel_f16_, weights, rows * cols * sizeof(std::uint16_t), x, cols * sizeof(float), y, rows * sizeof(float), rows, cols, error);
}

} // namespace laic::qpu
