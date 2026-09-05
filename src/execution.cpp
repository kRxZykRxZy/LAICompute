#include "laic/execution.hpp"
#include <stdexcept>
namespace laic {
const char* execution_mode_name(ExecutionMode m) noexcept { switch(m){case ExecutionMode::CPU:return "CPU";case ExecutionMode::GPU:return "GPU";case ExecutionMode::BOTH:return "BOTH";default:return "AUTO";} }
ExecutionMode execution_mode_from_string(const std::string&s) noexcept { if(s=="CPU"||s=="cpu")return ExecutionMode::CPU; if(s=="GPU"||s=="gpu")return ExecutionMode::GPU; if(s=="BOTH"||s=="both")return ExecutionMode::BOTH; return ExecutionMode::AUTO; }
ExecutionRuntime::ExecutionRuntime()=default;
ExecutionRuntime::~ExecutionRuntime()=default;
void ExecutionRuntime::load(const std::string&path){ cpu_=std::make_unique<LlamaRuntime>();cpu_->load(path); gpu_=std::make_unique<qpu::QpuLlamaRuntime>();try{gpu_->load(path);}catch(...){gpu_.reset();} }
std::vector<uint32_t> ExecutionRuntime::generate_ids(const std::string&prompt,const GenerationConfig&cfg,const LlamaRuntime::TokenCallback&cb){
    bool use_gpu=(mode_!=ExecutionMode::CPU)&&gpu_&&gpu_->gpu_ready();
    if(mode_==ExecutionMode::GPU&&!use_gpu) throw std::runtime_error("GPU mode requested but VideoCore/VC4CL is unavailable");
    if(use_gpu) return gpu_->generate_ids(prompt,cfg,cb);
    if(!cpu_)throw std::runtime_error("no model loaded"); return cpu_->generate_ids(prompt,cfg,cb);
}
void ExecutionRuntime::request_stop() noexcept { if(cpu_)cpu_->request_stop();if(gpu_)gpu_->request_stop(); }
bool ExecutionRuntime::stop_requested() const noexcept { return (cpu_&&cpu_->stop_requested())||(gpu_&&gpu_->stop_requested()); }
bool ExecutionRuntime::gpu_available() const noexcept{return gpu_&&gpu_->gpu_ready();}
std::string ExecutionRuntime::gpu_device() const{return gpu_?gpu_->device_name():std::string{};}
size_t ExecutionRuntime::qpu_matvecs() const noexcept{return gpu_?gpu_->qpu_matvecs():0;}
size_t ExecutionRuntime::cpu_matvecs() const noexcept{return gpu_?gpu_->cpu_matvecs():0;}
const Gpt2Tokenizer& ExecutionRuntime::tokenizer() const {if(gpu_&&gpu_->gpu_ready()&&mode_!=ExecutionMode::CPU)return gpu_->tokenizer();if(cpu_)return cpu_->tokenizer();throw std::runtime_error("no model loaded");}
} // namespace laic
