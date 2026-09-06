#include "laic/inference.hpp"
#include "laic/videocore.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
static double ms_since(const std::chrono::steady_clock::time_point&a,const std::chrono::steady_clock::time_point&b){return std::chrono::duration<double,std::milli>(b-a).count();}
int main(int argc,char**argv){
 if(argc<2){std::fprintf(stderr,"usage: %s MODEL.gguf [cpu|gpu|both] [max_tokens] [prompt]\n",argv[0]);return 2;}
 const std::string model=argv[1],backend_arg=argc>2?argv[2]:"gpu";const size_t max_tokens=argc>3?std::strtoull(argv[3],nullptr,10):32;const std::string prompt=argc>4?argv[4]:"Explain how a Raspberry Pi works in simple terms.";const auto backend=laic::videocore::backend_from_string(backend_arg);auto dev=laic::videocore::detect();
 std::printf("LAICompute end-to-end benchmark\nmodel=%s\nbackend=%s\nGPU=%s generation=%s runtime=%s QPUs=%u clock=%uMHz theoretical=%.2f GFLOPS\n\n",model.c_str(),laic::videocore::backend_name(backend),dev.name.c_str(),laic::videocore::generation_name(dev.generation).c_str(),dev.runtime.c_str(),dev.qpus,dev.clock_mhz,dev.theoretical_gflops);
 laic::LlamaRuntime rt;try{rt.load(model);}catch(const std::exception&e){std::fprintf(stderr,"load failed: %s\n",e.what());return 1;}rt.set_backend(backend);laic::GenerationConfig cfg;cfg.max_tokens=max_tokens;cfg.temperature=0.0f;
 try{rt.generate_ids(prompt,cfg);}catch(const std::exception&e){std::fprintf(stderr,"warmup failed: %s\n",e.what());return 1;}
 const int runs=3;double total_ms=0;size_t total_tokens=0;for(int r=0;r<runs;r++){auto t0=std::chrono::steady_clock::now();std::vector<uint32_t>ids;try{ids=rt.generate_ids(prompt,cfg);}catch(const std::exception&e){std::fprintf(stderr,"run %d failed: %s\n",r+1,e.what());return 1;}auto t1=std::chrono::steady_clock::now();double ms=ms_since(t0,t1);total_ms+=ms;total_tokens+=ids.size();std::printf("run %d: %.3f ms, %zu generated tokens, %.3f tok/s\n",r+1,ms,ids.size(),ids.empty()?0.0:1000.0*ids.size()/ms);}
 const double avg_ms=total_ms/runs,avg_tokens=double(total_tokens)/runs,tps=avg_tokens/(avg_ms/1000.0);std::printf("\nE2E average: %.3f ms, %.2f tokens, %.3f tok/s\n",avg_ms,avg_tokens,tps);
 if(const auto*s=rt.gpu_stats();s&&rt.gpu_available()){double gpu_ms=double(s->gpu_ns)/1e6;double gf=s->gpu_ns?double(s->gpu_flops)/double(s->gpu_ns):0.0;double util=dev.theoretical_gflops>0?100.0*gf/dev.theoretical_gflops:0.0;std::printf("GPU kernels: %.3f ms, %.3f GFLOPS achieved, %.2f%% of theoretical peak\n",gpu_ms,gf,util);std::printf("GPU matvecs: %llu/%llu successful, fallbacks=%llu\n",(unsigned long long)s->matvec_gpu_calls,(unsigned long long)s->matvec_calls,(unsigned long long)s->matvec_fallbacks);std::printf("workgroup: selected=%zu max=%zu preferred-multiple=%zu compute-units=%zu clock=%zuMHz\n",s->selected_work_group_size,s->max_work_group_size,s->preferred_work_group_multiple,s->compute_units,s->clock_mhz);std::printf("GPU traffic: read=%llu bytes write=%llu bytes\n",(unsigned long long)s->bytes_read,(unsigned long long)s->bytes_written);}else std::printf("GPU runtime: unavailable; CPU path/fallback was used.\n");
 std::printf("Theoretical peak is a hardware ceiling; E2E tok/s is the meaningful model benchmark.\n");return 0;
}
