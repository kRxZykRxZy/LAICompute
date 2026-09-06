#include "laic/server.hpp"
#include "laic/cpu.hpp"
#include "laic/inference.hpp"
#include "laic/videocore.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

namespace laic { namespace {
std::string jsonq(const std::string&s){std::string o="\"";for(unsigned char c:s){if(c=='"'||c=='\\')o+='\\',o+=char(c);else if(c=='\n')o+="\\n";else if(c=='\r')o+="\\r";else if(c=='\t')o+="\\t";else if(c<0x20){char h[7];std::snprintf(h,sizeof(h),"\\u%04x",c);o+=h;}else o+=char(c);}return o+'"';}
std::string js(const std::string&b,const std::string&k){auto p=b.find("\""+k+"\"");if(p==std::string::npos)return{};p=b.find(':',p);p=b.find('"',p);if(p==std::string::npos)return{};std::string o;for(++p;p<b.size()&&b[p]!='"';++p){if(b[p]=='\\'&&p+1<b.size())++p;o+=b[p];}return o;}
size_t jn(const std::string&b,const std::string&k,size_t d){auto p=b.find("\""+k+"\"");if(p==std::string::npos)return d;p=b.find(':',p);try{return std::stoull(b.substr(p+1));}catch(...){return d;}}
std::string reply(int code,const std::string&data,const std::string&type="application/json"){const char*msg=code==200?"OK":code==201?"Created":code==400?"Bad Request":code==404?"Not Found":code==409?"Conflict":"Internal Server Error";return "HTTP/1.1 "+std::to_string(code)+" "+msg+"\r\nContent-Type: "+type+"\r\nContent-Length: "+std::to_string(data.size())+"\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type,X-Filename\r\nAccess-Control-Allow-Methods: GET,POST,OPTIONS\r\nConnection: close\r\n\r\n"+data;}
void writeall(int fd,const std::string&s){for(size_t p=0;p<s.size();){ssize_t n=send(fd,s.data()+p,s.size()-p,MSG_NOSIGNAL);if(n<=0)return;p+=size_t(n);}}
std::string header(const std::string&r,const std::string&n){auto p=r.find(n);if(p==std::string::npos)return{};p+=n.size();auto e=r.find("\r\n",p);return r.substr(p,e-p);}
std::string backend_panel(const videocore::DeviceInfo&d){
    std::string g=d.present?d.name+" detected":"No VideoCore compute device detected";
    std::string peak=d.present?std::to_string(d.theoretical_gflops).substr(0,std::to_string(d.theoretical_gflops).find('.')+3):"0";
    return std::string("<div id=\"computeMode\" style=\"border-top:1px solid var(--line);padding-top:12px\"><div class=\"section-title\"><span>Inference engine</span></div><div style=\"display:grid;gap:7px;margin-top:8px\"><select id=\"backendSelect\" style=\"width:100%;padding:10px;border-radius:12px;background:#151519;color:var(--text);border:1px solid var(--line)\"><option value=\"cpu\">CPU</option><option value=\"gpu\"")+(d.compute_available?"":" disabled")+">GPU / QPU</option><option value=\"both\""+(d.compute_available?"":" disabled")+">CPU + GPU</option></select><div id=\"gpuInfo\" style=\"font-size:10px;color:var(--muted);line-height:1.4\">"+g+(d.present?" · theoretical "+peak+" GFLOP/s":"")+"</div></div><div style=\"margin-top:10px;padding:10px;border:1px solid var(--line);border-radius:13px;background:#111114\"><div style=\"font-weight:800;font-size:11px;margin-bottom:7px\">GPU performance</div><div id=\"gpuStats\" style=\"font-size:10px;color:var(--muted);line-height:1.55\">Waiting for inference…</div><button id=\"gpuBench\" style=\"width:100%;margin-top:8px;padding:8px;border-radius:9px;background:#eee;color:#111;font-size:10px;font-weight:800\">Run GPU benchmark</button></div></div><script>(function(){const s=document.getElementById('backendSelect'),st=document.getElementById('gpuStats'),bi=document.getElementById('gpuInfo'),bb=document.getElementById('gpuBench');if(!s)return;const saved=localStorage.getItem('laic-backend');if(saved&&[...s.options].some(o=>o.value===saved&&!o.disabled))s.value=saved;async function choose(){localStorage.setItem('laic-backend',s.value);try{await fetch('/api/backend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({backend:s.value})})}catch(e){}}s.addEventListener('change',choose);window.LAIC_BACKEND=()=>s.value;async function stats(){try{const r=await fetch('/api/gpu/stats'),d=await r.json();if(d.available){const util=d.theoretical_gflops>0?Math.min(100,d.gflops/d.theoretical_gflops*100):0;st.innerHTML='Kernel: '+d.kernel_ms.toFixed(2)+' ms · '+d.gflops.toFixed(2)+' GFLOP/s<br>Peak utilization: '+util.toFixed(1)+'% · GPU calls: '+d.gpu_calls+' · CPU fallbacks: '+d.fallbacks+'<br>Workgroup: '+d.workgroup_size+' / '+d.max_workgroup_size+' · compute units: '+d.compute_units+' · clock: '+d.clock_mhz+' MHz';bi.textContent=d.name+' · '+d.runtime+' · '+d.qpus+' QPUs · theoretical '+d.theoretical_gflops.toFixed(2)+' GFLOP/s'}else st.textContent=d.reason||'GPU runtime unavailable'}catch(e){}}async function bench(){if(bb.disabled)return;bb.disabled=true;bb.textContent='Benchmarking…';try{const r=await fetch('/api/benchmark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({backend:s.value,max_tokens:32})}),d=await r.json();if(!r.ok)throw Error(d.error||'Benchmark failed');st.innerHTML='Benchmark: <b>'+d.tokens_per_sec.toFixed(2)+' tok/s</b> · '+d.elapsed_ms.toFixed(0)+' ms · '+d.tokens+' tokens';if(d.gpu_gflops!==undefined)st.innerHTML+=' · '+d.gpu_gflops.toFixed(2)+' GFLOP/s';}catch(e){st.textContent=e.message}finally{bb.disabled=false;bb.textContent='Run GPU benchmark'}}bb.addEventListener('click',bench);stats();setInterval(stats,1000)})();</script>";
}
}

struct ApiServer::Impl{
    uint16_t port;std::string dir;int fd=-1;std::atomic<bool>running{false};std::mutex mu;std::unique_ptr<LlamaRuntime>rt;std::string loaded;std::atomic<bool>generating{false};std::atomic<size_t>tokens{0};std::atomic<double>tps{0};std::atomic<double>cpu_percent{0};std::chrono::steady_clock::time_point started;uint64_t prev_cpu_total=0,prev_cpu_idle=0;std::mutex cpu_mu;videocore::DeviceInfo gpu=videocore::detect();videocore::Backend selected_backend=videocore::Backend::CPU;
    Impl(uint16_t p,std::string d):port(p),dir(std::move(d)){std::filesystem::create_directories(dir);}
    ~Impl(){stop();}
    double cpu_usage(){std::ifstream f("/proc/stat");std::string label;uint64_t user=0,nice=0,system=0,idle=0,iowait=0,irq=0,softirq=0,steal=0;if(!(f>>label>>user>>nice>>system>>idle>>iowait>>irq>>softirq>>steal))return cpu_percent.load();uint64_t total=user+nice+system+idle+iowait+irq+softirq+steal,idle_all=idle+iowait;std::lock_guard<std::mutex>g(cpu_mu);if(prev_cpu_total){uint64_t dt=total-prev_cpu_total,di=idle_all-prev_cpu_idle;if(dt)cpu_percent=100.0*double(dt-di)/double(dt);}prev_cpu_total=total;prev_cpu_idle=idle_all;return cpu_percent.load();}
    std::string models(){std::string o="[";bool first=true;for(auto&e:std::filesystem::directory_iterator(dir)){if(!e.is_regular_file()||e.path().extension()!=".gguf")continue;if(!first)o+=',';first=false;o+="{\"name\":"+jsonq(e.path().filename().string())+",\"size\":"+std::to_string(std::filesystem::file_size(e.path()))+",\"loaded\":"+(e.path().filename().string()==loaded?"true":"false")+"}";}return o+"]";}
    std::string status(){std::ifstream f("/proc/meminfo");std::string k;size_t v,total=0,avail=0;while(f>>k>>v){if(k=="MemTotal:")total=v*1024;else if(k=="MemAvailable:"){avail=v*1024;break;}}auto c=CpuInfo::detect();size_t ram=total?100-(avail*100/total):0;double cpu=cpu_usage();return "{\"generating\":"+std::string(generating?"true":"false")+",\"tokens\":"+std::to_string(tokens.load())+",\"tokens_per_sec\":"+std::to_string(tps.load())+",\"cpu_percent\":"+std::to_string(cpu)+",\"ram_percent\":"+std::to_string(ram)+",\"cpu_logical\":"+std::to_string(c.logical_cpus)+",\"cpu_physical\":"+std::to_string(c.physical_cores)+",\"inference_threads\":"+std::to_string(std::thread::hardware_concurrency())+",\"loaded\":"+jsonq(loaded)+",\"backend\":"+jsonq(videocore::backend_name(selected_backend))+"}";}
