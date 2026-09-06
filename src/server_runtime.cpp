#include "laic/server.hpp"
#include "laic/cpu.hpp"
#include "laic/inference.hpp"
#include "laic/videocore.hpp"
#include "laic/videocore_runtime.hpp"
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

namespace laic {
namespace {

std::string jsonq(const std::string& s) { std::string o="\""; for(unsigned char c:s){ if(c=='"'||c=='\\')o+='\\',o+=char(c); else if(c=='\n')o+="\\n"; else if(c=='\r')o+="\\r"; else if(c=='\t')o+="\\t"; else if(c<0x20){char h[7];std::snprintf(h,sizeof(h),"\\u%04x",c);o+=h;} else o+=char(c);} return o+'"'; }
std::string js(const std::string& b,const std::string& k){auto p=b.find("\""+k+"\"");if(p==std::string::npos)return{};p=b.find(':',p);p=b.find('"',p);if(p==std::string::npos)return{};std::string o;for(++p;p<b.size()&&b[p]!='"';++p){if(b[p]=='\\'&&p+1<b.size())++p;o+=b[p];}return o;}
size_t jn(const std::string& b,const std::string& k,size_t d){auto p=b.find("\""+k+"\"");if(p==std::string::npos)return d;p=b.find(':',p);try{return std::stoull(b.substr(p+1));}catch(...){return d;}}
std::string reply(int code,const std::string& data,const std::string& type="application/json"){const char* msg=code==200?"OK":code==201?"Created":code==400?"Bad Request":code==404?"Not Found":code==409?"Conflict":"Internal Server Error";return "HTTP/1.1 "+std::to_string(code)+" "+msg+"\r\nContent-Type: "+type+"\r\nContent-Length: "+std::to_string(data.size())+"\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type,X-Filename\r\nAccess-Control-Allow-Methods: GET,POST,OPTIONS\r\nConnection: close\r\n\r\n"+data;}
void writeall(int fd,const std::string& s){for(size_t p=0;p<s.size();){ssize_t n=send(fd,s.data()+p,s.size()-p,MSG_NOSIGNAL);if(n<=0)return;p+=size_t(n);}}
std::string header(const std::string& r,const std::string& n){auto p=r.find(n);if(p==std::string::npos)return{};p+=n.size();auto e=r.find("\r\n",p);return r.substr(p,e-p);}

std::string backend_panel(const videocore::DeviceInfo& d){
    std::string g=d.present?d.name+" detected":"No VideoCore compute device detected";
    std::string peak=d.present?std::to_string(d.theoretical_gflops).substr(0,std::to_string(d.theoretical_gflops).find('.')+3):"0";
    std::string disabled=d.compute_available?"":" disabled";
    std::string why=d.present&&!d.compute_available?" (compute runtime unavailable — see GPU performance panel)":"";
    std::string html;
    html += "<div id=\"computeMode\" style=\"border-top:1px solid var(--line);padding-top:12px\"><div class=\"section-title\"><span>Inference engine</span></div><div style=\"display:grid;gap:7px;margin-top:8px\"><select id=\"backendSelect\" style=\"width:100%;padding:10px;border-radius:12px;background:#151519;color:var(--text);border:1px solid var(--line)\"><option value=\"cpu\">CPU</option>";
    html += "<option value=\"gpu\"" + disabled + ">GPU / QPU</option><option value=\"both\"" + disabled + ">CPU + GPU</option></select>";
    html += "<div id=\"gpuInfo\" style=\"font-size:10px;color:var(--muted);line-height:1.4\">" + g + why + (d.present?" · theoretical "+peak+" GFLOP/s":"") + "</div></div>";
    html += "<div style=\"margin-top:10px;padding:10px;border:1px solid var(--line);border-radius:13px;background:#111114\"><div style=\"font-weight:800;font-size:11px;margin-bottom:7px\">GPU performance</div><div id=\"gpuStats\" style=\"font-size:10px;color:var(--muted);line-height:1.55\">Waiting for inference…</div><button id=\"gpuBench\" style=\"width:100%;margin-top:8px;padding:8px;border-radius:9px;background:#eee;color:#111;font-size:10px;font-weight:800\">Run GPU benchmark</button></div></div>";
    html += "<script>(function(){const s=document.getElementById('backendSelect'),st=document.getElementById('gpuStats'),bi=document.getElementById('gpuInfo'),bb=document.getElementById('gpuBench');if(!s)return;const saved=localStorage.getItem('laic-backend');if(saved&&[...s.options].some(o=>o.value===saved&&!o.disabled))s.value=saved;async function choose(){localStorage.setItem('laic-backend',s.value);try{await fetch('/api/backend',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({backend:s.value})})}catch(e){}}s.addEventListener('change',choose);window.LAIC_BACKEND=()=>s.value;async function detectInfo(){try{const r=await fetch('/api/gpu/detect'),d=await r.json();let msg='';if(d.present)msg=d.runtime_name+' · '+d.qpus+' QPUs · '+d.clock_mhz+' MHz · '+d.runtime_string;if(!d.runtime_available)msg+='<br>'+d.runtime_detail+(d.all_devices&&d.all_devices.length?'<br>Devices: '+d.all_devices.join(', '):'');return msg}catch(e){return null}}async function stats(){try{const r=await fetch('/api/gpu/stats'),d=await r.json();if(d.available){const util=d.theoretical_gflops>0?Math.min(100,d.gflops/d.theoretical_gflops*100):0;st.innerHTML='Kernel: '+d.kernel_ms.toFixed(2)+' ms · '+d.gflops.toFixed(2)+' GFLOP/s<br>Peak utilization: '+util.toFixed(1)+'% · GPU calls: '+d.gpu_calls+' · CPU fallbacks: '+d.fallbacks+'<br>Workgroup: '+d.workgroup_size+' / '+d.max_workgroup_size+' · compute units: '+d.compute_units+' · clock: '+d.clock_mhz+' MHz';bi.textContent=d.name+' · '+d.runtime+' · '+d.qpus+' QPUs · theoretical '+d.theoretical_gflops.toFixed(2)+' GFLOP/s'}else{const m=await detectInfo();st.textContent=d.reason||'GPU runtime unavailable';if(m)st.innerHTML+='<br>'+m}}catch(e){}}async function bench(){if(bb.disabled)return;bb.disabled=true;bb.textContent='Benchmarking…';try{const r=await fetch('/api/benchmark',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({backend:s.value,max_tokens:32})}),d=await r.json();if(!r.ok)throw Error(d.error||'Benchmark failed');st.innerHTML='Benchmark: <b>'+d.tokens_per_sec.toFixed(2)+' tok/s</b> · '+d.elapsed_ms.toFixed(0)+' ms · '+d.tokens+' tokens';if(d.gpu_gflops!==undefined)st.innerHTML+=' · '+d.gpu_gflops.toFixed(2)+' GFLOP/s';}catch(e){st.textContent=e.message}finally{bb.disabled=false;bb.textContent='Run GPU benchmark'}}bb.addEventListener('click',bench);stats();setInterval(stats,1000)})();</script>";
    return html;
}

} // namespace

struct ApiServer::Impl {
    uint16_t port; std::string dir; int fd=-1; std::atomic<bool> running{false}; std::mutex mu; std::unique_ptr<LlamaRuntime> rt; std::string loaded; std::atomic<bool> generating{false}; std::atomic<size_t> tokens{0}; std::atomic<double> tps{0}; std::atomic<double> cpu_percent{0}; std::chrono::steady_clock::time_point started; std::mutex cpu_mu; uint64_t prev_cpu_total=0,prev_cpu_idle=0; videocore::DeviceInfo gpu=videocore::detect(); videocore::Backend selected_backend=videocore::Backend::CPU;
    Impl(uint16_t p,std::string d):port(p),dir(std::move(d)){std::filesystem::create_directories(dir);} ~Impl(){stop();}
    double cpu_usage(){std::ifstream f("/proc/stat");std::string label;uint64_t user=0,nice=0,system=0,idle=0,iowait=0,irq=0,softirq=0,steal=0;if(!(f>>label>>user>>nice>>system>>idle>>iowait>>irq>>softirq>>steal))return cpu_percent.load();uint64_t total=user+nice+system+idle+iowait+irq+softirq+steal,idle_all=idle+iowait;std::lock_guard<std::mutex>g(cpu_mu);if(prev_cpu_total){uint64_t dt=total-prev_cpu_total,di=idle_all-prev_cpu_idle;if(dt)cpu_percent=100.0*double(dt-di)/double(dt);}prev_cpu_total=total;prev_cpu_idle=idle_all;return cpu_percent.load();}
    std::string models(){std::string o="[";bool first=true;for(auto&e:std::filesystem::directory_iterator(dir)){if(!e.is_regular_file()||e.path().extension()!=".gguf")continue;if(!first)o+=',';first=false;o+="{\"name\":"+jsonq(e.path().filename().string())+",\"size\":"+std::to_string(std::filesystem::file_size(e.path()))+",\"loaded\":"+(e.path().filename().string()==loaded?"true":"false")+"}";}return o+"]";}
    std::string status(){std::ifstream f("/proc/meminfo");std::string k;size_t v,total=0,avail=0;while(f>>k>>v){if(k=="MemTotal:")total=v*1024;else if(k=="MemAvailable:"){avail=v*1024;break;}}auto c=CpuInfo::detect();size_t ram=total?100-(avail*100/total):0;double cpu=cpu_usage();return "{\"generating\":"+std::string(generating?"true":"false")+",\"tokens\":"+std::to_string(tokens.load())+",\"tokens_per_sec\":"+std::to_string(tps.load())+",\"cpu_percent\":"+std::to_string(cpu)+",\"ram_percent\":"+std::to_string(ram)+",\"cpu_logical\":"+std::to_string(c.logical_cpus)+",\"cpu_physical\":"+std::to_string(c.physical_cores)+",\"inference_threads\":"+std::to_string(std::thread::hardware_concurrency())+",\"loaded\":"+jsonq(loaded)+",\"backend\":"+jsonq(videocore::backend_name(selected_backend))+"}";}
    std::string backends(){return "{\"selected\":"+jsonq(videocore::backend_name(selected_backend))+",\"present\":"+(gpu.present?"true":"false")+",\"compute_available\":"+(gpu.compute_available?"true":"false")+",\"generation\":"+jsonq(videocore::generation_name(gpu.generation))+",\"name\":"+jsonq(gpu.name)+",\"runtime\":"+jsonq(gpu.runtime)+",\"qpus\":"+std::to_string(gpu.qpus)+",\"clock_mhz\":"+std::to_string(gpu.clock_mhz)+",\"theoretical_gflops\":"+std::to_string(gpu.theoretical_gflops)+"}";}
    std::string gpu_stats(){auto*d=rt?rt->gpu_stats():nullptr;bool av=d&&rt&&rt->gpu_available();if(!av)return "{\"available\":false,\"reason\":"+jsonq(gpu.compute_available?"GPU engine not active for the current model/backend":"VideoCore compute runtime unavailable")+"}";double ms=double(d->gpu_ns)/1e6;double gf=d->gpu_ns?double(d->gpu_flops)/double(d->gpu_ns):0.0;return "{\"available\":true,\"name\":"+jsonq(gpu.name)+",\"runtime\":"+jsonq(gpu.runtime)+",\"qpus\":"+std::to_string(gpu.qpus)+",\"clock_mhz\":"+std::to_string(d->clock_mhz?d->clock_mhz:gpu.clock_mhz)+",\"theoretical_gflops\":"+std::to_string(gpu.theoretical_gflops)+",\"kernel_ms\":"+std::to_string(ms)+",\"gflops\":"+std::to_string(gf)+",\"gpu_calls\":"+std::to_string(d->matvec_gpu_calls)+",\"fallbacks\":"+std::to_string(d->matvec_fallbacks)+",\"workgroup_size\":"+std::to_string(d->selected_work_group_size)+",\"max_workgroup_size\":"+std::to_string(d->max_work_group_size)+",\"preferred_multiple\":"+std::to_string(d->preferred_work_group_multiple)+",\"compute_units\":"+std::to_string(d->compute_units)+"}";}
    std::string gpu_detect(){
        auto rt=videocore::probe_runtime(gpu);
        std::string o="{";
        o+="\"present\":"+(gpu.present?"true":"false");
        o+=",\"generation\":"+jsonq(videocore::generation_name(gpu.generation));
        o+=",\"compute_available\":"+(gpu.compute_available?"true":"false");
        o+=",\"runtime_api\":"+jsonq(rt.api);
        o+=",\"runtime_device\":"+jsonq(rt.device);
        o+=",\"runtime_detail\":"+jsonq(rt.detail);
        o+=",\"runtime_available\":"+(rt.available?"true":"false");
        o+=",\"qpus\":"+std::to_string(gpu.qpus);
        o+=",\"clock_mhz\":"+std::to_string(gpu.clock_mhz);
        o+=",\"theoretical_gflops\":"+std::to_string(gpu.theoretical_gflops);
        o+=",\"runtime_name\":"+jsonq(gpu.name);
        o+=",\"runtime_string\":"+jsonq(gpu.runtime);
        o+=",\"all_devices\":[";
        for(size_t i=0;i<rt.all_devices.size();i++){if(i)o+=",";o+=jsonq(rt.all_devices[i]);}
        o+="]";
        o+=",\"all_device_versions\":"+jsonq(rt.all_device_versions);
        o+=",\"opencl_build\":" + std::to_string(LAIC_HAVE_OPENCL);
        o+=",\"vulkan_build\":" + std::to_string(LAIC_HAVE_VULKAN);
        o+="}";
        return o;
    }
    bool begin_generation(LlamaRuntime*&r){std::unique_lock<std::mutex>g(mu);if(!rt||generating)return false;r=rt.get();r->set_backend(selected_backend);r->reset_gpu_stats();generating=true;tokens=0;tps=0;started=std::chrono::steady_clock::now();return true;}void finish_generation(){generating=false;}
    void handle(int c){std::string req;char b[8192];size_t want=0;for(;;){ssize_t n=recv(c,b,sizeof(b),0);if(n<=0){close(c);return;}req.append(b,size_t(n));auto hp=req.find("\r\n\r\n");if(hp!=std::string::npos){auto cl=header(req,"Content-Length: ");if(!cl.empty())try{want=std::stoull(cl);}catch(...){want=0;}size_t have=req.size()-hp-4;while(have<want){n=recv(c,b,sizeof(b),0);if(n<=0){close(c);return;}req.append(b,size_t(n));have+=size_t(n);}break;}}std::istringstream q(req);std::string method,path,ver;q>>method>>path>>ver;if(method=="OPTIONS"){writeall(c,reply(200,"{}"));close(c);return;}auto hp=req.find("\r\n\r\n");std::string body=hp==std::string::npos?"":req.substr(hp+4);
        if(method=="GET"&&path=="/"){std::ifstream f("web/index_v2.html");std::string x((std::istreambuf_iterator<char>(f)),{});if(x.empty()){writeall(c,reply(404,"{\"error\":\"web/index_v2.html not found\"}"));close(c);return;}auto marker="<div class=\"side-bottom\">";auto p=x.find(marker);if(p!=std::string::npos)x.insert(p,backend_panel(gpu));auto end=x.rfind("</body>");if(end!=std::string::npos)x.insert(end,"<script>window.LAIC_BACKEND=window.LAIC_BACKEND||(()=>localStorage.getItem('laic-backend')||'cpu');</script>");writeall(c,reply(200,x,"text/html; charset=utf-8"));close(c);return;}
        if(method=="GET"&&path=="/api/status"){writeall(c,reply(200,status()));close(c);return;}if(method=="GET"&&path=="/api/backends"){writeall(c,reply(200,backends()));close(c);return;}if(method=="GET"&&path=="/api/gpu/stats"){writeall(c,reply(200,gpu_stats()));close(c);return;}if(method=="GET"&&path=="/api/gpu/detect"){writeall(c,reply(200,gpu_detect()));close(c);return;}
        if(method=="POST"&&path=="/api/backend"){auto v=js(body,"backend");auto x=videocore::backend_from_string(v);if(x!=videocore::Backend::CPU&&!gpu.compute_available){writeall(c,reply(409,"{\"error\":\"GPU backend unavailable\"}"));close(c);return;}selected_backend=x;videocore::set_requested_backend(x);{std::lock_guard<std::mutex>g(mu);if(rt)rt->set_backend(x);}writeall(c,reply(200,"{\"backend\":"+jsonq(videocore::backend_name(x))+"}"));close(c);return;}
        if(method=="POST"&&path=="/api/benchmark"){LlamaRuntime*r=nullptr;if(!begin_generation(r)){writeall(c,reply(409,"{\"error\":\"model unavailable or generation already running\"}"));close(c);return;}auto t0=std::chrono::steady_clock::now();size_t max_tokens=jn(body,"max_tokens",32);if(!max_tokens)max_tokens=32;std::string prompt=js(body,"prompt");if(prompt.empty())prompt="Explain how a Raspberry Pi works in simple terms.";try{auto out=r->generate(prompt,max_tokens);auto t1=std::chrono::steady_clock::now();double ms=std::chrono::duration<double,std::milli>(t1-t0).count();size_t n=out.size();double rate=ms>0?double(n)*1000.0/ms:0.0;auto*gs=r->gpu_stats();double gf=gs&&gs->gpu_ns?double(gs->gpu_flops)/double(gs->gpu_ns):0.0;finish_generation();writeall(c,reply(200,"{\"tokens_per_sec\":"+std::to_string(rate)+",\"elapsed_ms\":"+std::to_string(ms)+",\"tokens\":"+std::to_string(n)+",\"gpu_gflops\":"+std::to_string(gf)+"}"));}catch(const std::exception&e){finish_generation();writeall(c,reply(500,std::string("{\"error\":")+jsonq(e.what())+"}"));}close(c);return;}
        if(method=="POST"&&path=="/api/model/load"){auto name=js(body,"name");if(name.empty()){writeall(c,reply(400,"{\"error\":\"missing model name\"}"));close(c);return;}try{auto r=std::make_unique<LlamaRuntime>();r->set_backend(selected_backend);r->load(dir+"/"+name);{std::lock_guard<std::mutex>g(mu);rt=std::move(r);loaded=name;}writeall(c,reply(200,"{\"ok\":true}"));}catch(const std::exception&e){writeall(c,reply(500,std::string("{\"error\":")+jsonq(e.what())+"}"));}close(c);return;}
        if(method=="POST"&&path=="/api/model/unload"){std::lock_guard<std::mutex>g(mu);rt.reset();loaded.clear();writeall(c,reply(200,"{\"ok\":true}"));close(c);return;}if(method=="GET"&&path=="/api/models"){writeall(c,reply(200,models()));close(c);return;}if(method=="POST"&&path=="/api/stop"){std::lock_guard<std::mutex>g(mu);if(rt)rt->stop();writeall(c,reply(200,"{\"ok\":true}"));close(c);return;}
        if(method=="POST"&&path=="/api/chat"){LlamaRuntime*r=nullptr;if(!begin_generation(r)){writeall(c,reply(409,"{\"error\":\"model unavailable or generation already running\"}"));close(c);return;}auto prompt=js(body,"prompt");if(prompt.empty())prompt="Hello";auto t0=std::chrono::steady_clock::now();try{auto out=r->generate(prompt,jn(body,"max_tokens",128));auto t1=std::chrono::steady_clock::now();double ms=std::chrono::duration<double,std::milli>(t1-t0).count();tokens=out.size();tps=ms>0?double(tokens.load())*1000.0/ms:0.0;finish_generation();writeall(c,reply(200,"{\"text\":"+jsonq(out)+",\"tokens\":"+std::to_string(tokens.load())+",\"tokens_per_sec\":"+std::to_string(tps.load())+"}"));}catch(const std::exception&e){finish_generation();writeall(c,reply(500,std::string("{\"error\":")+jsonq(e.what())+"}"));}close(c);return;}
        writeall(c,reply(404,"{\"error\":\"not found\"}"));close(c);
    }
    void run(){fd=socket(AF_INET,SOCK_STREAM,0);if(fd<0)throw std::runtime_error("socket failed");int one=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_ANY);a.sin_port=htons(port);if(bind(fd,(sockaddr*)&a,sizeof(a))<0){close(fd);fd=-1;throw std::runtime_error("bind failed");}if(listen(fd,32)<0){close(fd);fd=-1;throw std::runtime_error("listen failed");}running=true;while(running){int c=accept(fd,nullptr,nullptr);if(c<0){if(running)continue;break;}std::thread([this,c]{handle(c);}).detach();}if(fd>=0){close(fd);fd=-1;}}void stop()noexcept{running=false;if(fd>=0){shutdown(fd,SHUT_RDWR);close(fd);fd=-1;}}
};

ApiServer::ApiServer(uint16_t p,std::string d):impl_(new Impl(p,std::move(d))){} ApiServer::~ApiServer()=default; void ApiServer::run(){impl_->run();} void ApiServer::stop()noexcept{impl_->stop();}

}
