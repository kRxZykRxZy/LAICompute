#include "laic/server.hpp"
#include "laic/cpu.hpp"
#include "laic/inference.hpp"
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
void writeall(int fd,const std::string&s){for(size_t p=0;p<s.size();){ssize_t n=send(fd,s.data()+p,s.size()-p,0);if(n<=0)return;p+=size_t(n);}}
std::string header(const std::string&r,const std::string&n){auto p=r.find(n);if(p==std::string::npos)return{};p+=n.size();auto e=r.find("\r\n",p);return r.substr(p,e-p);}
}

struct ApiServer::Impl{
    uint16_t port;std::string dir;int fd=-1;std::atomic<bool>running{false};std::mutex mu;
    std::unique_ptr<LlamaRuntime>rt;std::string loaded;std::atomic<bool>generating{false};
    std::atomic<size_t>tokens{0};std::atomic<double>tps{0};std::chrono::steady_clock::time_point started;
    Impl(uint16_t p,std::string d):port(p),dir(std::move(d)){std::filesystem::create_directories(dir);}
    ~Impl(){stop();}

    std::string models(){std::string o="[";bool first=true;for(auto&e:std::filesystem::directory_iterator(dir)){if(!e.is_regular_file()||e.path().extension()!=".gguf")continue;if(!first)o+=',';first=false;o+="{\"name\":"+jsonq(e.path().filename().string())+",\"size\":"+std::to_string(std::filesystem::file_size(e.path()))+",\"loaded\":"+(e.path().filename().string()==loaded?"true":"false")+"}";}return o+"]";}
    std::string status(){
        std::ifstream f("/proc/meminfo");std::string k;size_t v,total=0,avail=0;
        while(f>>k>>v){if(k=="MemTotal:")total=v*1024;else if(k=="MemAvailable:"){avail=v*1024;break;}}
        auto c=CpuInfo::detect();size_t ram=total?100-(avail*100/total):0;
        std::string o="{\"generating\":";o+=generating.load()?"true":"false";
        o+=",\"tokens\":"+std::to_string(tokens.load())+",\"tokens_per_sec\":"+std::to_string(tps.load());
        o+=",\"ram_percent\":"+std::to_string(ram)+",\"cpu_logical\":"+std::to_string(c.logical_cpus)+",\"cpu_physical\":"+std::to_string(c.physical_cores);
        o+=",\"inference_threads\":"+std::to_string(std::thread::hardware_concurrency());
        o+=",\"loaded\":"+jsonq(loaded)+"}";return o;
    }
    bool begin_generation(LlamaRuntime*&r){std::unique_lock<std::mutex>g(mu);if(!rt)return false;if(generating)return false;r=rt.get();generating=true;tokens=0;tps=0;started=std::chrono::steady_clock::now();return true;}
    void finish_generation(){generating=false;}

    void handle(int c){
        std::string req;char b[8192];size_t want=0;
        for(;;){ssize_t n=recv(c,b,sizeof(b),0);if(n<=0){close(c);return;}req.append(b,size_t(n));auto hp=req.find("\r\n\r\n");if(hp!=std::string::npos){auto cl=header(req,"Content-Length: ");if(!cl.empty())try{want=std::stoull(cl);}catch(...){want=0;}size_t have=req.size()-hp-4;while(have<want){n=recv(c,b,sizeof(b),0);if(n<=0){close(c);return;}req.append(b,size_t(n));have+=size_t(n);}break;}}
        std::istringstream q(req);std::string method,path,ver;q>>method>>path>>ver;
        if(method=="OPTIONS"){writeall(c,reply(200,"{}"));close(c);return;}
        auto hp=req.find("\r\n\r\n");std::string body=hp==std::string::npos?"":req.substr(hp+4);
        if(method=="GET"&&path=="/"){std::ifstream f("web/index.html");std::string x((std::istreambuf_iterator<char>(f)),{});if(x.empty())writeall(c,reply(404,"{\"error\":\"web/index.html not found\"}"));else writeall(c,reply(200,x,"text/html; charset=utf-8"));close(c);return;}
        if(method=="GET"&&path=="/api/models"){writeall(c,reply(200,models()));close(c);return;}
        if(method=="GET"&&path=="/api/status"){writeall(c,reply(200,status()));close(c);return;}
        if(method=="POST"&&path=="/api/models/upload"){
            std::string name=header(req,"X-Filename: ");if(name.empty())name="model.gguf";std::filesystem::path p=std::filesystem::path(dir)/std::filesystem::path(name).filename();
            if(p.extension()!=".gguf"||body.size()<4||body.compare(0,4,"GGUF")!=0){writeall(c,reply(400,"{\"error\":\"Valid GGUF file required\"}"));close(c);return;}
            std::ofstream o(p,std::ios::binary|std::ios::trunc);if(!o){writeall(c,reply(500,"{\"error\":\"cannot create model file\"}"));close(c);return;}o.write(body.data(),std::streamsize(body.size()));
            if(!o){writeall(c,reply(500,"{\"error\":\"failed to save model\"}"));close(c);return;}o.close();
            writeall(c,reply(201,"{\"name\":"+jsonq(p.filename().string())+",\"size\":"+std::to_string(body.size())+"}"));close(c);return;
        }
        if(method=="POST"&&path=="/api/models/load"){
            try{auto name=js(body,"name");auto p=std::filesystem::path(dir)/std::filesystem::path(name).filename();if(name.empty()||p.extension()!=".gguf"||!std::filesystem::exists(p))throw std::runtime_error("model not found");
                auto x=std::make_unique<LlamaRuntime>();x->load(p.string());std::lock_guard<std::mutex>g(mu);if(generating)throw std::runtime_error("generation in progress");rt=std::move(x);loaded=p.filename().string();writeall(c,reply(200,"{\"ok\":true}"));
            }catch(const std::exception&e){writeall(c,reply(400,"{\"error\":"+jsonq(e.what())+"}"));}close(c);return;
        }
        if(method=="POST"&&path=="/api/models/unload"){std::lock_guard<std::mutex>g(mu);if(generating){writeall(c,reply(409,"{\"error\":\"stop generation before unloading\"}"));close(c);return;}rt.reset();loaded.clear();writeall(c,reply(200,"{\"ok\":true}"));close(c);return;}
        if(method=="POST"&&path=="/api/stop"){std::lock_guard<std::mutex>g(mu);if(rt)rt->request_stop();writeall(c,reply(200,"{\"ok\":true}"));close(c);return;}

        if(method=="POST"&&(path=="/api/chat"||path=="/api/chat/stream")){
            LlamaRuntime*r=nullptr;if(!begin_generation(r)){std::lock_guard<std::mutex>g(mu);if(!rt)writeall(c,reply(409,"{\"error\":\"no model loaded\"}"));else writeall(c,reply(409,"{\"error\":\"generation already running\"}"));close(c);return;}
            GenerationConfig cfg;cfg.max_tokens=jn(body,"max_tokens",128);cfg.temperature=float(jn(body,"temperature_milli",0))/1000.f;bool stream=path=="/api/chat/stream";
            if(stream){writeall(c,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type\r\n\r\n");}
            std::string out;bool ok=true;
            try{
                r->generate_ids(js(body,"message"),cfg,[&](uint32_t id){
                    std::string piece=r->tokenizer().decode(id);out+=piece;size_t n=tokens.fetch_add(1)+1;
                    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();tps=sec>0?n/sec:0;
                    if(stream){std::string event="data: {\"token\":"+jsonq(piece)+",\"tokens\":"+std::to_string(n)+",\"tokens_per_sec\":"+std::to_string(tps.load())+"}\n\n";writeall(c,event);}
                    return !r->stop_requested();
                });
                double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();tps=sec>0?tokens.load()/sec:0;bool stopped=r->stop_requested();finish_generation();
                if(stream){writeall(c,"data: {\"done\":true,\"tokens\":"+std::to_string(tokens.load())+",\"tokens_per_sec\":"+std::to_string(tps.load())+",\"stopped\":"+(stopped?"true":"false")+"}\n\n");}
                else writeall(c,reply(200,"{\"text\":"+jsonq(out)+",\"tokens\":"+std::to_string(tokens.load())+",\"tokens_per_sec\":"+std::to_string(tps.load())+",\"stopped\":"+(stopped?"true":"false")+"}"));
            }catch(const std::exception&e){finish_generation();if(stream)writeall(c,"data: {\"error\":"+jsonq(e.what())+"}\n\ndata: {\"done\":true}\n\n");else writeall(c,reply(500,"{\"error\":"+jsonq(e.what())+"}"));}
            close(c);return;
        }
        writeall(c,reply(404,"{\"error\":\"not found\"}"));close(c);
    }
    void run(){fd=socket(AF_INET,SOCK_STREAM,0);if(fd<0)throw std::runtime_error("socket failed");int one=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_ANY);a.sin_port=htons(port);if(bind(fd,(sockaddr*)&a,sizeof(a))<0)throw std::runtime_error("bind failed");if(listen(fd,32)<0)throw std::runtime_error("listen failed");running=true;while(running){int c=accept(fd,nullptr,nullptr);if(c>=0)std::thread([this,c]{handle(c);}).detach();}}
    void stop(){running=false;if(fd>=0){shutdown(fd,SHUT_RDWR);close(fd);fd=-1;}}
};
ApiServer::ApiServer(uint16_t p,std::string d):impl_(new Impl(p,std::move(d))){}ApiServer::~ApiServer()=default;void ApiServer::run(){impl_->run();}void ApiServer::stop()noexcept{impl_->stop();}
}