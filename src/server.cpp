#include "laic/server.hpp"
#include "laic/cpu.hpp"
#include "laic/inference.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <netinet/in.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/statvfs.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace laic { namespace {
std::string esc(const std::string&s){std::string o="\"";for(unsigned char c:s){if(c=='\"'||c=='\\')o+='\\',o+=char(c);else if(c=='\n')o+="\\n";else if(c=='\r')o+="\\r";else if(c<32){char b[7];snprintf(b,sizeof(b),"\\u%04x",c);o+=b;}else o+=char(c);}return o+'\"';}
std::string body(const std::string&req){auto p=req.find("\r\n\r\n");return p==std::string::npos?"":req.substr(p+4);}
std::string header(const std::string&req,const std::string&name){auto p=req.find(name);if(p==std::string::npos)return{};p+=name.size();auto e=req.find("\r\n",p);return req.substr(p,e-p);}
std::string json_string(const std::string&b,const std::string&key){std::string k="\""+key+"\"";auto p=b.find(k);if(p==std::string::npos)return{};p=b.find(':',p);if(p==std::string::npos)return{};p=b.find('"',p);if(p==std::string::npos)return{};++p;std::string o;for(;p<b.size();++p){if(b[p]=='"')break;if(b[p]=='\\'&&p+1<b.size()){++p;o+=b[p];}else o+=b[p];}return o;}
size_t json_num(const std::string&b,const std::string&key,size_t d){auto k="\""+key+"\"";auto p=b.find(k);if(p==std::string::npos)return d;p=b.find(':',p);if(p==std::string::npos)return d;try{return std::stoull(b.substr(p+1));}catch(...){return d;}}
std::string response(int code,const std::string&data,const std::string&type="application/json"){std::string text=code==200?"OK":code==201?"Created":code==400?"Bad Request":code==404?"Not Found":code==409?"Conflict":code==500?"Internal Server Error":"OK";return "HTTP/1.1 "+std::to_string(code)+" "+text+"\r\nContent-Type: "+type+"\r\nContent-Length: "+std::to_string(data.size())+"\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type,X-Filename\r\nAccess-Control-Allow-Methods: GET,POST,OPTIONS\r\nConnection: close\r\n\r\n"+data;}
}
struct ApiServer::Impl {
 uint16_t port; std::string dir; int fd=-1; std::atomic<bool> running{false}; std::mutex mu; std::unique_ptr<LlamaRuntime> runtime; std::string loaded; std::atomic<bool> generating{false}; std::atomic<size_t> generated{0}; double last_tps=0; std::chrono::steady_clock::time_point started;
 Impl(uint16_t p,std::string d):port(p),dir(std::move(d)){std::filesystem::create_directories(dir);}
 ~Impl(){stop();}
 std::string models(){std::string o="[";bool first=true;for(auto&e:std::filesystem::directory_iterator(dir)){if(!e.is_regular_file()||e.path().extension()!=".gguf")continue;auto s=std::filesystem::file_size(e.path());if(!first)o+=',';first=false;o+="{\"name\":"+esc(e.path().filename().string())+",\"size\":"+std::to_string(s)+",\"loaded\":"+(e.path().filename().string()==loaded?"true":"false")+"}";}return o+"]";}
 std::string status(){auto c=CpuInfo::detect();std::ifstream f("/proc/meminfo");std::string k;size_t v=0,freeb=0;while(f>>k>>v){if(k=="MemTotal:")v*=1024;else if(k=="MemAvailable:"){freeb=v*1024;break;}}size_t total=0,avail=0;std::ifstream m("/proc/meminfo");while(m>>k>>v){if(k=="MemTotal:")total=v*1024;else if(k=="MemAvailable:"){avail=v*1024;break;}}size_t ram=total?((total-avail)*100/total):0;double tps=last_tps;return "{\"generating\":"+(generating.load()?"true":"false")+",\"tokens\":"+std::to_string(generated.load())+",\"tokens_per_sec\":"+std::to_string(tps)+",\"ram_percent\":"+std::to_string(ram)+",\"cpu_logical\":"+std::to_string(c.logical_cpus)+",\"cpu_physical\":"+std::to_string(c.physical_cores)+",\"loaded\":"+esc(loaded)+"}";}
 void load(const std::string&name){std::filesystem::path p=std::filesystem::path(dir)/name;if(p.extension()!=".gguf"||!std::filesystem::exists(p))throw std::runtime_error("model not found");auto r=std::make_unique<LlamaRuntime>();r->load(p.string());std::lock_guard<std::mutex>g(mu);runtime=std::move(r);loaded=name;}
 void unload(){std::lock_guard<std::mutex>g(mu);runtime.reset();loaded.clear();}
 void chat(int fd,const std::string&b){std::string prompt=json_string(b,"message");size_t max=json_num(b,"max_tokens",64);float temp=float(json_num(b,"temperature",0))/1000.f;std::unique_lock<std::mutex>g(mu);if(!runtime){g.unlock();send(fd,response(409,"{\"error\":\"no model loaded\"}").c_str(),0,0);return;}auto*r=runtime.get();generating=true;generated=0;started=std::chrono::steady_clock::now();g.unlock();GenerationConfig cfg;cfg.max_tokens=max;cfg.temperature=temp;std::string out;try{auto ids=r->generate_ids(prompt,cfg,[&](uint32_t id){out+=r->tokenizer().decode(id);generated++;auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();last_tps=sec>0?generated/sec:0;return !r->stop_requested();});auto sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();last_tps=sec>0?generated/sec:0;generating=false;send(fd,response(200,"{\"text\":"+esc(out)+",\"tokens\":"+std::to_string(generated.load())+",\"tokens_per_sec\":"+std::to_string(last_tps)+",\"stopped\":"+(r->stop_requested()?"true":"false")+"}").c_str(),0,0);}catch(const std::exception&e){generating=false;send(fd,response(500,"{\"error\":"+esc(e.what())+"}").c_str(),0,0);}}
 void handle(int fd){char buf[65536];ssize_t n=recv(fd,buf,sizeof(buf),0);if(n<=0){close(fd);return;}std::string req(buf,size_t(n));std::istringstream q(req);std::string method,path,ver;q>>method>>path>>ver;if(method=="OPTIONS"){send(fd,response(200,"{} ").c_str(),0,0);close(fd);return;}if(method=="GET"&&path=="/"){std::ifstream f("web/index_v2.html");std::string s((std::istreambuf_iterator<char>(f)),{});auto r=response(200,s,"text/html; charset=utf-8");send(fd,r.c_str(),r.size(),0);close(fd);return;}if(method=="GET"&&path=="/api/models"){auto r=response(200,models());send(fd,r.c_str(),r.size(),0);close(fd);return;}if(method=="GET"&&path=="/api/status"){auto r=response(200,status());send(fd,r.c_str(),r.size(),0);close(fd);return;}if(method=="POST"&&path=="/api/models/load"){try{load(json_string(body(req),"name"));auto r=response(200,"{\"ok\":true}");send(fd,r.c_str(),r.size(),0);}catch(const std::exception&e){auto r=response(400,"{\"error\":"+esc(e.what())+"}");send(fd,r.c_str(),r.size(),0);}close(fd);return;}if(method=="POST"&&path=="/api/models/unload"){unload();auto r=response(200,"{\"ok\":true}");send(fd,r.c_str(),r.size(),0);close(fd);return;}if(method=="POST"&&path=="/api/stop"){std::lock_guard<std::mutex>g(mu);if(runtime)runtime->request_stop();auto r=response(200,"{\"ok\":true}");send(fd,r.c_str(),r.size(),0);close(fd);return;}if(method=="POST"&&path=="/api/models/upload"){auto name=header(req,"X-Filename: ");if(name.empty())name="model.gguf";std::filesystem::path p=std::filesystem::path(dir)/std::filesystem::path(name).filename();auto b=body(req);if(p.extension()!=".gguf"||b.size()<4){auto r=response(400,"{\"error\":\"GGUF file required\"}");send(fd,r.c_str(),r.size(),0);close(fd);return;}std::ofstream o(p,std::ios::binary);o.write(b.data(),std::streamsize(b.size()));o.close();auto r=response(201,"{\"name\":"+esc(p.filename().string())+",\"size\":"+std::to_string(b.size())+"}");send(fd,r.c_str(),r.size(),0);close(fd);return;}if(method=="POST"&&path=="/api/chat"){chat(fd,body(req));close(fd);return;}auto r=response(404,"{\"error\":\"not found\"}");send(fd,r.c_str(),r.size(),0);close(fd);}
 void run(){fd=socket(AF_INET,SOCK_STREAM,0);if(fd<0)throw std::runtime_error("socket failed");int one=1;setsockopt(fd,SOL_SOCKET,SO_REUSEADDR,&one,sizeof(one));sockaddr_in a{};a.sin_family=AF_INET;a.sin_addr.s_addr=htonl(INADDR_ANY);a.sin_port=htons(port);if(bind(fd,(sockaddr*)&a,sizeof(a))<0)throw std::runtime_error("bind failed");if(listen(fd,16)<0)throw std::runtime_error("listen failed");running=true;while(running){int c=accept(fd,nullptr,nullptr);if(c>=0)std::thread([this,c]{handle(c);}).detach();}}
 void stop(){running=false;if(fd>=0){shutdown(fd,SHUT_RDWR);close(fd);fd=-1;}}
};
ApiServer::ApiServer(uint16_t p,std::string d):impl_(new Impl(p,std::move(d))){ }
ApiServer::~ApiServer()=default;
void ApiServer::run(){impl_->run();}
void ApiServer::stop()noexcept{impl_->stop();}
}