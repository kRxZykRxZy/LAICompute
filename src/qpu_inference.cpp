#include "laic/qpu_inference.hpp"
#include "laic/quant_v2.hpp"
#include "laic/compute_dot.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace laic::qpu {
namespace {
static float silu(float x){ return x/(1.f+std::exp(-x)); }
static void rope(std::vector<float>& x,size_t heads,size_t hd,size_t rd,size_t pos,float theta){
    rd=std::min(rd,hd);
    for(size_t h=0;h<heads;h++) for(size_t i=0;i<rd;i+=2){
        float a=float(pos)*std::pow(theta,-float(i)/float(hd)); float c=std::cos(a),s=std::sin(a); size_t b=h*hd+i; float u=x[b],v=x[b+1]; x[b]=u*c-v*s; x[b+1]=u*s+v*c;
    }
}
static float scalar_dot(const GgufTensor&w,const float*x,size_t base,size_t n){
    float s=0.f; for(size_t i=0;i<n;i++) s+=quant_v2::value(w.type,w.data,base+i)*x[i]; return s;
}
}

QpuLlamaRuntime::QpuLlamaRuntime() = default;
QpuLlamaRuntime::~QpuLlamaRuntime(){ gpu_.shutdown(); }

void QpuLlamaRuntime::load(const std::string& path){
    stop_.store(false,std::memory_order_relaxed); qpu_matvecs_=cpu_matvecs_=0; cache_plan_=CachePlan::detect();
    model_.load(path);
    if(model_.str("general.architecture","")!="llama") throw std::runtime_error("unsupported architecture: expected llama-family GGUF");
    hidden_=model_.u64("llama.embedding_length"); layers_=model_.u64("llama.block_count"); heads_=model_.u64("llama.attention.head_count"); kv_heads_=model_.u64("llama.attention.head_count_kv",heads_); ffn_=model_.u64("llama.feed_forward_length"); ctx_=model_.u64("llama.context_length",2048); rope_dim_=model_.u64("llama.rope.dimension_count",hidden_/std::max<size_t>(1,heads_)); theta_=float(model_.f64("llama.rope.freq_base",10000)); eps_=float(model_.f64("llama.attention.layer_norm_rms_epsilon",1e-5));
    if(!hidden_||!layers_||!heads_||!kv_heads_||hidden_%heads_||heads_%kv_heads_||!ffn_) throw std::runtime_error("invalid llama architecture metadata");
    head_dim_=hidden_/heads_; if(!rope_dim_||rope_dim_>head_dim_||(rope_dim_&1)) throw std::runtime_error("invalid RoPE dimension");
    tokenizer_.load(model_); kcache_.assign(layers_,{}); vcache_.assign(layers_,{}); pos_=0;
    if(!model_.has_tensor("token_embd.weight")||!model_.has_tensor("output_norm.weight")) throw std::runtime_error("missing required tensors");
    std::string err; gpu_.initialize(&err); // GPU is optional; CPU fallback keeps all GGUFs usable.
}

std::vector<float> QpuLlamaRuntime::embedding(uint32_t id) const{
    const auto&w=model_.tensor("token_embd.weight"); if(w.shape.size()!=2||w.shape[0]!=hidden_||id>=w.shape[1]) throw std::runtime_error("invalid embedding tensor");
    std::vector<float>x(hidden_); for(size_t i=0;i<hidden_;i++) x[i]=(w.type==GgmlType::F16||w.type==GgmlType::F32)?w.value(id*hidden_+i):quant_v2::value(w.type,w.data,id*hidden_+i); return x;
}

std::vector<float> QpuLlamaRuntime::matvec(const GgufTensor&w,const std::vector<float>&x,size_t out,size_t in){
    if(w.shape.size()!=2||w.shape[0]!=in||w.shape[1]!=out) throw std::runtime_error("unexpected weight shape: "+w.name);
    std::vector<float> y(out); bool gpu_ok=gpu_.available()&&(w.type==GgmlType::F16||w.type==GgmlType::F32);
    std::string err;
    if(gpu_ok){
        bool ok=false;
        if(w.type==GgmlType::F16) ok=gpu_.matvec_f16(reinterpret_cast<const uint16_t*>(w.data),x.data(),y.data(),out,in,&err);
        else ok=gpu_.matvec_f32(reinterpret_cast<const float*>(w.data),x.data(),y.data(),out,in,&err);
        if(ok){ ++qpu_matvecs_; return y; }
        gpu_.shutdown();
    }
    ++cpu_matvecs_;
    for(size_t o=0;o<out;o++){
        if(stop_.load(std::memory_order_relaxed)) throw std::runtime_error("generation stopped");
        if(w.type==GgmlType::F16) y[o]=compute::dot_f16(w.data+o*in*2,x.data(),in);
        else if(w.type==GgmlType::F32) y[o]=compute::dot_f32(w.data+o*in*4,x.data(),in);
        else y[o]=scalar_dot(w,x.data(),o*in,in);
    }
    return y;
}

void QpuLlamaRuntime::norm(std::vector<float>&x,const GgufTensor&w) const{
    if(w.elements()!=x.size()) throw std::runtime_error("invalid norm tensor: "+w.name); float ss=0; for(float v:x) ss+=v*v; float inv=1.f/std::sqrt(ss/float(x.size())+eps_); for(size_t i=0;i<x.size();i++) x[i]*=inv*w.value(i);
}

std::vector<float> QpuLlamaRuntime::step(uint32_t token){
    if(pos_>=ctx_) throw std::runtime_error("context length exceeded"); std::vector<float>x=embedding(token);
    for(size_t l=0;l<layers_;l++){
        if(stop_.load()) throw std::runtime_error("generation stopped"); std::string p="blk."+std::to_string(l)+"."; auto a=x; norm(a,model_.tensor(p+"attn_norm.weight"));
        auto q=matvec(model_.tensor(p+"attn_q.weight"),a,hidden_,hidden_); auto k=matvec(model_.tensor(p+"attn_k.weight"),a,kv_heads_*head_dim_,hidden_); auto v=matvec(model_.tensor(p+"attn_v.weight"),a,kv_heads_*head_dim_,hidden_); rope(q,heads_,head_dim_,rope_dim_,pos_,theta_); rope(k,kv_heads_,head_dim_,rope_dim_,pos_,theta_);
        auto&kc=kcache_[l]; auto&vc=vcache_[l]; kc.insert(kc.end(),k.begin(),k.end()); vc.insert(vc.end(),v.begin(),v.end()); size_t stride=kv_heads_*head_dim_, group=heads_/kv_heads_; std::vector<float>attn(hidden_);
        for(size_t h=0;h<heads_;h++){ size_t kh=h/group; std::vector<float>s(pos_+1); float mx=-std::numeric_limits<float>::infinity(); for(size_t t=0;t<=pos_;t++){float z=0;for(size_t d=0;d<head_dim_;d++)z+=q[h*head_dim_+d]*kc[t*stride+kh*head_dim_+d];s[t]=z/std::sqrt(float(head_dim_));mx=std::max(mx,s[t]);}float den=0;for(float&z:s){z=std::exp(z-mx);den+=z;}for(float&z:s)z/=den;for(size_t d=0;d<head_dim_;d++){float z=0;for(size_t t=0;t<=pos_;t++)z+=s[t]*vc[t*stride+kh*head_dim_+d];attn[h*head_dim_+d]=z;}}
        auto ao=matvec(model_.tensor(p+"attn_output.weight"),attn,hidden_,hidden_); compute::axpy_f32(x.data(),ao.data(),1.f,hidden_); auto f=x; norm(f,model_.tensor(p+"ffn_norm.weight")); auto gate=matvec(model_.tensor(p+"ffn_gate.weight"),f,ffn_,hidden_); auto up=matvec(model_.tensor(p+"ffn_up.weight"),f,ffn_,hidden_); for(size_t i=0;i<ffn_;i++)gate[i]=silu(gate[i])*up[i]; auto down=matvec(model_.tensor(p+"ffn_down.weight"),gate,hidden_,ffn_); compute::axpy_f32(x.data(),down.data(),1.f,hidden_);
    }
    norm(x,model_.tensor("output_norm.weight")); const auto&out=model_.has_tensor("output.weight")?model_.tensor("output.weight"):model_.tensor("token_embd.weight"); return matvec(out,x,tokenizer_.vocab_size(),hidden_);
}

uint32_t QpuLlamaRuntime::sample(const std::vector<float>&logits,const GenerationConfig&c,const std::vector<uint32_t>&history) const{
    std::vector<float>l=logits; if(c.repeat_penalty!=1.f)for(uint32_t id:history)if(id<l.size()){if(l[id]>=0)l[id]/=c.repeat_penalty;else l[id]*=c.repeat_penalty;} if(c.temperature<=0)return uint32_t(std::max_element(l.begin(),l.end())-l.begin());
    size_t k=c.top_k?std::min(c.top_k,l.size()):l.size(); std::vector<size_t>ix(l.size());std::iota(ix.begin(),ix.end(),0);if(k<ix.size()){std::nth_element(ix.begin(),ix.begin()+k,ix.end(),[&](size_t a,size_t b){return l[a]>l[b];});ix.resize(k);}float mx=-std::numeric_limits<float>::infinity();for(auto i:ix)mx=std::max(mx,l[i]);std::vector<double>p(k);double sum=0;for(size_t i=0;i<k;i++){p[i]=std::exp((l[ix[i]]-mx)/c.temperature);sum+=p[i];}for(auto&v:p)v/=sum;std::mt19937_64 rng(c.seed?c.seed:std::random_device{}());std::discrete_distribution<size_t>d(p.begin(),p.end());return uint32_t(ix[d(rng)]);
}

std::vector<uint32_t> QpuLlamaRuntime::generate_ids(const std::string&prompt,const GenerationConfig&cfg){std::vector<uint32_t>o;generate_ids(prompt,cfg,[&](uint32_t id){o.push_back(id);return true;});return o;}
std::vector<uint32_t> QpuLlamaRuntime::generate_ids(const std::string&prompt,const GenerationConfig&cfg,const TokenCallback&cb){
    stop_.store(false); kcache_.assign(layers_,{});vcache_.assign(layers_,{});pos_=0;std::string input=prompt;std::string templ=model_.str("tokenizer.chat_template","");if(templ.find("<|im_start|>")!=std::string::npos&&input.find("<|im_start|>")==std::string::npos)input="<|im_start|>user\n"+input+"<|im_end|>\n<|im_start|>assistant\n";auto ids=tokenizer_.encode(input);if(ids.empty())throw std::runtime_error("tokenization produced no tokens");if(tokenizer_.add_bos()&&ids.front()!=tokenizer_.bos_id())ids.insert(ids.begin(),tokenizer_.bos_id());std::vector<uint32_t>hist=ids;std::vector<float>logits;for(uint32_t id:ids){logits=step(id);++pos_;}std::vector<uint32_t>out;for(size_t n=0;n<cfg.max_tokens;n++){if(stop_.load())break;uint32_t next=sample(logits,cfg,hist);if(next==tokenizer_.eos_id())break;hist.push_back(next);out.push_back(next);if(cb&&!cb(next))break;logits=step(next);++pos_;}return out;
}
std::string QpuLlamaRuntime::generate(const std::string&prompt,const GenerationConfig&cfg){std::string o;generate_ids(prompt,cfg,[&](uint32_t id){o+=tokenizer_.decode(id);return true;});return o;}

} // namespace laic::qpu
