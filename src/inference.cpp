#include "laic/inference.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>

namespace laic { namespace {

void rope(std::vector<float>& x,size_t heads,size_t hd,size_t rd,size_t pos,float theta){
    rd=std::min(rd,hd);
    for(size_t h=0;h<heads;++h) for(size_t i=0;i<rd;i+=2){
        float inv=std::pow(theta,-float(i)/float(hd));
        float a=float(pos)*inv,c=std::cos(a),s=std::sin(a);
        size_t b=h*hd+i; float x0=x[b],x1=x[b+1];
        x[b]=x0*c-x1*s; x[b+1]=x0*s+x1*c;
    }
}
float silu(float x){return x/(1.0f+std::exp(-x));}

// Persistent workers: one worker per logical CPU thread. The pool is reused for
// every matvec instead of creating/joining threads for every token.
class ParallelPool {
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_, done_;
    std::atomic<size_t> next_{0};
    size_t jobs_=0, active_=0, generation_=0;
    bool stop_=false;
    std::function<void(size_t)> fn_;

    void worker(){
        size_t seen=0;
        for(;;){
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk,[&]{return stop_||generation_!=seen;});
            if(stop_) return;
            seen=generation_;
            auto fn=fn_; size_t jobs=jobs_;
            lk.unlock();
            for(;;){
                size_t i=next_.fetch_add(1,std::memory_order_relaxed);
                if(i>=jobs) break;
                fn(i);
            }
            lk.lock();
            if(--active_==0) done_.notify_one();
        }
    }
public:
    ParallelPool(){
        unsigned n=std::thread::hardware_concurrency();
        if(n==0) n=1;
        workers_.reserve(n);
        for(unsigned i=0;i<n;++i) workers_.emplace_back(&ParallelPool::worker,this);
    }
    ~ParallelPool(){
        {std::lock_guard<std::mutex> lk(mu_);stop_=true;cv_.notify_all();}
        for(auto& t:workers_) if(t.joinable()) t.join();
    }
    unsigned threads() const noexcept {return unsigned(workers_.size());}
    void run(size_t jobs,const std::function<void(size_t)>& fn){
        if(!jobs) return;
        // Avoid pool overhead for tiny jobs while still using every logical CPU
        // for the large matrix operations that dominate transformer inference.
        if(jobs<workers_.size()*2){for(size_t i=0;i<jobs;++i)fn(i);return;}
        {
            std::lock_guard<std::mutex> lk(mu_);
            jobs_=jobs; fn_=fn; next_.store(0,std::memory_order_relaxed);
            active_=workers_.size(); ++generation_;
        }
        cv_.notify_all();
        std::unique_lock<std::mutex> lk(mu_);
        done_.wait(lk,[&]{return active_==0;});
    }
};
ParallelPool& pool(){static ParallelPool p;return p;}

}

void LlamaRuntime::load(const std::string& path){
    stop_requested_.store(false);
    model_.load(path);
    if(model_.str("general.architecture","")!="llama") throw std::runtime_error("M7 currently supports GGUF llama architecture only");
    hidden_=model_.u64("llama.embedding_length"); layers_=model_.u64("llama.block_count");
    heads_=model_.u64("llama.attention.head_count"); kv_heads_=model_.u64("llama.attention.head_count_kv",heads_);
    ffn_=model_.u64("llama.feed_forward_length"); ctx_=model_.u64("llama.context_length",2048);
    rope_dim_=model_.u64("llama.rope.dimension_count",hidden_/heads_);
    theta_=float(model_.f64("llama.rope.freq_base",10000));
    eps_=float(model_.f64("llama.attention.layer_norm_rms_epsilon",1e-5));
    if(!hidden_||!layers_||!heads_||!kv_heads_||hidden_%heads_||heads_%kv_heads_) throw std::runtime_error("invalid llama dimensions");
    head_dim_=hidden_/heads_;
    tokenizer_.load(model_);
    if(tokenizer_.vocab_size()!=model_.u64("llama.vocab_size",tokenizer_.vocab_size())) throw std::runtime_error("tokenizer/model vocabulary mismatch");
    cache_.assign(layers_,{});
    for(auto& c:cache_){c.k.resize(ctx_*kv_heads_*head_dim_);c.v.resize(ctx_*kv_heads_*head_dim_);}
    pos_=0;
    if(!model_.has_tensor("token_embd.weight")||!model_.has_tensor("output_norm.weight")) throw std::runtime_error("missing required Llama tensors");
}

std::vector<float> LlamaRuntime::embedding(uint32_t id)const{
    const auto&w=model_.tensor("token_embd.weight");
    if(w.shape.size()!=2||id>=w.shape[1]||w.shape[0]!=hidden_) throw std::runtime_error("invalid embedding tensor");
    std::vector<float>x(hidden_); for(size_t i=0;i<hidden_;++i)x[i]=w.value(id*hidden_+i); return x;
}

std::vector<float> LlamaRuntime::matvec(const GgufTensor&w,const std::vector<float>&x,size_t out,size_t in)const{
    if(w.shape.size()!=2||w.shape[0]!=in||w.shape[1]!=out) throw std::runtime_error("unexpected weight shape: "+w.name);
    std::vector<float> y(out,0.0f);
    // Each output row is independent, so split rows across every logical CPU.
    pool().run(out,[&](size_t o){
        float sum=0.0f;
        for(size_t i=0;i<in;++i) sum+=w.value(o*in+i)*x[i];
        y[o]=sum;
    });
    return y;
}

void LlamaRuntime::norm(std::vector<float>&x,const GgufTensor&w)const{
    if(w.elements()!=x.size()) throw std::runtime_error("invalid RMSNorm tensor: "+w.name);
    float ss=0; for(float v:x)ss+=v*v;
    float inv=1.0f/std::sqrt(ss/float(x.size())+eps_);
    pool().run(x.size(),[&](size_t i){x[i]=x[i]*inv*w.value(i);});
}

std::vector<float> LlamaRuntime::step(uint32_t token){
    if(pos_>=ctx_) throw std::runtime_error("context length exceeded");
    std::vector<float>x=embedding(token);
    for(size_t l=0;l<layers_;++l){
        std::string p="blk."+std::to_string(l)+".";
        std::vector<float>a=x; norm(a,model_.tensor(p+"attn_norm.weight"));
        auto q=matvec(model_.tensor(p+"attn_q.weight"),a,hidden_,hidden_);
        auto k=matvec(model_.tensor(p+"attn_k.weight"),a,kv_heads_*head_dim_,hidden_);
        auto v=matvec(model_.tensor(p+"attn_v.weight"),a,kv_heads_*head_dim_,hidden_);
        rope(q,heads_,head_dim_,rope_dim_,pos_,theta_); rope(k,kv_heads_,head_dim_,rope_dim_,pos_,theta_);
        auto&lc=cache_[l]; size_t stride=kv_heads_*head_dim_;
        std::copy(k.begin(),k.end(),lc.k.begin()+pos_*stride); std::copy(v.begin(),v.end(),lc.v.begin()+pos_*stride);
        std::vector<float>attn(hidden_,0); size_t group=heads_/kv_heads_;
        // Attention heads are independent. Run all heads concurrently.
        pool().run(heads_,[&](size_t h){
            size_t kh=h/group; std::vector<float>scores(pos_+1);
            float mx=-std::numeric_limits<float>::infinity();
            for(size_t t=0;t<=pos_;++t){
                float s=0; for(size_t d=0;d<head_dim_;++d)s+=q[h*head_dim_+d]*lc.k[t*stride+kh*head_dim_+d];
                scores[t]=s/std::sqrt(float(head_dim_)); mx=std::max(mx,scores[t]);
            }
            float den=0; for(float&s:scores){s=std::exp(s-mx);den+=s;} for(float&s:scores)s/=den;
            for(size_t d=0;d<head_dim_;++d){float z=0;for(size_t t=0;t<=pos_;++t)z+=scores[t]*lc.v[t*stride+kh*head_dim_+d];attn[h*head_dim_+d]=z;}
        });
        auto ao=matvec(model_.tensor(p+"attn_output.weight"),attn,hidden_,hidden_);
        pool().run(hidden_,[&](size_t i){x[i]+=ao[i];});
        std::vector<float>f=x; norm(f,model_.tensor(p+"ffn_norm.weight"));
        auto gate=matvec(model_.tensor(p+"ffn_gate.weight"),f,ffn_,hidden_);
        auto up=matvec(model_.tensor(p+"ffn_up.weight"),f,ffn_,hidden_);
        pool().run(ffn_,[&](size_t i){gate[i]=silu(gate[i])*up[i];});
        auto down=matvec(model_.tensor(p+"ffn_down.weight"),gate,hidden_,ffn_);
        pool().run(hidden_,[&](size_t i){x[i]+=down[i];});
    }
    norm(x,model_.tensor("output_norm.weight"));
    const auto&out=model_.has_tensor("output.weight")?model_.tensor("output.weight"):model_.tensor("token_embd.weight");
    return matvec(out,x,tokenizer_.vocab_size(),hidden_);
}

uint32_t LlamaRuntime::sample(const std::vector<float>&logits,const GenerationConfig&cfg)const{
    if(cfg.temperature<=0)return uint32_t(std::max_element(logits.begin(),logits.end())-logits.begin());
    size_t k=cfg.top_k?std::min(cfg.top_k,logits.size()):logits.size(); std::vector<size_t>ix(logits.size()); std::iota(ix.begin(),ix.end(),0);
    if(k<ix.size()){std::nth_element(ix.begin(),ix.begin()+k,ix.end(),[&](size_t a,size_t b){return logits[a]>logits[b];});ix.resize(k);}
    float mx=-std::numeric_limits<float>::infinity();for(auto i:ix)mx=std::max(mx,logits[i]);std::vector<double>p(k);double sum=0;
    for(size_t j=0;j<k;++j){p[j]=std::exp((logits[ix[j]]-mx)/cfg.temperature);sum+=p[j];}
    static thread_local std::mt19937 rng(std::random_device{}());std::uniform_real_distribution<double>d(0,sum);double r=d(rng);
    for(size_t j=0;j<k;++j)if((r-=p[j])<=0)return uint32_t(ix[j]);return uint32_t(ix.back());
}

std::vector<uint32_t>LlamaRuntime::generate_ids(const std::string&prompt,const GenerationConfig&cfg){return generate_ids(prompt,cfg,{});}
std::vector<uint32_t>LlamaRuntime::generate_ids(const std::string&prompt,const GenerationConfig&cfg,const TokenCallback&callback){
    stop_requested_.store(false);pos_=0;
    for(auto&c:cache_){std::fill(c.k.begin(),c.k.end(),0);std::fill(c.v.begin(),c.v.end(),0);}
    auto ids=tokenizer_.encode(prompt);if(ids.empty())ids.push_back(tokenizer_.bos_id());
    std::vector<uint32_t>out;out.reserve(cfg.max_tokens);std::vector<float>logits;
    for(uint32_t id:ids){if(stop_requested_.load(std::memory_order_relaxed)||pos_>=ctx_)break;logits=step(id);++pos_;}
    for(size_t n=0;n<cfg.max_tokens&&pos_<ctx_;++n){
        if(stop_requested_.load(std::memory_order_relaxed))break;
        uint32_t next=sample(logits,cfg);out.push_back(next);
        if(callback&&!callback(next))break;
        if(next==tokenizer_.eos_id())break;
        logits=step(next);++pos_;
    }
    return out;
}
std::string LlamaRuntime::generate(const std::string&prompt,const GenerationConfig&cfg){std::string out;for(auto id:generate_ids(prompt,cfg))out+=tokenizer_.decode(id);return out;}

} // namespace laic