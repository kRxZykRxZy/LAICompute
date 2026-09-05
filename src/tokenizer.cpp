#include "laic/tokenizer.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace laic { namespace {
std::string cp(uint32_t x){std::string s;if(x<0x80)s.push_back(char(x));else if(x<0x800){s.push_back(char(0xC0|(x>>6)));s.push_back(char(0x80|(x&63)));}else{s.push_back(char(0xE0|(x>>12)));s.push_back(char(0x80|((x>>6)&63)));s.push_back(char(0x80|(x&63)));}return s;}
std::vector<std::string> utf8_symbols(const std::string& s){std::vector<std::string> r;for(size_t i=0;i<s.size();){unsigned char c=s[i];size_t n=(c<0x80)?1:((c&0xE0)==0xC0?2:((c&0xF0)==0xE0?3:4));if(i+n>s.size())n=1;r.push_back(s.substr(i,n));i+=n;}return r;}
}
void Gpt2Tokenizer::load(const GgufModel& m){
    tokens_=m.strings("tokenizer.ggml.tokens");if(tokens_.empty())throw std::runtime_error("GGUF has no tokenizer vocabulary");ids_.clear();for(uint32_t i=0;i<tokens_.size();++i)ids_.emplace(tokens_[i],i);
    const auto& merges=m.strings("tokenizer.ggml.merges");ranks_.clear();for(uint32_t i=0;i<merges.size();++i)ranks_[merges[i]]=i;
    bos_=uint32_t(m.u64("tokenizer.ggml.bos_token_id",0));eos_=uint32_t(m.u64("tokenizer.ggml.eos_token_id",0));unk_=uint32_t(m.u64("tokenizer.ggml.unknown_token_id",0));add_bos_=m.u64("tokenizer.ggml.add_bos_token",0)!=0;
    if(m.str("tokenizer.ggml.model","")!="gpt2")throw std::runtime_error("M7 currently supports GGUF GPT-2 tokenizers only");
}
std::string Gpt2Tokenizer::byte_encode(const std::string& s){
    std::vector<int> direct;for(int b=33;b<=126;++b)direct.push_back(b);for(int b=161;b<=172;++b)direct.push_back(b);for(int b=174;b<=255;++b)direct.push_back(b);
    std::unordered_map<int,int> mp;for(int i=0;i<direct.size();++i)mp[direct[i]]=direct[i];int extra=0;for(int b=0;b<256;++b)if(!mp.count(b)){while(std::find(direct.begin(),direct.end(),128+extra)!=direct.end())++extra;mp[b]=128+extra++;}
    std::string out;for(unsigned char b:s)out+=cp(uint32_t(mp[b]));return out;
}
std::string Gpt2Tokenizer::byte_decode(const std::string& s){
    std::vector<int> direct;for(int b=33;b<=126;++b)direct.push_back(b);for(int b=161;b<=172;++b)direct.push_back(b);for(int b=174;b<=255;++b)direct.push_back(b);
    std::unordered_map<int,int> inv;for(int b:direct)inv[b]=b;int extra=0;for(int b=0;b<256;++b){if(std::find(direct.begin(),direct.end(),b)==direct.end())inv[128+extra++]=b;}
    std::string out;for(auto& sym:utf8_symbols(s)){uint32_t x=0;for(unsigned char c:sym){x=(x<<6)|(c&63);if(!(c&0x80)){x=c;break;}}if(sym.size()==2)x=((unsigned char)sym[0]&31)<<6|((unsigned char)sym[1]&63);else if(sym.size()==3)x=((unsigned char)sym[0]&15)<<12|((unsigned char)sym[1]&63)<<6|((unsigned char)sym[2]&63);else if(sym.size()==4)x=((unsigned char)sym[0]&7)<<18|((unsigned char)sym[1]&63)<<12|((unsigned char)sym[2]&63)<<6|((unsigned char)sym[3]&63);auto it=inv.find(int(x));if(it!=inv.end())out.push_back(char(it->second));else out+=sym;}return out;
}
std::vector<uint32_t> Gpt2Tokenizer::encode(const std::string& text) const{
    std::vector<uint32_t> result;if(add_bos_)result.push_back(bos_);size_t pos=0;
    while(pos<text.size()){
        size_t special=text.size(),sid=UINT32_MAX;for(const auto& kv:ids_)if(kv.first.size()>2&&kv.first.rfind("<|",0)==0&&text.compare(pos,kv.first.size(),kv.first)==0){special=pos;sid=kv.second;break;}
        if(sid!=UINT32_MAX){result.push_back(uint32_t(sid));pos=text.find('>',pos)+1;continue;}
        size_t end=pos;while(end<text.size()){bool hit=false;for(const auto& kv:ids_)if(kv.first.size()>2&&kv.first.rfind("<|",0)==0&&text.compare(end,kv.first.size(),kv.first)==0){hit=true;break;}if(hit)break;++end;}
        std::string piece=byte_encode(text.substr(pos,end-pos));auto syms=utf8_symbols(piece);
        while(syms.size()>1){uint32_t best=UINT32_MAX;size_t bi=0;for(size_t i=0;i+1<syms.size();++i){auto it=ranks_.find(syms[i]+" "+syms[i+1]);if(it!=ranks_.end()&&it->second<best){best=it->second;bi=i;}}if(best==UINT32_MAX)break;syms[bi]+=syms[bi+1];syms.erase(syms.begin()+bi+1);}
        for(auto& s:syms){auto it=ids_.find(s);result.push_back(it==ids_.end()?unk_:it->second);}pos=end;
    }return result;
}
std::string Gpt2Tokenizer::decode(uint32_t id) const{if(id>=tokens_.size())return {};return byte_decode(tokens_[id]);}
} // namespace laic
