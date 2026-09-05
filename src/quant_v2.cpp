#include "laic/gguf.hpp"
#include "laic/half.hpp"
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace laic::quant_v2 {
namespace {
inline float h(const uint8_t *p) noexcept { return Half::from_bits(uint16_t(p[0]) | (uint16_t(p[1])<<8)).to_float(); }
inline void sm4(int j,const uint8_t*q,uint8_t&d,uint8_t&m) noexcept { if(j<4){d=q[j]&63;m=q[j+4]&63;}else{d=(q[j+4]&15)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);} }
}
float value(GgmlType type,const uint8_t*data,size_t i){
 const size_t b=i/32,o=i%32;
 switch(type){
 case GgmlType::Q8_0:{auto*p=data+b*34;return h(p)*float(int8_t(p[2+o]));}
 case GgmlType::Q4_0:{auto*p=data+b*18;uint8_t z=p[2+o/2];return h(p)*float(((o&1)?z>>4:z&15)-8);}
 case GgmlType::Q4_1:{auto*p=data+b*20;uint8_t z=p[4+o/2];return h(p)*float((o&1)?z>>4:z&15)+h(p+2);}
 case GgmlType::Q5_0:{auto*p=data+b*22;uint32_t qh;std::memcpy(&qh,p+2,4);uint8_t z=p[6+o/2];int v=(o&1)?z>>4:z&15;v|=((qh>>o)&1)<<4;return h(p)*float(v-16);}
 case GgmlType::Q5_1:{auto*p=data+b*24;uint32_t qh;std::memcpy(&qh,p+4,4);uint8_t z=p[8+o/2];int v=(o&1)?z>>4:z&15;v|=((qh>>o)&1)<<4;return h(p)*float(v)+h(p+2);}
 case GgmlType::Q2_K:{auto*B=data+(i/256)*84;size_t p=i%256,n=p/128,l=p%32,j=(p%128)/32;uint8_t sc=B[8*n+l/16+2*j],q=(B[16+32*n+l]>>(2*j))&3;return h(B+80)*float(sc&15)*q-h(B+82)*float(sc>>4);}
 case GgmlType::Q3_K:{auto*B=data+(i/256)*110;size_t p=i%256,n=p/128,l=p%32,j=(p%128)/32,is0=l/16,is=8*n+2*j+is0;auto*s=B+96;int us;if(is<4)us=(s[is]&15)|(((s[is+8])&3)<<4);else if(is<8)us=(s[is]&15)|(((s[is+4]>>2)&3)<<4);else if(is<12)us=(s[is-8]>>4)|(((s[is]>>4)&3)<<4);else us=(s[is-8]>>4)|(((s[is-4]>>6)&3)<<4);uint8_t q=(B[32+32*n+l]>>(2*j))&3,mask=uint8_t(1u<<(4*n+j));return h(B+108)*float(us-32)*float(int(q)-((B[l]&mask)?0:4));}
 case GgmlType::Q4_K:{auto*B=data+(i/256)*144;size_t p=i%256,g=p/64,half=(p%64)/32,e=p%32,si=2*g+half;uint8_t sc,m;sm4(int(si),B+4,sc,m);uint8_t q=B[16+e];int v=half?q>>4:q&15;return h(B)*float(sc*v)-h(B+2)*float(m);}
 case GgmlType::Q5_K:{auto*B=data+(i/256)*176;size_t p=i%256,il=p/64,w=p%64,half=w/32,e=w%32,si=2*il+half;uint8_t sc,m;sm4(int(si),B+4,sc,m);uint8_t ql=B[16+32*il+e/2],qh=B[144+e/2];int nib=half?q l>>4:ql&15;int bit=2*il+half;return h(B)*float(sc*(nib+((qh&(1u<<bit))?16:0)))-h(B+2)*float(m);}
 case GgmlType::Q6_K:{auto*B=data+(i/256)*210;size_t p=i%256,ip=p/128,w=p%128,sub=w/32,l=w%32;uint8_t ql=(sub&1)?B[2+64*ip+32+l]:B[2+64*ip+l];int q4=sub<2?(ql&15):(ql>>4);uint8_t qh=B[130+32*ip+l];int q6=q4|(((qh>>(2*(sub%2+2*(sub/2))))&3)<<4);int scidx=8*ip+sub/2*4+sub%2*2;return h(B+208)*float(int8_t(B[194+scidx]))*float(q6-32);}
 default: throw std::runtime_error("unsupported quantized GGML type"); }
}
}
