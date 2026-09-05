#include "laic/matmul.hpp"
#include "laic/cache.hpp"
#include <algorithm>
#include <thread>
#include <vector>
#if defined(__AVX__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif
namespace laic { namespace {
MatmulConfig auto_cfg(){CachePlan p=CachePlan::detect();MatmulConfig c;size_t budget=p.l1_tile_bytes?p.l1_tile_bytes:16384;size_t n=16;while(n*n*3*sizeof(float)<budget&&n<128)n*=2;c.tile_m=std::max<size_t>(8,n/2);c.tile_n=std::max<size_t>(8,n/2);c.tile_k=std::max<size_t>(16,n);c.threads=std::max(1u,p.cpu.logical_cpus);return c;}
MatmulConfig norm(MatmulConfig c){auto a=auto_cfg();if(!c.tile_m)c.tile_m=a.tile_m;if(!c.tile_n)c.tile_n=a.tile_n;if(!c.tile_k)c.tile_k=a.tile_k;if(!c.threads)c.threads=a.threads;return c;}
void scalar(const float*A,const float*B,float*C,size_t M,size_t N,size_t K,size_t i0,size_t i1,const MatmulConfig&c){for(size_t i=i0;i<i1;i+=c.tile_m)for(size_t k0=0;k0<K;k0+=c.tile_k)for(size_t j0=0;j0<N;j0+=c.tile_n){size_t im=std::min(i+c.tile_m,i1),km=std::min(k0+c.tile_k,K),jm=std::min(j0+c.tile_n,N);for(size_t ii=i;ii<im;ii++)for(size_t k=k0;k<km;k++){float a=A[ii*K+k];const float*b=B+k*N;float*d=C+ii*N;for(size_t j=j0;j<jm;j++)d[j]+=a*b[j];}}}
#if defined(__AVX__) && (defined(__x86_64__) || defined(__i386__))
void avx(const float*A,const float*B,float*C,size_t M,size_t N,size_t K,size_t i0,size_t i1,const MatmulConfig&c){for(size_t i=i0;i<i1;i+=c.tile_m)for(size_t k0=0;k0<K;k0+=c.tile_k)for(size_t j0=0;j0<N;j0+=c.tile_n){size_t im=std::min(i+c.tile_m,i1),km=std::min(k0+c.tile_k,K),jm=std::min(j0+c.tile_n,N);for(size_t ii=i;ii<im;ii++)for(size_t k=k0;k<km;k++){__m256 av=_mm256_set1_ps(A[ii*K+k]);const float*b=B+k*N;float*d=C+ii*N;size_t j=j0;for(;j+8<=jm;j+=8){__m256 cv=_mm256_loadu_ps(d+j);_mm256_storeu_ps(d+j,_mm256_add_ps(cv,_mm256_mul_ps(av,_mm256_loadu_ps(b+j))));}for(;j<jm;j++)d[j]+=A[ii*K+k]*b[j];}}}
#endif
}
void matmul_tiled(const float*A,const float*B,float*C,size_t M,size_t N,size_t K,const MatmulConfig&cfg){auto c=norm(cfg);std::fill(C,C+M*N,0.0f);unsigned t=std::max(1u,std::min<unsigned>(c.threads,M));std::vector<std::thread>w;for(unsigned x=0;x<t;x++){size_t a=M*x/t,b=M*(x+1)/t;w.emplace_back([=](){
#if defined(__AVX__) && (defined(__x86_64__) || defined(__i386__))
 avx(A,B,C,M,N,K,a,b,c);
#else
 scalar(A,B,C,M,N,K,a,b,c);
#endif
 });}for(auto&x:w)x.join();}
void matmul_fp16(const Half*A,const Half*B,float*C,size_t M,size_t N,size_t K,const MatmulConfig&cfg){auto c=norm(cfg);std::fill(C,C+M*N,0.0f);unsigned t=std::max(1u,std::min<unsigned>(c.threads,M));std::vector<std::thread>w;for(unsigned x=0;x<t;x++){size_t i0=M*x/t,i1=M*(x+1)/t;w.emplace_back([=](){for(size_t i=i0;i<i1;i+=c.tile_m)for(size_t k0=0;k0<K;k0+=c.tile_k)for(size_t j0=0;j0<N;j0+=c.tile_n){size_t im=std::min(i+c.tile_m,i1),km=std::min(k0+c.tile_k,K),jm=std::min(j0+c.tile_n,N);for(size_t ii=i;ii<im;ii++)for(size_t kk=k0;kk<km;kk++){float a=A[ii*K+kk].to_float();const Half*b=B+kk*N;float*d=C+ii*N;for(size_t j=j0;j<jm;j++)d[j]+=a*b[j].to_float();}}});}for(auto&x:w)x.join();}
} // namespace laic
