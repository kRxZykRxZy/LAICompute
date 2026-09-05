#include "laic/engine.hpp"
namespace laic {
Engine::Engine(CachePlan plan):cache_(plan),pipeline_(plan){}
void Engine::matmul(const float*A,const float*B,float*C,size_t M,size_t N,size_t K,const MatmulConfig&cfg){::laic::matmul_tiled(A,B,C,M,N,K,cfg);}
void Engine::matmul_fp16(const Half*A,const Half*B,float*C,size_t M,size_t N,size_t K,const MatmulConfig&cfg){::laic::matmul_fp16(A,B,C,M,N,K,cfg);}
} // namespace laic
