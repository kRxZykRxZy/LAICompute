#include "laic/half.hpp"
#include "laic/tensor16.hpp"
#include "laic/cache.hpp"
#include "laic/matmul.hpp"
#include "laic/inference.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
int main(){using namespace laic;for(float x:{0.f,1.f,-2.5f,3.1415f,100.f}){float y=Half(x).to_float();assert(std::fabs(x-y)<std::max(.01f,std::fabs(x)*.002f));}Tensor16 t({2,3});assert(t.size()==6&&t.bytes()==12);CachePlan p=CachePlan::detect();if(p.cache_line_bytes)assert(p.cache_line_bytes>=8);float A[4]={1,2,3,4},B[4]={5,6,7,8},C[4]{};matmul_tiled(A,B,C,2,2,2,{8,8,8,1});assert(std::fabs(C[0]-19)<1e-5&&std::fabs(C[1]-22)<1e-5&&std::fabs(C[2]-43)<1e-5&&std::fabs(C[3]-50)<1e-5);Half hA[4]={Half(1),Half(2),Half(3),Half(4)},hB[4]={Half(5),Half(6),Half(7),Half(8)};matmul_fp16(hA,hB,C,2,2,2,{8,8,8,1});assert(std::fabs(C[0]-19)<.01&&std::fabs(C[3]-50)<.01);GenerationConfig g;assert(g.max_tokens==32&&g.temperature==0.0f);std::cout<<"LAICompute core tests passed\n";}
