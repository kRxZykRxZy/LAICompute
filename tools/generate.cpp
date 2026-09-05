#include "laic/inference.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
int main(int argc,char**argv){
    if(argc<3){std::cerr<<"usage: "<<argv[0]<<" MODEL.gguf \"prompt\" [max_tokens] [temperature] [top_k]\n";return 2;}
    try{laic::LlamaRuntime rt;rt.load(argv[1]);laic::GenerationConfig c;if(argc>3)c.max_tokens=std::stoul(argv[3]);if(argc>4)c.temperature=std::stof(argv[4]);if(argc>5)c.top_k=std::stoul(argv[5]);std::cout<<rt.generate(argv[2],c)<<std::flush<<'\n';return 0;}catch(const std::exception&e){std::cerr<<"laic_generate: "<<e.what()<<'\n';return 1;}
}
