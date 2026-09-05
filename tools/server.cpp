#include "laic/server.hpp"
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
int main(int argc,char**argv){uint16_t port=8080;if(argc>1)port=static_cast<uint16_t>(std::stoul(argv[1]));std::string dir=argc>2?argv[2]:"models";try{laic::ApiServer server(port,dir);std::cout<<"LAICompute UI/API: http://0.0.0.0:"<<port<<"\nModels: "<<dir<<"\n"<<std::flush;server.run();}catch(const std::exception&e){std::cerr<<"laic_server: "<<e.what()<<"\n";return 1;}return 0;}
