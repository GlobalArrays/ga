#include "environment.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance();
  int rank = env->getWorldGroup()->rank();
  int size = env->getWorldGroup()->size();
  std::cout<<"Create CMX::Environment on rank "<<rank<<std::endl;
  std::cout<<"Create CMX::Environment size of world group "<<size<<std::endl;
  // Allocate a segment of memory
  void *ptrs;
  int64_t bytes = 10485676;
  std::cout<<"p["<<rank<<"] "<<"Calling dist_malloc"<<std::endl;
  env->dist_malloc(&ptrs,bytes,env->getWorldGroup());
  std::cout<<"p["<<rank<<"] "<<"Create allocation on rank "<<rank<<std::endl;  
  env->free(ptrs,env->getWorldGroup());
  std::cout<<"p["<<rank<<"] "<<"Free allocation on rank "<<rank<<std::endl;  
  env->finalize();
  MPI_Finalize();
  return 0;
}
