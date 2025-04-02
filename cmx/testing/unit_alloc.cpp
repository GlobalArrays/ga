#include "cmx_group.hpp"
#include "cmx_environment.hpp"
#include "cmx_alloc.hpp"
#include <iostream>


#define TEST_SIZE 1048576
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  int rank;
  int wrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
  {
    CMX::Group *world = env->getWorldGroup();
    rank = world->rank();
    CMX::Allocation alloc;
    int64_t bytes = TEST_SIZE;
    alloc.malloc(bytes,world);
    if (rank == 0) {
      std::cout <<"Distributed malloc completed"<<std::endl;
    }
    world->barrier();
    alloc.free();
    if (rank == 0) {
      std::cout <<"Allocation freed"<<std::endl;
    }
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
