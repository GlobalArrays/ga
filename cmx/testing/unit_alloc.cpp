#include "group.hpp"
#include "environment.hpp"
#include "alloc.hpp"
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
    printf("p[%d] Got to 1\n",wrank);
    alloc.malloc(bytes,world);
    printf("p[%d] Got to 2\n",wrank);
    if (rank == 0) {
      std::cout <<"Distributed malloc completed"<<std::endl;
    }
    world->barrier();
    printf("p[%d] Got to 3\n",wrank);
    alloc.free();
    printf("p[%d] Got to 4\n",wrank);
    if (rank == 0) {
      std::cout <<"Allocation freed"<<std::endl;
    }
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
