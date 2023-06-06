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
  {
    CMX::Group *world = env->getWorldGroup();
    rank = world->rank();
    CMX::Allocation alloc;
    int64_t bytes = TEST_SIZE;
    alloc.malloc(bytes,world);
    world->barrier();
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
