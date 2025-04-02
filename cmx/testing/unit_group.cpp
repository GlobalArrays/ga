#include "cmx_group.hpp"
#include "cmx_environment.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  {
    CMX::Group *group = env->getWorldGroup();
    int size = group->size();
    int rank = group->rank();
    std::cout << "Printing from rank "<<rank<<" of "<<size<<std::endl;
    std::cout << std::endl;
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
