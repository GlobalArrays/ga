#include "cmx.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  {
    CMX::Group *group = env->getWorldGroup();
    int size = group->size();
    int rank = group->rank();
    std::cout << "Printing from rank "<<rank<<" of "<<size<<std::endl;
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
