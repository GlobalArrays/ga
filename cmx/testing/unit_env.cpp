#include "cmx.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  std::cout<<"Create CMX::Environment"<<std::endl;
  env->finalize();
  MPI_Finalize();
  return 0;
}
