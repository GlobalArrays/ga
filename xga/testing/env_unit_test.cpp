#include "xga_environment.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  XGA::Environment *env = XGA::Environment::instance();
  int rank = env->getWorldGroup()->rank();
  int size = env->getWorldGroup()->size();
  printf("Create XGA::Environment on rank %d\n",rank);
  printf("XGA::Environment size of world group  %d\n",size);
  env->finalize();
  MPI_Finalize();
  return 0;
}
