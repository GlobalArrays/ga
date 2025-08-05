#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  XGA::Environment *env = XGA::Environment::instance(&argc,&argv);
  {
    XGA::Group *group = env->getWorldGroup();
    int size = group->size();
    int rank = group->rank();
    printf("Printing from rank %d of %d\n",rank,size);
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
