#include "group.hpp"
#include "environment.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  printf("p[%d] (main) Got to 1\n",rank);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  printf("p[%d] (main) Got to 2\n",rank);
  {
    CMX::Group *group = env->getWorldGroup();
  printf("p[%d] (main) Got to 3\n",rank);
    int size = group->size();
  printf("p[%d] (main) Got to 4\n",rank);
    int rank = group->rank();
    std::cout << "Printing from rank "<<rank<<" of "<<size<<std::endl;
    std::cout << std::endl;
  }
  printf("p[%d] (main) Got to 5\n",rank);
  env->finalize();
  printf("p[%d] (main) Got to 6\n",rank);
  MPI_Finalize();
  return 0;
}
