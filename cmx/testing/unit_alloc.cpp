#include "group.hpp"
#include "cmx.hpp"
#include <iostream>


#define TEST_SIZE 1024
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  {
    CMX::Group *world = env->getWorldGroup();
    int rank = world->rank();
    if (rank == 0) printf("Create void\n");
    CMX::CMX<void*> ga_void(world, TEST_SIZE);
    if (rank == 0) printf("Create int\n");
    CMX::CMX<int> ga_int(world, TEST_SIZE);
    if (rank == 0) printf("Create long\n");
    CMX::CMX<long> ga_long(world, TEST_SIZE);
    if (rank == 0) printf("Create float\n");
    CMX::CMX<float> ga_float(world, TEST_SIZE);
    if (rank == 0) printf("Create double\n");
    CMX::CMX<double> ga_double(world, TEST_SIZE);
    if (rank == 0) printf("Create complex\n");
    CMX::CMX<std::complex<float> > ga_complex(world, TEST_SIZE);
    if (rank == 0) printf("Create dcomplex\n");
    CMX::CMX<std::complex<double> > ga_dcomplex(world, TEST_SIZE);
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
