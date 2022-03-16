#include "cmx.hpp"
#include <iostream>

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  {
    printf("Create void\n");
    CMX::CMX<void*> ga_void;
    printf("Create int\n");
    CMX::CMX<int> ga_int;
    printf("Create long\n");
    CMX::CMX<long> ga_long;
    printf("Create float\n");
    CMX::CMX<float> ga_float;
    printf("Create double\n");
    CMX::CMX<double> ga_double;
    printf("Create complex\n");
    CMX::CMX<std::complex<float> > ga_complex;
    printf("Create dcomplex\n");
    CMX::CMX<std::complex<double> > ga_dcomplex;
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
