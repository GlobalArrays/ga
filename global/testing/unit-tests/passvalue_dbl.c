#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

main(int argv, char **argc)
{
  int rank, nprocs, i, j;
  int dims[2]={5,5};
  double g_A, value=1.4;
  long g_B, value_long=4;
  float g_C, value_flt=2.3;

  MPI_Init(&argv, &argc);
  MA_init(C_FLOAT, 1000, 1000);
  GA_Initialize();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  g_A=NGA_Create(C_DBL, 2, dims, "array_A", NULL);
  GA_Fill(g_A, &value);
  GA_Print(g_A);

  g_B=NGA_Create(C_LONG, 2, dims, "array_Long", NULL);
  GA_Fill(g_B, &value_long);
  GA_Print(g_B);

  g_C=NGA_Create(C_FLOAT, 2, dims, "array_Long", NULL);
  GA_Fill(g_C, &value_flt);
  GA_Print(g_C);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
