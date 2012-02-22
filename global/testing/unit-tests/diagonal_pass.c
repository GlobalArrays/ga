#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

main(int argv, char **argc)
{
  int g_A, g_V, dims[2]={6,6}, dims2[6], value1=5, value2=1;

  MPI_Init(&argv,&argc);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);

  g_A=NGA_Create(C_INT, 2, dims, "array_A", NULL);
  GA_Fill(g_A, &value1);
  //GA_Zero_diagonal(g_A);
  GA_Sync();
  GA_Print(g_A);

  g_V=NGA_Create(C_INT, 1, dims2, "array_V", NULL);
  GA_Fill(g_V, &value2);
  //GA_Shift_diagonal(g_A, &value2);
  GA_Set_diagonal(g_A, g_V);

  GA_Sync();
  GA_Print(g_A);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
