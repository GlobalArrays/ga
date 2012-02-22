#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

main(int argv, char **argc)
{
  int i, j, dims1[2]={5,4}, dims2[2]={4,5}, lo[3]={1,3,1}, hi[3]={2,3,2};
  int g_A, g_B, value=5, val_patch=1;

  MPI_Init(&argv, &argc);
  MA_init(C_INT, 1000, 1000);
  GA_Initialize();

  g_A=NGA_Create(C_INT, 2, dims1, "array_A", NULL);
  g_B=NGA_Create(C_INT, 2, dims2, "array_B", NULL);                         

  GA_Fill(g_A, &value);
  //  GA_Zero(g_A);
  GA_Print(g_A);                                                             
  //  NGA_Fill_patch(g_A, lo, hi, &val_patch);
  GA_Transpose(g_A,g_B);
  GA_Print(g_B);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
