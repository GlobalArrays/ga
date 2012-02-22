#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

main(int argc, char **argv)
{
  int g_A, dims[2]={5,5}, value=5, i, j;
  int lo[2]={1,1}, hi[2]={3,3}, local_A[3][4], ld=3, alpha=1;
 
  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);

  ga_nbhdl_t nbhandle;

  for(i=0; i<3; i++)
    for(j=0; j<4; j++)local_A[i][j]=(i+j)+2;

  for(i=0; i<3; i++)
    {
      for(j=0; j<4; j++)printf("%d ", local_A[i][j]);
      printf("\n");
    }

  g_A=NGA_Create(C_INT, 2, dims, "array_A", NULL);
  GA_Fill(g_A, &value);
  GA_Sync();
  GA_Print(g_A);

  NGA_NbAcc(g_A, lo, hi, local_A, &ld, &alpha, &nbhandle);

  NGA_NbWait(&nbhandle);

  GA_Sync();
  GA_Print(g_A);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
