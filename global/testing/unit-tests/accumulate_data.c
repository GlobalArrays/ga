#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

main(int argc, char **argv)
{
  int g_A, dims[2]={5,5},lo[2]={1,1}, hi[2]={3,3}, local_A[3][3], ld=3, alpha=1;
  int i, j, value=1;

  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);

  for(i=0; i<3; i++)
    for(j=0; j<3; j++)local_A[i][j]=rand()%5;
  
  for(i=0; i<3; i++)
    {
      for(j=0; j<3; j++)
	printf("%d ", local_A[i][j]);
      printf("\n");
    }
  
  g_A=NGA_Create(C_INT, 2, dims, "array_A", NULL);
  GA_Fill(g_A, &value);
  GA_Sync();
  GA_Print(g_A);
  
  NGA_Acc(g_A, lo, hi, local_A, &ld, &alpha);
  GA_Sync();
  GA_Print(g_A);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
