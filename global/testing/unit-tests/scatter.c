#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

#define N 5
#define D 2
main(int argc, char **argv)
{
  int g_A, dims[D]={5,10}, n=N, local_A[N], i, j, sub_array[N][D], value=1;

  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);

  for(i=0; i<N; i++)
    for(j=0; j<D; j++)sub_array[i][j]=rand()%5;

  for(i=0; i<N; i++)
    {
      for(j=0; j<D; j++)printf("%d ",sub_array[i][j]);
      printf("\n");
    }

  for(i=0; i<N; i++)printf("%d \n",local_A[i]=rand()%10);

  g_A=NGA_Create(C_INT, D, dims, "array_A", NULL);
  GA_Fill(g_A, &value);
  GA_Sync();
  GA_Print(g_A);

  NGA_Scatter(g_A, local_A, sub_array, N);
  GA_Sync();
  GA_Print(g_A);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
