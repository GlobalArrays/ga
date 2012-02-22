#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

#define DIM 2
main(int argc, char **argv)
{
  int i, j, rank, nprocs;
  int g_A, local_A[6][6], dims[DIM]={6,6}, lo[DIM], hi[DIM], ld=6;

  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  for(i=0; i<6; i++)
    for(j=0; j<6; j++)local_A[i][j]=i+j+rank;

  for(i=0; i<6; i++)
    {
      for(j=0; j<6; j++)printf("%d ",local_A[i][j]);
      printf("\n");
    }

  g_A=NGA_Create(C_INT, 2, dims, "array_A", NULL);
  GA_Zero(g_A);
  GA_Print(g_A);

  GA_Print_distribution(g_A);
  NGA_Distribution(g_A, rank, lo, hi);
  GA_Sync(); 

  NGA_Put(g_A, lo, hi, local_A, &ld);
  GA_Print(g_A);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
