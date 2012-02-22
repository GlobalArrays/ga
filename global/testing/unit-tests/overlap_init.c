#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

#define DIM 2
#define GSIZE 12
#define LSIZE 4 
#define ZERO 0

main(int argc, char **argv)
{
  int i, j, g_A, dims[DIM]={GSIZE,GSIZE}, nprocs, rank;
  int lsize, local_A[lsize][lsize], lo[DIM], hi[DIM];
  int lo1[2], hi1[2], row, col;

  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  lsize = GSIZE/sqrt(nprocs);
  row = rank / (int)sqrt(nprocs);
  col = rank % (int)sqrt(nprocs);

  lo1[0] = row*lsize;
  lo1[1] = col*lsize;

  hi1[0] = lo1[0] + (lsize-1);
  hi1[1] = lo1[1] + (lsize-1);

  printf("Process -- %d --(%d:%d , %d:%d)\n", rank, lo1[0], hi1[0], lo1[1], hi1[1]);

  g_A=NGA_Create(C_INT, DIM, dims, "array_A", NULL);
  GA_Zero(g_A);
  GA_Print(g_A);

  GA_Print_distribution(g_A);
  GA_Terminate();
  MPI_Finalize();
  return 0;
}
