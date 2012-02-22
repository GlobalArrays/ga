/*
 * Test Program for GA
 * This is to test GA_Zero(is a collective operation)
 * GA_Zero -- used to fill value zero into global array --- simple to make all values of GA as zero .., 
 */

#include<stdio.h>
#include<stdlib.h>
#include"mpi.h"
#include"ga.h"
#include"macdecls.h"
#include"ga_unit.h"

#define DIM 2
#define SIZE 5

main(int argc, char **argv)
{
  int rank, nprocs;
  int g_A, **local_A=NULL, local_B[SIZE][SIZE]; 
  int dims[DIM]={SIZE,SIZE}, lo[DIM]={0,0}, hi[DIM]={4,4}, ld=5, i, j;
  //  int dims[DIM]={SIZE,SIZE}, lo[DIM]={SIZE-SIZE,SIZE-SIZE}, hi[DIM]={SIZE-1,SIZE-1}, ld=5, i, j;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MA_init(C_INT, 1000, 1000);

  GA_Initialize();


  local_A=(int**)malloc(SIZE*sizeof(int*));
  for(i=0; i<SIZE; i++)
    {
      local_A[i]=(int*)malloc(SIZE*sizeof(int));
      for(j=0; j<SIZE; j++) local_A[i][j]=1;
    }
  
  g_A = NGA_Create(C_INT, DIM, dims, "array_A", NULL);
  GA_Zero(g_A);
  GA_Print(g_A);


  NGA_Get(g_A, lo, hi, local_B, &ld);

  if(rank == 0)
    {
      for(i=0; i<SIZE; i++)
	{
	  for(j=0; j<SIZE; j++)
	    if(local_B[i][j]!=0) GA_ERROR_MSG();
	}
    }

  GA_Sync();
  if(rank == 0)
    GA_PRINT_MSG();

  GA_Destroy(g_A);

  GA_Terminate();
  MPI_Finalize();
}

/* getting error on dynamic allocated array when in use --  finding report that why it genrates error*/
