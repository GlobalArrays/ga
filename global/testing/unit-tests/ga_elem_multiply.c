/*
 * Test Program for GA
 * This is to test GA_Elem_multiply (is a collective operation)
 * GA_Create -- used to create a global array using handles like 'g_A' 
 * GA_Elem_multiply -- used to multiply two array and store it in one 
 */

#include<stdio.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

#define DIM 2

main(int argc, char **argv)
{
  int rank, nprocs, i, j;
  int g_A, g_B, g_C, local_C[DIM][DIM]; 
  int dims[DIM]={5,5}, dims2[DIM], ndim, type;
  int val_A=5, val_B=3, lo[DIM]={2,2}, hi[DIM]={4,4}, ld=DIM;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MA_init(C_INT, 1000, 1000);

  GA_Initialize();
  
  g_A = NGA_Create(C_INT, DIM, dims, "array_A", NULL);
  g_B = NGA_Create(C_INT, DIM, dims, "array_B", NULL);
  g_C = NGA_Create(C_INT, DIM, dims, "array_C", NULL);

  GA_Fill(g_A, &val_A);
  GA_Fill(g_B, &val_B);

  GA_Elem_multiply(g_A, g_B, g_C);
  GA_Print(g_C);
  GA_Sync();

  NGA_Get(g_C, lo, hi, local_C, &ld);
  if(rank==1)
    {
      /*
      for(i=0; i<DIM; i++)
	{
	  for(j=0; j<DIM; j++)printf("%d ", local_C[i][j]);
	  printf("\n");
	}
      */
      for(i=0; i<DIM; i++)
	{
	  for(j=0; j<DIM; j++)
	    if(local_C[i][j]!=val_A*val_B) printf("GA Error : \n");
	}
    }
      
  if(rank == 0)
    printf("Test Completed \n");

  GA_Terminate();
  MPI_Finalize();

}
