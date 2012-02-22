/*
This program to automate lo_ and hi_ size; each process copies local array to corresponding global array memory space. 

Here both arrays are of same size.

get the values from global array, by over-lapping. to local array B (of local size)

Here each process does it own put and get

 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

#define DIM 2
#define GSIZE 12

int main(int argc, char **argv)
{
  int i, j, g_A, dims[DIM]={GSIZE,GSIZE}, nprocs, rank;
  int lsize, local_A[GSIZE][GSIZE], local_B[GSIZE][GSIZE], lo[DIM], hi[DIM];
  int lo1[DIM], hi1[DIM], row, col, row1, col1, ld=GSIZE, offset_one, offset_two;
  ga_nbhdl_t nb_get;

  //printf(" check \n ");
  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  
  lsize = GSIZE/sqrt(nprocs);
  row = rank / (int)sqrt(nprocs);
  col = rank % (int)sqrt(nprocs);

  lo[0] = row*lsize;
  lo[1] = col*lsize;

  hi[0] = lo[0] + (lsize-1);
  hi[1] = lo[1] + (lsize-1);

  for(i=0; i<GSIZE; i++)
    for(j=0; j<GSIZE; j++)
      local_A[i][j]= rand()%lsize + rank;

  g_A = NGA_Create(C_INT, DIM, dims, "array_A", NULL);
  GA_Zero(g_A);
  //GA_Print(g_A);

  /*  if(rank==0) {
    for(i=0; i<GSIZE; i++){
      for(j=0; j<GSIZE; j++)
	printf("%d ",local_A[i][j]);
      printf("\n ");
    }
    
    for(i=0; i<nprocs; i++) {
      
      row1 = i / (int)sqrt(nprocs);
      col1 = i % (int)sqrt(nprocs);
      
      lo1[0] = row1*lsize;
      lo1[1] = col1*lsize;
      
      hi1[0] = lo1[0] + (lsize-1);
      hi1[1] = lo1[1] + (lsize-1);
      
      printf("\n");
      NGA_Put(g_A, lo1, hi1, &local_A[lo1[0]][lo1[1]], &ld);
    }
  }
  */

  if(rank==0)
    {
      NGA_NbGet(g_A, lo, hi, local_B, &ld, &nb_get);
      NGA_Put(g_A, lo, hi, &local_A[lo[0]][lo[1]], &ld);
      
      for(i=0; i<GSIZE; i++){
	for(j=0; j<GSIZE; j++)
	  printf("%d ",local_A[i][j]);
	printf("\n ");
      }

      NGA_NbWait(&nb_get);
    }
  if(rank==0)
    {
      for(i=0; i<lsize; i++){
	for(j=0; j<lsize; j++)
	  printf("%d ",local_B[i][j]);
	printf("\n ");
      }
    }
  offset_one=0;
  offset_two=1;
  /*
  do{
    if(rank==offset_one)
      {
	NGA_NbWait(&nb_get);
	printf("%d-- check -- %d\n  ", rank, offset_one);
      }
    if(rank==offset_two)
      {
	NGA_NbGet(g_A, lo, hi, local_B, &ld, &nb_get);
	printf("%d ***** check ****** %d\n  ", rank, offset_two);
	for(i=0; i<lsize; i++){
	  for(j=0; j<lsize; j++)
	    printf("%d ",local_B[i][j]);
	  printf("\n ");
	}
	NGA_Put(g_A, lo, hi, &local_A[lo[0]][lo[1]], &ld);

	offset_one=offset_two;
	offset_two++;
	printf(" check \n ");
      }
  }while(offset_two<nprocs);
  */
  GA_Print(g_A);
  GA_Terminate();
  MPI_Finalize();
  return 0;
}
