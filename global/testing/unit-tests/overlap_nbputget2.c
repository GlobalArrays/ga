/*
This program to automate lo_ and hi_ size;

trying to implement and test the _nb_put in this example.

Here both arrays are of same size.(G_ and local_A)

get the values from global array, by over-lapping. to local array B (of local size)

overlapping is done by changing lo_ and hi_ values

Here process 1(rank 0) does put(from local_A) and then get( to local_B) value from G-array

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
  int i, j, p, q, g_A, dims[DIM]={GSIZE,GSIZE}, nprocs, rank;
  int lsize, local_A[GSIZE][GSIZE], local_B[GSIZE][GSIZE], lo[DIM], hi[DIM];
  int lo1[DIM], hi1[DIM], row, col, row1, col1, ld=GSIZE;
  int lo_prev[DIM], hi_prev[DIM], row_prev, col_prev;
  ga_nbhdl_t nb_get, nb_put;

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

  g_A = NGA_Create(C_INT, DIM, dims, "array_A", NULL);
  GA_Zero(g_A);
  //GA_Print(g_A);

  if(rank==0) {

    for(i=0; i<GSIZE; i++)
      for(j=0; j<GSIZE; j++)
	local_A[i][j]= rand()%lsize + rank;

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

      if(i==0)
	{  
    	  NGA_Put(g_A, lo1, hi1, &local_A[lo1[0]][lo1[1]], &ld);  
	  printf("%d ---- check \n", i);
	}
      
      else
	{
	    NGA_NbPut(g_A, lo1, hi1, &local_A[lo1[0]][lo1[1]], &ld, &nb_put);     
	    
	    NGA_Get(g_A, lo_prev, hi_prev, local_B, &ld);
	  
	    for(p=0; p<lsize; p++){
	      for(q=0; q<lsize; q++)
		printf("%d ",local_B[p][q]);
	      printf("\n ");
	    }
	    
	    NGA_NbWait(&nb_put);
	    printf(" %d *******\n", i);
	}
      lo_prev[0] = lo1[0];      
      lo_prev[1] = lo1[1];
      hi_prev[0] = hi1[0];
      hi_prev[1] = hi1[1];
    }
    
    NGA_Get(g_A, lo_prev, hi_prev, local_B, &ld);
    for(p=0; p<lsize; p++){
      for(q=0; q<lsize; q++)
	printf("%d ",local_B[p][q]);
      printf("\n ");
    }
    
    printf(" ****** %d *******\n", i);
    //NGA_Put(g_A, lo, hi, &local_A[lo[0]][lo[1]], &ld);
  }
  
  GA_Print(g_A);        
  GA_Terminate();
  MPI_Finalize();
  return 0;
}
