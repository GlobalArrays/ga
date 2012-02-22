/*
This program to automate lo_ and hi_ size;

trying to implement and test the _nb_put in this example.

Here both arrays are of same size.(G_ and local_A)

get the values from global array, by over-lapping. to local array B (of local size)

Here each process 1(rank 0) does put(from local_A) and then rest get( to local_B) value from G-array

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
  int i, j, g_A, dims[DIM]={GSIZE,GSIZE}, nprocs, rank, p, q, k=0, l=0;
  int lsize, local_A[GSIZE][GSIZE], local_B[GSIZE][GSIZE], lo[DIM], hi[DIM];
  int lo1[DIM], hi1[DIM], row, col, row1, col1, ld=GSIZE;
  int offset_one, offset_two, process_number, lock;
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
    
    // below content is for refrence    

    /*
    for(i=0; i<nprocs; i++) {
      
      row1 = i / (int)sqrt(nprocs);
      col1 = i % (int)sqrt(nprocs);
      
      lo1[0] = row1*lsize;
      lo1[1] = col1*lsize;
      
      hi1[0] = lo1[0] + (lsize-1);
      hi1[1] = lo1[1] + (lsize-1);
      
      printf("\n");

      

    i=0;
    if(i==0)
      {
	NGA_NbPut(g_A, lo1, hi1, &local_A[lo1[0]][lo1[1]], &ld, &nb_put);  
	printf(" check \n");
      }

    offset_one = 0;
    offset_two = 1;
    process_number = 0;
    do{
      if(process_number == offset_one)
	{
	  i = process_number;
	  NGA_NbWait(&nb_put);
	  process_number += 1;
	}
      if(process_number == offset_two)
	{
	  i = process_number;
	  NGA_NbPut(g_A, lo1, hi1, &local_A[lo1[0]][lo1[1]], &ld, &nb_put);  
	  i = process_number-1;
	  NGA_Get(g_A, lo1, hi1, local_B, &ld);
	  
	  for(p=0; p<lsize; p++){
	    for(q=0; q<lsize; q++)
	      printf("%d ",local_B[p][q]);
	    printf("\n ");
	  }
	  printf(" %d\n", k);
	  k++;
	  offset_one = offset_two;
	  offset_two += 1; 
	}
    }while(offset_two <= nprocs);
    
    }
    */

    // above is the original content

    offset_one = 0;
    offset_two = 1;
    process_number = 0;
    
    for(i=0; i<nprocs; i++) {
      
      row1 = i / (int)sqrt(nprocs);
      col1 = i % (int)sqrt(nprocs);
      
      lo1[0] = row1*lsize;
      lo1[1] = col1*lsize;
      
      hi1[0] = lo1[0] + (lsize-1);
      hi1[1] = lo1[1] + (lsize-1);
      
      
      if(i==0)
	{
	  NGA_NbPut(g_A, lo1, hi1, &local_A[lo1[0]][lo1[1]], &ld, &nb_put);  
	  printf("%d ---- check \n", i);
	}
      
      printf("%d ******* check/ before do_ loop \n", i);
      do{
	if(i == offset_one)
	  {
	    NGA_Get(g_A, lo1, hi1, local_B, &ld);
	  
	    for(p=0; p<lsize; p++){
	      for(q=0; q<lsize; q++)
		printf("%d ",local_B[p][q]);
	      printf("\n ");
	    }
	    
	    NGA_NbWait(&nb_put);
	    printf(" %d ---%d\n", i, l);
	    lock=offset_one;
	  }
	if(i == offset_two)
	  {
	    NGA_NbPut(g_A, lo1, hi1, &local_A[lo1[0]][lo1[1]], &ld, &nb_put);  
	    // i = process_number-1;
	    //i = i-1;
	    //NGA_Get(g_A, lo1, hi1, local_B, &ld);
	  
	    /*for(p=0; p<lsize; p++){
	      for(q=0; q<lsize; q++)
		printf("%d ",local_B[p][q]);
	      printf("\n ");
	    }
	    */
	    printf(" %d *******%d\n", i, k);
	    k++;
	    offset_one = offset_two;
	    offset_two += 1; 
	  }
	printf(" %d *******%d -- %d--- %d\n", i,offset_one, offset_two, lock);
      }while(i < lock);
      
    }
    
    //NGA_Put(g_A, lo, hi, &local_A[lo[0]][lo[1]], &ld);
  }
  
  GA_Print(g_A);        
  
  GA_Terminate();
  MPI_Finalize();
  return 0;
}
