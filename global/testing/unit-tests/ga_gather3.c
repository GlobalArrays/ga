/*                                                                                                       
 * Test Program for GA                                                                                   
 * This is to test GA_Gather (is a one-sided operation)                                                 
 * GA_Create -- used to create a global array using handles like 'g_A'                                   
 * GA_gather -- used to get data from different location
 */

#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"
#include"ga_unit.h"

#define SIZE 10
#define N 5
#define D 2

int main(int argc, char **argv)
{
  int rank, nprocs;
  int g_A, dims[D]={SIZE,SIZE}, *local_A=NULL, *local_G=NULL, **sub_array=NULL, **s_array=NULL;
  int i, j, value=5;
  
  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  s_array=(int**)malloc(N*sizeof(int*));
  for(i=0; i<N; i++)
    {
      s_array[i]=(int*)malloc(D*sizeof(int));
      for(j=0; j<D; j++) s_array[i][j]=rand()%10;
    }

  sub_array=(int**)malloc(N*sizeof(int*));
  for(i=0; i<N; i++)
    {
      sub_array[i]=(int*)malloc(D*sizeof(int));
      for(j=0; j<D; j++) sub_array[i][j]=rand()%10;
    }

  for(i=0; i<N; i++) local_A=(int*)malloc(N*sizeof(int));

  for(i=0; i<N; i++) local_G=(int*)malloc(N*sizeof(int));
  
  g_A=NGA_Create(C_INT, D, dims, "array_A", NULL);
  GA_Fill(g_A, &value);
  GA_Sync();
                                                   
  NGA_Scatter(g_A, local_A, s_array, N);
  NGA_Gather(g_A, local_G, s_array, N);
  GA_Sync();
  GA_Print(g_A);

  if(rank==0)
    {
      for(i=0; i<N; i++)
	if(local_G[i]!=local_A[i]) printf("GA Error: \n");
    }
  GA_Sync();
  if(rank==0)
    GA_PRINT_MSG();

  GA_Terminate();
  MPI_Finalize();
  return 0;
}

