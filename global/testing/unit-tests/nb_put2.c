#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

main(int argc, char **argv)
{
  int g_A, dims[2]={5,5}, lo[2]={2,2}, hi[2]={3,3}, local_A[2][2], ld=2, value=1;
  ga_nbhdl_t nbhandle, nbget;
  int rank, nprocs, i, j, lo2[2]={1,1}, hi2[2]={2,2}, lo3[2]={3,3}, hi3[2]={4,4};
 
  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  g_A= NGA_Create(C_INT, 2, dims, "array_A", NULL);
  GA_Fill(g_A, &value);
  GA_Sync();
  GA_Print(g_A);

  NGA_Get(g_A, lo2, hi2, local_A, &ld); 
  for(i=0; i<2; i++)
    for(j=0; j<2; j++)local_A[i][j]+=5;

  if(rank==0)
    {
      NGA_NbPut(g_A, lo, hi, local_A, &ld, &nbhandle);
      // do some dummy computation to overlap communication
      for(i=0; i<2; i++)
	{
	  for(j=0; j<2; j++)printf("%d ", local_A[i][j]);
	  printf("\n");
	}
      NGA_NbWait(&nbhandle);
    }
  
  //for(i=0; i<1000000; i++) j=i*2;
  GA_Sync();
  GA_Print(g_A);
    
  if(rank==1)
    {
      NGA_NbGet(g_A, lo, hi, local_A, &ld, &nbget);
      NGA_NbWait(&nbget);
      printf(" The output for the process -- %d \n", rank);
      for(i=0; i<2; i++)
	{
	  for(j=0; j<2; j++)printf("%d ", local_A[i][j]);
	  printf("\n");
	}
    }
  GA_Sync();
  GA_Print(g_A);

  GA_Destroy(g_A);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
