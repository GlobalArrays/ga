#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

//#define GA_DATA_TYPE MP_F_INT

int main(int argv, char **argc)
{
  int g_A, local_A[5][5], i, j, ld=5;
  int value=5;
  int rank, nprocs;
  int dims[2]={10,10}, lo[2]={0,0}, hi[2]={4,4};

  MPI_Init(&argv, &argc);
  MA_init(MT_F_INT, 1000, 1000);
  GA_Initialize();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  g_A=NGA_Create(MT_F_INT, 2, dims, "array_A", NULL);

  GA_Fill(g_A, &value);
  GA_Print(g_A);
    
  if(rank==0)
    {
      NGA_Get(g_A, lo, hi, local_A, &ld);
      printf("Rank -- %d\n", rank);
    }
  printf("\n");
  GA_Sync();

  if(rank==1)
    {
      for(i=0; i<5; i++)
	{
	  for(j=0; j<5; j++)
	    printf("%d ", local_A[i][j]);
	  printf("\n");
	}
      printf("Rank -- %d\n", rank);
    }
      
  GA_Destroy(g_A);

  GA_Terminate();
  MPI_Finalize();

  return 0;
}
