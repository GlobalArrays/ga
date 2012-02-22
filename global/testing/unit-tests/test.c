#include<stdio.h>

#include"mpi.h"
#include"ga.h"

int main(int argv, char **argc)
{
  int rank, nprocs;
  MPI_Init(&argv, &argc);
  GA_Initialize();

  MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if(rank==0)
    printf("Total number of processes running: %d\n", nprocs);

  if(rank==0)
    printf("rank %d: Hello, World\n", rank);
  else
    printf("rank %d: Hello, Universe\n", rank);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
