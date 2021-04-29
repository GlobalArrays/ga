#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*#include "cuda.h"*/
#include "mpi.h"

#include <cuda_runtime.h>

#define BLOCKSIZE 4

int main(int argc, char **argv) {

  int rank, nprocs;
  int ilo, ihi;
  int ndev;
  int i, size;
  void *ptr;
  int nghbr;
  void *nptr;
  int *iptr;
  int ok;
  cudaIpcMemHandle_t *exchng;
  cudaIpcMemHandle_t handle;

  MPI_Init(&argc, &argv);


  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  if (rank > 0) {
    ilo = (rank-1)*BLOCKSIZE;
    ihi = rank*BLOCKSIZE-1;
  } else {
    ilo = (nprocs-1)*BLOCKSIZE;
    ihi = nprocs*BLOCKSIZE-1;
  }

  /* Check to see if number of processors equals the number of GPUs */
  cudaGetDeviceCount(&ndev);
  if (ndev != nprocs) {
    printf("Number of procs %d must equal number of devices %d\n",nprocs,ndev);
    return 1;
  }
  printf("p[%d] Number of devices is %d\n",rank,ndev);

  /* Allocate a segment on each GPU and initial to zero */
  size = BLOCKSIZE*sizeof(int);
  cudaSetDevice(rank);
  cudaMalloc(&ptr, size);
  printf("p[%d] pointer %p on device %d\n",rank,ptr,rank);
  cudaMemset(ptr, 0, size);
  printf("p[%d] allocated %d bytes\n",rank,size);

  /* Get cudaIpcMemHandle_t object for allocation */
  cudaIpcGetMemHandle(&handle, ptr);

  /* Exchange handles with other processors */
  exchng = (cudaIpcMemHandle_t*)malloc(nprocs*sizeof(cudaIpcMemHandle_t));
  size = sizeof(cudaIpcMemHandle_t);
  MPI_Allgather(&handle, size, MPI_BYTE, exchng, size, MPI_BYTE, MPI_COMM_WORLD);
  printf("p[%d] exchanged handles size: %d\n",rank,size);

  /* Get pointer on process with next lowest rank */
  if (rank > 0) {
    nghbr = rank-1;
  } else {
    nghbr = nprocs-1;
  }
  printf("p[%d] open neighbor %d\n",rank,nghbr);
  cudaIpcOpenMemHandle(&nptr,exchng[nghbr],cudaIpcMemLazyEnablePeerAccess);
  printf("p[%d] neigbor pointer %p on device %d\n",rank,nptr,nghbr);

  /* write data to neighbor */
  size = BLOCKSIZE*sizeof(int);
  printf("p[%d] size: %d\n",rank,size);
  iptr = (int*)malloc(size);
  for (i=ilo; i<=ihi; i++) iptr[i-ilo]=i;
  printf("p[%d] copy data to neighbor\n",rank);
  cudaMemcpy(nptr,iptr,size,cudaMemcpyHostToDevice);
  printf("p[%d] close neighbor\n",rank);
  cudaIpcCloseMemHandle(nptr);

  printf("p[%d] call barrier\n",rank);
  /* Synchronize everything */
  MPI_Barrier(MPI_COMM_WORLD);
  /* check data */
  printf("p[%d] copy data to buffer\n",rank);
  for (i=0; i<BLOCKSIZE; i++) iptr[i] = 0;
  cudaMemcpy(iptr,ptr,size,cudaMemcpyDeviceToHost);
  ilo = rank*BLOCKSIZE;
  ihi = (rank+1)*BLOCKSIZE-1;
  ok = 1;
  for (i=ilo; i<=ihi; i++) {
    if (iptr[i-ilo] != i) ok = 0;
  }
  printf("p[%d] completed local check %d\n",rank,ok);
  MPI_Allreduce(&ok, &i, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
  if (rank == 0) {
    if (i == 0) {
      printf("Test failed\n");
    } else {
      printf("Test succeeded\n");
    }
  }

  printf("p[%d] free buffers\n",rank);
  free(exchng);
  free(iptr);
  MPI_Finalize();
}
