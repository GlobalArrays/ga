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
  void *ptr_u;
  void *ptr_d;
  int nghbr;
  void *nptr;
  int *iptr;
  int ok;
  cudaIpcMemHandle_t *exchng_u;
  cudaIpcMemHandle_t *exchng_d;
  cudaIpcMemHandle_t handle_u;
  cudaIpcMemHandle_t handle_d;

  MPI_Init(&argc, &argv);


  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

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
  cudaMalloc(&ptr_u, size);
  cudaMalloc(&ptr_d, size);
  printf("p[%d] pointer_u %p on device %d\n",rank,ptr_u,rank);
  printf("p[%d] pointer_d %p on device %d\n",rank,ptr_d,rank);
  cudaMemset(ptr_u, 0, size);
  cudaMemset(ptr_d, 0, size);
  printf("p[%d] allocated %d bytes\n",rank,size);

  /* Get cudaIpcMemHandle_t object for allocation */
  cudaIpcGetMemHandle(&handle_u, ptr_u);
  cudaIpcGetMemHandle(&handle_d, ptr_d);

  /* Exchange handles with other processors */
  exchng_u = (cudaIpcMemHandle_t*)malloc(nprocs*sizeof(cudaIpcMemHandle_t));
  exchng_d = (cudaIpcMemHandle_t*)malloc(nprocs*sizeof(cudaIpcMemHandle_t));
  size = sizeof(cudaIpcMemHandle_t);
  MPI_Allgather(&handle_u, size, MPI_BYTE, exchng_u, size, MPI_BYTE, MPI_COMM_WORLD);
  MPI_Allgather(&handle_d, size, MPI_BYTE, exchng_d, size, MPI_BYTE, MPI_COMM_WORLD);
  printf("p[%d] exchanged handles size: %d\n",rank,size);

  /* Get pointer on process with next lowest rank */
  if (rank > 0) {
    nghbr = rank-1;
  } else {
    nghbr = nprocs-1;
  }
  if (nghbr > 0) {
    ilo = nghbr*BLOCKSIZE;
    ihi = (nghbr+1)*BLOCKSIZE-1;
  } else {
    ilo = (nprocs-1)*BLOCKSIZE;
    ihi = nprocs*BLOCKSIZE-1;
  }
  printf("p[%d] open neighbor below %d\n",rank,nghbr);
  cudaIpcOpenMemHandle(&nptr,exchng_d[nghbr],cudaIpcMemLazyEnablePeerAccess);
  printf("p[%d] below neigbor pointer %p on device %d\n",rank,nptr,nghbr);

  /* write data to neighbor */
  size = BLOCKSIZE*sizeof(int);
  printf("p[%d] size: %d\n",rank,size);
  iptr = (int*)malloc(size);
  for (i=ilo; i<=ihi; i++) iptr[i-ilo]=i;
  printf("p[%d] copy data to neighbor\n",rank);
  cudaMemcpy(nptr,iptr,size,cudaMemcpyHostToDevice);
  printf("p[%d] close neighbor\n",rank);
  cudaIpcCloseMemHandle(nptr);

  /* Get pointer on process with next highest rank */
  if (rank < nprocs-1) {
    nghbr = rank+1;
  } else {
    nghbr = 0;
  }
  ilo = nghbr*BLOCKSIZE;
  ihi = (nghbr+1)*BLOCKSIZE-1;
  printf("p[%d] open neighbor %d\n",rank,nghbr);
  cudaIpcOpenMemHandle(&nptr,exchng_u[nghbr],cudaIpcMemLazyEnablePeerAccess);
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
  cudaMemcpy(iptr,ptr_u,size,cudaMemcpyDeviceToHost);
  ilo = rank*BLOCKSIZE;
  ihi = (rank+1)*BLOCKSIZE-1;
  ok = 1;
  for (i=ilo; i<=ihi; i++) {
    if (iptr[i-ilo] != i) ok = 0;
  }
  printf("p[%d] completed local check for upper test %d\n",rank,ok);
  MPI_Allreduce(&ok, &i, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
  if (rank == 0) {
    if (i == 0) {
      printf("Test up failed\n");
    } else {
      printf("Test up succeeded\n");
    }
  }

  for (i=0; i<BLOCKSIZE; i++) iptr[i] = 0;
  cudaMemcpy(iptr,ptr_d,size,cudaMemcpyDeviceToHost);
  ilo = rank*BLOCKSIZE;
  ihi = (rank+1)*BLOCKSIZE-1;
  ok = 1;
  for (i=ilo; i<=ihi; i++) {
    if (iptr[i-ilo] != i) ok = 0;
  }
  printf("p[%d] completed local check for lower test %d\n",rank,ok);
  MPI_Allreduce(&ok, &i, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
  if (rank == 0) {
    if (i == 0) {
      printf("Test down failed\n");
    } else {
      printf("Test down succeeded\n");
    }
  }

  printf("p[%d] free buffers\n",rank);
  free(exchng_u);
  free(exchng_d);
  free(iptr);
  MPI_Finalize();
}
