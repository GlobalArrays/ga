#include "mpi.h"
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NLEN 1024*1024
#define MPI_TAG 12383

int main(int argc, char **argv)
{
  int me, nprocs, iproc;
  int *sbuf, *rbuf, *gpuBuf, *devbuf;
  void *vbuf;
  int nghbr, sndr;
  int *hostIDs;
  int i;
  int lowest;
  int ngpu, totgpu, myGPU;
  cudaIpcMemHandle_t *handles;
  cudaIpcMemHandle_t handle;
  MPI_Request req;
  MPI_Status status;
  int my_master;
  int *masters;
  int t_ok, ok;

  /* initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  /* Find host IDs for all processors */
  hostIDs = (int*)malloc(sizeof(int)*nprocs);
  sbuf = (int*)malloc(sizeof(int)*nprocs);
  for (i=0; i<nprocs; i++) sbuf[i] = 0;
  sbuf[me] = gethostid();
  MPI_Allreduce(sbuf,hostIDs,nprocs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

  /* Find the lowest rank with same host ID */
  lowest = nprocs;
  for (i=0; i<nprocs; i++) {
    if (hostIDs[i] == hostIDs[me] && i<lowest) lowest = i;
  }
  /* find the highest rank with the same host ID */
  for (i=me; i<nprocs; i++) {
    if (hostIDs[i] == hostIDs[me]) my_master = i;
    else break;
  }
  /* create global list of masters for each process */
  masters = (int*)malloc(sizeof(int)*nprocs);
  for (i=0; i<nprocs; i++) sbuf[i] = 0;
  sbuf[me] = my_master;
  MPI_Allreduce(sbuf,masters,nprocs,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  free(sbuf);


  /* find total number of GPUs visible on each node. Check to see if this
   * is greater than or equal to number of processors on each node minus one */ 
  cudaGetDeviceCount(&ngpu);
  t_ok = 0;
  if (ngpu >= my_master-lowest) t_ok = 1;
  MPI_Allreduce(&t_ok,&ok,1,MPI_INT,MPI_PROD,MPI_COMM_WORLD);
  if (!ok) {
    if (me == 0) printf("Not enough GPUs per node\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  /* if not a master, allocate memory on GPU */
  if (me != my_master) {
    myGPU = me-lowest;
    cudaSetDevice(myGPU);
    cudaMalloc(&vbuf, sizeof(int)*NLEN);
    gpuBuf = (int*)vbuf;
  } else {
    myGPU = -1;
  }
  handles = (cudaIpcMemHandle_t*)malloc(sizeof(cudaIpcMemHandle_t)*nprocs);
  cudaIpcGetMemHandle(&handle,gpuBuf);
  MPI_Allgather(&handle,sizeof(cudaIpcMemHandle_t),MPI_BYTE,handles,
      sizeof(cudaIpcMemHandle_t),MPI_BYTE,MPI_COMM_WORLD);

  /* initialize send buf */
  sbuf = (int*)malloc(sizeof(int)*NLEN);
  if (me != my_master-1 && me != my_master) {
    nghbr = (me+1)%nprocs;
    for (i=0; i<NLEN; i++) {
      sbuf[i] = ((me+1)%nprocs)*NLEN+i;
    }
  } else if (me != my_master) {
    nghbr = (me+2)%nprocs;
    for (i=0; i<NLEN; i++) {
      sbuf[i] = ((me+2)%nprocs)*NLEN+i;
    }
  } else  {
    nghbr = -1;
  }
  printf("p[%d] nghbr: %d\n",me,nghbr);

  /* post GPU-aware Irecv, if necessary. */
  if (me == my_master) {
    int id;
    if (lowest > 0) {
      sndr = lowest-2;
    } else {
      sndr = nprocs-2;
    }
    id = 0;
    cudaSetDevice(id);
    printf("p[%d] opening handle on %d, receiving from %d\n",me,lowest,sndr);
    fflush(stdout);
    cudaIpcOpenMemHandle(&vbuf,handles[lowest],cudaIpcMemLazyEnablePeerAccess);
    devbuf = (int*)vbuf;
    cudaDeviceSynchronize();
    MPI_Irecv(devbuf,NLEN,MPI_INT,sndr,MPI_TAG,MPI_COMM_WORLD,&req);
  }
  /* send data to neighbor */
  if (nghbr != -1 && hostIDs[me] == hostIDs[nghbr]) {
    /* Data is on the same SMP node so use a memcpy */
    int id = nghbr-lowest;
    cudaSetDevice(id);
    cudaIpcOpenMemHandle(&vbuf,handles[nghbr],cudaIpcMemLazyEnablePeerAccess);
    devbuf = (int*)vbuf;
    cudaDeviceSynchronize();
    cudaMemcpy(devbuf,sbuf,sizeof(int)*NLEN,cudaMemcpyHostToDevice);
    cudaIpcCloseMemHandle(vbuf);
    cudaDeviceSynchronize();
  } else if (nghbr != -1) {
    printf("p[%d] posting message to %d\n",me,masters[nghbr]);
    fflush(stdout);
    /* Data is on different SMP node so use MPI */
    MPI_Send(sbuf,NLEN,MPI_INT,masters[nghbr],MPI_TAG,MPI_COMM_WORLD);
  }
  printf("p[%d] Posting wait\n",me);
  fflush(stdout);

  /* wait on Irecv, if necessary */
  if (me == my_master) {
    int id = 0;
    MPI_Wait(&req, &status);
    cudaIpcCloseMemHandle(devbuf);
    cudaDeviceSynchronize();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  printf("p[%d] Completed wait\n",me);
  fflush(stdout);

  /* check for correctness */
  t_ok = 1;
  if (me != my_master) {
    /* copy data from GPU to host */
    int id = me-lowest;
    cudaSetDevice(id);
    cudaMemcpy(sbuf,gpuBuf,sizeof(int)*NLEN,cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for (i=0; i<NLEN; i++) {
      if (sbuf[i] != me*NLEN+i) t_ok = 0;
    }
  }
  MPI_Allreduce(&t_ok,&ok,1,MPI_INT,MPI_PROD,MPI_COMM_WORLD);
  if (!ok) {
    if (me == 0) printf("Error moving data\n");
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  if (ok && me == 0) {
    printf("Completed test\n");
    fflush(stdout);
  }


  MPI_Finalize();

  return 0;
}
