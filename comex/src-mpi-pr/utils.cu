#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <cuda_runtime.h>

/* avoid name mangling by the CUDA compiler */
extern "C" {

/* return the number of GPUs visible from this processor */
int numDevices()
{
  int ngpus;
  cudaGetDeviceCount(&ngpus);
  return ngpus;
}

/* set the GPU device for this processor */
void setDevice(int id)
{
  cudaSetDevice(id);
}

/* allocate a unified memory buffer */
void mallocDevice(void **buf, size_t size)
{
  cudaMallocManaged(buf, (int)size);
}

/* free unified memory */
void freeDevice(void *buf)
{
  cudaFree(buf);
}

/* copy data from host buffer to unified memory */
void copyToDevice(void *hostptr, void *devptr, int bytes)
{
  cudaMemcpy(devptr, hostptr, bytes, cudaMemcpyHostToDevice); 
}

/* copy data from unified memory to host buffer */
void copyToHost(void *hostptr, void *devptr, int bytes)
{
  cudaMemcpy(hostptr, devptr, bytes, cudaMemcpyDeviceToHost); 
}

};
