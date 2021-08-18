#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

#include <cuda_runtime.h>
#include "dev_mem_handle.h"

/* avoid name mangling by the CUDA compiler */
extern "C" {

/* return the number of GPUs visible from this processor*/
int numDevices()
{
  int ngpus;
  cudaError_t err;
  err = cudaGetDeviceCount(&ngpus);
  /*cuDeviceGetCount(&ngpus); */
  if (err != cudaSuccess) {
    printf("Error encountered by cudaGetDeviceCount\n");
  }
  return ngpus;
}

/* set the GPU device for this processor
 * id: id of device
 */
void setDevice(int id)
{
  cudaSetDevice(id);
}

/* allocate a unified memory buffer
 * buf: pointer to buffer
 * size: size of allocation in bytes
 */
void mallocDevice(void **buf, size_t size)
{
  cudaMalloc(buf, (int)size);
}

/* free unified memory
 * buf: pointer to memory allocation
 */
void freeDevice(void *buf)
{
  cudaFree(buf);
}

/* copy data from host buffer to unified memory
 * hostptr: pointer to allocation on host
 * devptr: pointer to allocation on device
 * bytes: number of bytes to copy
 */
void copyToDevice(void *hostptr, void *devptr, int bytes)
{
  cudaMemcpy(devptr, hostptr, bytes, cudaMemcpyHostToDevice); 
}

/* copy data from unified memory to host buffer
 * hostptr: pointer to allocation on host
 * devptr: pointer to allocation on device
 * bytes: number of bytes to copy
 */
void copyToHost(void *hostptr, void *devptr, int bytes)
{
  cudaMemcpy(hostptr, devptr, bytes, cudaMemcpyDeviceToHost); 
}

/* copy data between devices using unified memory
 * srcptr: source pointer
 * dstptr: destination pointer
 * bytes: number of bytes to copy
 */
void copyDevToDev(void *srcptr, void *dstptr, int bytes)
{
  cudaMemcpy(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice); 
}

/**
 * set values on the device to a specific value
 * ptr: pointer to device memory that needs to be set
 * val: integer representation of the value of each byte
 * size: number of bytes that should be set
 */
void deviceMemset(void *ptr, int val, size_t bytes)
{
  cudaMemset(ptr, val, bytes);
}

/* is pointer located on host?
 * return 1 data is located on host, 0 otherwise
 * ptr: pointer to data
 */
int isHostPointer(void *ptr)
{
  cudaPointerAttributes attr;
  cudaError_t  err = cudaPointerGetAttributes(&attr, ptr);
  /* Assume that if Cuda doesn't know anything about the pointer, it is on the
   * host */
  if (err != cudaSuccess) return 1;
  if (attr.devicePointer == NULL) {
    return  1;
  }
  return 0;
}

__global__ void iaxpy_kernel(int *dst, const int *src, int scale)
{
  int i = threadIdx.x;

  dst[i] = dst[i] + scale*src[i];
}

void deviceIaxpy(int *dst, const int *src, const int *scale, int n)
{
  iaxpy_kernel<<<1,n>>>(dst, src, *scale);
}

__global__ void inc_int_kernel(int *target, const int *inc)
{
  int i = threadIdx.x;
  target[i] += inc[i];
}

void deviceAddInt(int *ptr, const int inc)
{
  void *buf;
  void *ibuf = (void*)(&inc);
  cudaMalloc(&buf, sizeof(int));
  copyToDevice(ibuf, buf, sizeof(int));  
  inc_int_kernel<<<1,1>>>(ptr, (int*)buf);
  cudaFree(buf);
}

__global__ void inc_long_kernel(long *target, const long *inc)
{
  int i = threadIdx.x;
  target[i] += inc[i];
}

void deviceAddLong(long *ptr, const long inc)
{
  void *buf;
  void *lbuf = (void*)(&lbuf);
  cudaMalloc(&buf, sizeof(long));
  copyToDevice(lbuf, buf, sizeof(long));  
  inc_long_kernel<<<1,1>>>(ptr, (long*)buf);
  cudaFree(buf);
}

int deviceGetMemHandle(devMemHandle_t *handle, void *memory)
{
  return cudaIpcGetMemHandle(&handle->handle, memory);
}

int deviceOpenMemHandle(void **memory, devMemHandle_t handle)
{
 return cudaIpcOpenMemHandle(memory, handle.handle,cudaIpcMemLazyEnablePeerAccess);
}

int deviceCloseMemHandle(void *memory)
{
 return cudaIpcCloseMemHandle(memory);
}

};
