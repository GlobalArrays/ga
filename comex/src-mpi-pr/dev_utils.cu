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

#if 1
__global__ void iaxpy_kernel(int *dst, const int *src, int scale, int n)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  int i;
  for (i=index; i<n; i += stride) {
    dst[i] = dst[i] + scale*src[i];
  }
}
#else
__global__ void iaxpy_kernel(int *dst, const int *src, int scale)
{
  int i = threadIdx.x;

  dst[i] = dst[i] + scale*src[i];
}
#endif

void deviceIaxpy(int *dst, int *src, const int *scale, int n)
{
#if 1
  int nblocks = (n+1023)/1024;
  iaxpy_kernel<<<nblocks,1024>>>(dst, src, *scale, n);
#else
#if 1
  int nblk = n/1024;
  int rmndr = n%1024;
  int i;
  int *lsrc = src;
  int *ldst = dst;
  for (i=0; i<nblk; i++) {
    iaxpy_kernel<<<1,1024>>>(ldst, lsrc, *scale);
    lsrc += 1024;
    ldst += 1024;
  }
  if (rmndr > 0) {
    iaxpy_kernel<<<1,rmndr>>>(ldst, lsrc, *scale);
  }
#else
  iaxpy_kernel<<<1,n>>>(dst, src, *scale);
#endif
#endif
}

#if 1
__global__ void laxpy_kernel(long *dst, const long *src, long scale, int n)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  int i;
  for (i=index; i<n; i += stride) {
    dst[i] = dst[i] + scale*src[i];
  }
}
#else
__global__ void laxpy_kernel(long *dst, const long *src, long scale)
{
  int i = threadIdx.x;

  dst[i] = dst[i] + scale*src[i];
}
#endif

void deviceLaxpy(long *dst, long *src, const long *scale, int n)
{
#if 1
  int nblocks = (n+1023)/1024;
  laxpy_kernel<<<nblocks,1024>>>(dst, src, *scale, n);
#else
#if 1
  int nblk = n/1024;
  int rmndr = n%1024;
  int i;
  long *lsrc = src;
  long *ldst = dst;
  for (i=0; i<nblk; i++) {
    laxpy_kernel<<<1,1024>>>(ldst, lsrc, *scale);
    lsrc += 1024;
    ldst += 1024;
  }
  if (rmndr > 0) {
    laxpy_kernel<<<1,rmndr>>>(ldst, lsrc, *scale);
  }
#else
  laxpy_kernel<<<1,n>>>(dst, src, *scale);
#endif
#endif
}

__global__ void inc_int_kernel(int *target, const int *inc)
{
  int i = threadIdx.x;
  target[i] += inc[i];
}

void deviceAddInt(int *ptr, const int inc)
{
#if 0
  void *buf;
  void *ibuf = (void*)(&inc);
  cudaMalloc(&buf, sizeof(int));
  copyToDevice(ibuf, buf, sizeof(int));  
  inc_int_kernel<<<1,1>>>(ptr, (int*)buf);
  cudaFree(buf);
#else
  int tmp;
  copyToHost(&tmp,ptr,sizeof(int));
  tmp += inc;
  copyToDevice(&tmp,ptr,sizeof(int));
#endif
}

__global__ void inc_long_kernel(long *target, const long *inc)
{
  int i = threadIdx.x;
  target[i] += inc[i];
}

void deviceAddLong(long *ptr, const long inc)
{
#if 0
  void *buf;
  void *lbuf = (void*)(&lbuf);
  cudaMalloc(&buf, sizeof(long));
  copyToDevice(lbuf, buf, sizeof(long));  
  inc_long_kernel<<<1,1>>>(ptr, (long*)buf);
  cudaFree(buf);
#else
  long tmp;
  copyToHost(&tmp,ptr,sizeof(long));
  tmp += inc;
  copyToDevice(&tmp,ptr,sizeof(long));
#endif
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
