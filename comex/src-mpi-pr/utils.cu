#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <cuda_runtime.h>

/* avoid name mangling by the CUDA compiler */
extern "C" {

/* return the number of GPUs visible from this processor*/
int numDevices()
{
  int ngpus;
  cudaGetDeviceCount(&ngpus);
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
  cudaPointerAttributes attr;
  cudaError_t err;
  cudaMalloc(buf, (int)size);
  printf("cudaMalloc buffer: %p\n",*buf);
  err = cudaPointerGetAttributes(&attr, *buf);
  printf("mallocDevice buf: %p device: %d\n",attr.devicePointer,attr.device);
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
 * return 1 is data is only located on host, 0 otherwise
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

};
