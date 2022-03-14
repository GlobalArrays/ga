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
  cudaError_t ierr;
  ierr = cudaGetDeviceCount(&ngpus);
  /*cuDeviceGetCount(&ngpus); */
  if (ierr != cudaSuccess) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("p[%d] Error encountered by cudaGetDeviceCount\n",rank);
  }
  return ngpus;
}

/* set the GPU device for this processor
 * id: id of device
 */
void setDevice(int id)
{
  cudaError_t ierr = cudaSetDevice(id);
  if (ierr != cudaSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaSetDevice id: %d msg: %s\n",rank,id,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

/* allocate a unified memory buffer
 * buf: pointer to buffer
 * size: size of allocation in bytes
 */
void mallocDevice(void **buf, size_t size)
{
  cudaError_t ierr =cudaMalloc(buf, (int)size);
  cudaDeviceSynchronize();
  if (ierr != cudaSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMalloc buf: %p size: %d msg: %s\n",rank,*buf,size,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

/* free unified memory
 * buf: pointer to memory allocation
 */
void freeDevice(void *buf)
{
  cudaError_t ierr = cudaFree(buf);
  if (ierr != cudaSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaFree buf: %p msg: %s\n",rank,buf,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

/* copy data from host buffer to unified memory
 * hostptr: pointer to allocation on host
 * devptr: pointer to allocation on device
 * bytes: number of bytes to copy
 */
void copyToDevice(void *hostptr, void *devptr, int bytes)
{
  cudaError_t ierr = cudaMemcpy(devptr, hostptr, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  if (ierr != cudaSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMemcpy to device msg: %s\n",rank,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

/* copy data from unified memory to host buffer
 * hostptr: pointer to allocation on host
 * devptr: pointer to allocation on device
 * bytes: number of bytes to copy
 */
void copyToHost(void *hostptr, void *devptr, int bytes)
{
  cudaError_t ierr = cudaMemcpy(hostptr, devptr, bytes, cudaMemcpyDeviceToHost); 
  if (ierr != cudaSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMemcpy to host msg: %s\n",rank,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

/* copy data between devices using unified memory
 * srcptr: source pointer
 * dstptr: destination pointer
 * bytes: number of bytes to copy
 */
void copyDevToDev(void *srcptr, void *dstptr, int bytes)
{
  cudaError_t ierr = cudaMemcpy(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice); 
  if (ierr != cudaSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMemcpy dev to dev msg: %s\n",rank,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

/**
 * set values on the device to a specific value
 * ptr: pointer to device memory that needs to be set
 * val: integer representation of the value of each byte
 * size: number of bytes that should be set
 */
void deviceMemset(void *ptr, int val, size_t bytes)
{
  cudaError_t ierr = cudaMemset(ptr, val, bytes);
  if (ierr != cudaSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMemset ptr: %p bytes: %d msg: %s\n",rank,ptr,bytes,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

/* is pointer located on host?
 * return 1 data is located on host, 0 otherwise
 * ptr: pointer to data
 */
int isHostPointer(void *ptr)
{
  cudaPointerAttributes attr;
  cudaError_t tmp;
  cudaError_t  err = cudaPointerGetAttributes(&attr, ptr);
  /* Remove this error so that it doesn't trip up other error code */
  tmp = cudaGetLastError();
  /* Assume that if Cuda doesn't know anything about the pointer, it is on the
   * host */
  if (err != cudaSuccess) return 1;
  if (attr.devicePointer == NULL) {
    return  1;
  }
  return 0;
}

__global__ void iaxpy_kernel(int *dst, const int *src, int scale, int n)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  int i;
  for (i=index; i<n; i += stride) {
    dst[i] = dst[i] + scale*src[i];
  }
}

void deviceIaxpy(int *dst, int *src, const int *scale, int n)
{
  cudaError_t ierr;
  int nblocks = (n+1023)/1024;
  iaxpy_kernel<<<nblocks,1024>>>(dst, src, *scale, n);
  cudaDeviceSynchronize();
  ierr = cudaGetLastError();
  if (ierr != cudaSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] deviceIaxpy dst: %p src: %p scale: %d n: %d msg: %s\n",rank,dst,src,*scale,n,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

__global__ void laxpy_kernel(long *dst, const long *src, long scale, int n)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  int stride = blockDim.x*gridDim.x;
  int i;
  for (i=index; i<n; i += stride) {
    dst[i] = dst[i] + scale*src[i];
  }
}

void deviceLaxpy(long *dst, long *src, const long *scale, int n)
{
  int nblocks = (n+1023)/1024;
  laxpy_kernel<<<nblocks,1024>>>(dst, src, *scale, n);
}

__global__ void inc_int_kernel(int *target, const int *inc)
{
  int i = threadIdx.x;
  target[i] += inc[i];
}

void deviceAddInt(int *ptr, const int inc)
{
  int tmp;
  copyToHost(&tmp,ptr,sizeof(int));
  tmp += inc;
  copyToDevice(&tmp,ptr,sizeof(int));
}

__global__ void inc_long_kernel(long *target, const long *inc)
{
  int i = threadIdx.x;
  target[i] += inc[i];
}

void deviceAddLong(long *ptr, const long inc)
{
  long tmp;
  copyToHost(&tmp,ptr,sizeof(long));
  tmp += inc;
  copyToDevice(&tmp,ptr,sizeof(long));
}

int deviceGetMemHandle(devMemHandle_t *handle, void *memory)
{
  return cudaIpcGetMemHandle(&handle->handle, memory);
}

int deviceOpenMemHandle(void **memory, devMemHandle_t handle)
{
  cudaError_t ierr;
  ierr = cudaIpcOpenMemHandle(memory, handle.handle, cudaIpcMemLazyEnablePeerAccess);
  if (ierr != cudaSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaIpcOpenMemHandle msg: %s\n",rank,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
  return ierr;
}

int deviceCloseMemHandle(void *memory)
{
 return cudaIpcCloseMemHandle(memory);
}

};
