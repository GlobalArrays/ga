#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "dev_mem_handle.h"

#define cudaErrCheck(stat)                                                                                             \
  {                                                                                                                    \
    cudaErrCheck_((stat), __FILE__, __LINE__);                                                                         \
  }

void cudaErrCheck_(cudaError_t stat, const char *file, int line)
{
  if (stat != cudaSuccess) { fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line); }
}

/* avoid name mangling by the CUDA compiler */
extern "C" {

extern int MPI_Wrapper_world_rank();
extern void MPI_Wrapper_abort(int err);

/* return the number of GPUs visible from this processor*/
int numDevices()
{
  int ngpus;
  cudaError_t ierr;
  ierr = cudaGetDeviceCount(&ngpus);
  /*cuDeviceGetCount(&ngpus); */
  if (ierr != cudaSuccess) {
    int rank = MPI_Wrapper_world_rank();
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
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaSetDevice id: %d msg: %s\n",rank,id,msg);
    MPI_Wrapper_abort(err);
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
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMalloc buf: %p size: %d msg: %s\n",rank,*buf,size,msg);
    MPI_Wrapper_abort(err);
  }
}

/* free unified memory
 * buf: pointer to memory allocation
 */
void freeDevice(void *buf)
{
  cudaError_t ierr = cudaFree(buf);
  if (ierr != cudaSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaFree buf: %p msg: %s\n",rank,buf,msg);
    MPI_Wrapper_abort(err);
  }
}

/* is pointer located on host?
 * return 1 data is located on host, 0 otherwise
 * ptr: pointer to data
 */
int isHostPointer(void *ptr)
{
  cudaPointerAttributes attr;
  cudaError_t  tmp;
  cudaError_t  err = cudaPointerGetAttributes(&attr, ptr);
  /* Remove this error so that it doesn't trip up other error code */
  tmp = cudaGetLastError();
  /* Assume that if Cuda doesn't know anything about the pointer, it is on the
   * host */
  if (err != cudaSuccess) {
    return 1;
  }

  if (attr.type != cudaMemoryTypeDevice) {
    return  1;
  }
  return 0;
}

/* return local ID of device hosting buffer. Return -1
 * if buffer is on host
 * ptr: pointer to data
 */
int getDeviceID(void *ptr)
{
  cudaPointerAttributes attr;
  cudaError_t  tmp;
  cudaError_t  err = cudaPointerGetAttributes(&attr, ptr);
  /* Remove this error so that it doesn't trip up other error code */
  tmp = cudaGetLastError();
  /* Assume that if Cuda doesn't know anything about the pointer, it is on the
   * host */
  if (err != cudaSuccess) {
    return -1;
  }

  if (attr.type == cudaMemoryTypeDevice) {
    return  attr.device;
  }
  return -1;
}

/* copy data from host buffer to unified memory
 * devptr: pointer to allocation on device
 * hostptr: pointer to allocation on host
 * bytes: number of bytes to copy
 */
void copyToDevice(void *devptr, void *hostptr, int bytes)
{
  cudaError_t ierr = cudaMemcpy(devptr, hostptr, bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  if (ierr != cudaSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMemcpy to device msg: %s\n",rank,msg);
    MPI_Wrapper_abort(err);
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
  cudaErrCheck(ierr);
  if (ierr != cudaSuccess) {
    cudaPointerAttributes src_attr, dst_attr;
    char hosttype[32];
    char devtype[32];
    int hostid, devid;
    int err=0;
    const char *msg = cudaGetErrorString(ierr);
    cudaErrCheck(cudaPointerGetAttributes(&dst_attr, hostptr));
    if (dst_attr.type == cudaMemoryTypeDevice) {
	    strcpy(hosttype,"device");
	    hostid = dst_attr.device;
    } else {
	    strcpy(hosttype,"host");
	    hostid = -1;
    }
    cudaErrCheck(cudaPointerGetAttributes(&src_attr, devptr));
    if (src_attr.type == cudaMemoryTypeDevice) {
	    strcpy(devtype,"device");
	    devid = src_attr.device;
    } else {
	    strcpy(devtype,"host");
	    devid = -1;
    }
    int rank = MPI_Wrapper_world_rank();
    printf("p[%d] cudaMemcpy to host dev: %d on %s host: %d on %s msg: %s\n",rank,hostid,hosttype,devid,devtype,msg);
    MPI_Wrapper_abort(err);
  }
  cudaDeviceSynchronize();
}

/* copy data between buffers on same device
 * dstptr: destination pointer
 * srcptr: source pointer
 * bytes: number of bytes to copy
 */
void copyDevToDev(void *dstptr, void *srcptr, int bytes)
{
  cudaError_t ierr;
  ierr = cudaMemcpy(dstptr, srcptr, bytes, cudaMemcpyDeviceToDevice); 
#if 0
  {
    cudaPointerAttributes src_attr, dst_attr;
    int rank = MPI_Wrapper_world_rank();
    cudaPointerGetAttributes(&src_attr, srcptr);
    cudaPointerGetAttributes(&dst_attr, dstptr);
    if (src_attr.device != dst_attr.device) {
      printf("p[%d] copyDevToDev devices don't match src_dev: %d dst_dev: %d\n",
          rank,src_attr.device,dst_attr.device);
    }
  }
#endif
  cudaErrCheck(ierr);
  cudaDeviceSynchronize();
  if (ierr != cudaSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMemcpy dev to dev msg: %s\n",rank,msg);
    MPI_Wrapper_abort(err);
  }
}

/* copy data between buffers on different devices
 * dstptr: destination pointer
 * dstID: device ID of destination
 * srcptr: source pointer
 * srcID: device ID of source
 * bytes: number of bytes to copy
 */
void copyPeerToPeer(void *dstptr, int dstID, void *srcptr, int srcID, int bytes)
{
  cudaError_t ierr;
  ierr = cudaMemcpyPeer(dstptr,dstID,srcptr,srcID,bytes);
  cudaErrCheck(ierr);
  if (ierr != cudaSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMemcpyPeer dev to dev msg: %s\n",rank,msg);
    MPI_Wrapper_abort(err);
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
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaMemset ptr: %p bytes: %d msg: %s\n",rank,ptr,bytes,msg);
    MPI_Wrapper_abort(err);
  }
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
  ierr = cudaGetLastError();
  if (ierr != cudaSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    cudaPointerAttributes src_attr;
    cudaPointerAttributes dst_attr;
    cudaError_t  perr = cudaPointerGetAttributes(&src_attr, src);
    if (perr != cudaSuccess || src_attr.type != cudaMemoryTypeDevice) {
      printf("p[%d] deviceIaxpy src pointer is on host\n",rank);
    } else if (src_attr.type == cudaMemoryTypeDevice)  {
      printf("p[%d] deviceIaxpy src pointer is on device %d\n",rank,src_attr.device);
    }
    perr = cudaPointerGetAttributes(&dst_attr, src);
    if (perr != cudaSuccess || dst_attr.type != cudaMemoryTypeDevice) {
      printf("p[%d] deviceIaxpy src pointer is on host\n",rank);
    } else if (dst_attr.type == cudaMemoryTypeDevice)  {
      printf("p[%d] deviceIaxpy dst pointer is on device %d\n",rank,dst_attr.device);
    }
    printf("p[%d] deviceIaxpy dst: %p src: %p scale: %d n: %d msg: %s\n",
        rank,dst,src,*scale,n,msg);
    MPI_Wrapper_abort(err);
  }
  cudaDeviceSynchronize();
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
  copyToDevice(ptr,&tmp,sizeof(int));
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
  copyToDevice(ptr,&tmp,sizeof(long));
}

int deviceGetMemHandle(devMemHandle_t *handle, void *memory)
{
  cudaError_t ierr;
  ierr = cudaIpcGetMemHandle(&handle->handle, memory);
  cudaErrCheck(ierr);
  if (ierr != cudaSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaGetMemHandle msg: %s\n",rank,msg);
    MPI_Wrapper_abort(err);
  }
  return ierr;
}

int deviceOpenMemHandle(void **memory, devMemHandle_t handle)
{
  cudaError_t ierr;
  ierr = cudaIpcOpenMemHandle(memory, handle.handle, cudaIpcMemLazyEnablePeerAccess);
  cudaErrCheck(ierr);
#if 0
  {
    int rank = MPI_Wrapper_world_rank();
    printf("p[%d] deviceOpenMemHandle pointer: %p\n",rank,*memory);
  }
#endif
  if (ierr != cudaSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = cudaGetErrorString(ierr);
    printf("p[%d] cudaIpcOpenMemHandle msg: %s\n",rank,msg);
    MPI_Wrapper_abort(err);
  }
  return ierr;
}

int deviceCloseMemHandle(void *memory)
{
#if 0
  {
    int rank = MPI_Wrapper_world_rank();
    printf("p[%d] deviceCloseMemHandle pointer: %p\n",rank,memory);
  }
#endif
 return cudaIpcCloseMemHandle(memory);
}

};
