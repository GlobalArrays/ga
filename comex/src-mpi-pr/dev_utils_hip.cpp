
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

#include "dev_mem_handle.h"
#include <hip/hip_runtime_api.h>

#define hipErrCheck(stat)                     \
{                                             \
  hipErrCheck_((stat), __FILE__, __LINE__);   \
}

void hipErrCheck_(hipError_t stat, const char *file, int line)
{
  if (stat != hipSuccess) { fprintf(stderr, "hip Error: %s %s %d\n", hipGetErrorString(stat), file, line); }
}

/* avoid name mangling by the HIP compiler */
extern "C" {

extern int MPI_Wrapper_world_rank();
extern void MPI_Wrapper_abort(int err);

/* return the number of GPUs visible from this processor*/
int numDevices()
{
  int ngpus;
  hipError_t ierr;
  ierr = hipGetDeviceCount(&ngpus);
  /*cuDeviceGetCount(&ngpus); */
  if (ierr != hipSuccess) {
    int rank = MPI_Wrapper_world_rank();
    printf("p[%d] Error encountered by hipGetDeviceCount\n",rank);
  }
  return ngpus;
}

/* set the GPU device for this processor
 * id: id of device
 */
void setDevice(int id)
{
  hipError_t ierr = hipSetDevice(id);
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipSetDevice id: %d msg: %s\n",rank,id,msg);
    MPI_Wrapper_abort(err);
  }
}

/* allocate a unified memory buffer
 * buf: pointer to buffer
 * size: size of allocation in bytes
 */
void mallocDevice(void **buf, size_t size)
{
  hipError_t ierr =hipMalloc(buf, (int)size);
  hipDeviceSynchronize();
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMalloc buf: %p size: %zu msg: %s\n",rank,*buf,size,msg);
    MPI_Wrapper_abort(err);
  }
}

/* free unified memory
 * buf: pointer to memory allocation
 */
void freeDevice(void *buf)
{
  hipError_t ierr = hipFree(buf);
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipFree buf: %p msg: %s\n",rank,buf,msg);
    MPI_Wrapper_abort(err);
  }
}

/* is pointer located on host?
 * return 1 data is located on host, 0 otherwise
 * ptr: pointer to data
 */
int isHostPointer(void *ptr)
{
  hipPointerAttribute_t attr;
  hipError_t  tmp;
  hipError_t  err = hipPointerGetAttributes(&attr, ptr);
  /* Remove this error so that it doesn't trip up other error code */
  tmp = hipGetLastError();
  /* Assume that if hip doesn't know anything about the pointer, it is on the
   * host */
  if (err != hipSuccess) {
    return 1;
  }

  //TODO: no equivalent for cudaMemoryTypeUnregistered
  if (attr.memoryType == hipMemoryTypeHost /*|| attr.memoryType == cudaMemoryTypeUnregistered*/) {
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
  hipPointerAttribute_t attr;
  hipError_t  tmp;
  hipError_t  err = hipPointerGetAttributes(&attr, ptr);
  /* Remove this error so that it doesn't trip up other error code */
  tmp = hipGetLastError();
  /* Assume that if hip doesn't know anything about the pointer, it is on the
   * host */
  if (err != hipSuccess) {
    return -1;
  }

  if (attr.memoryType == hipMemoryTypeDevice) {
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
  hipError_t ierr = hipMemcpy(devptr, hostptr, bytes, hipMemcpyHostToDevice);
  hipDeviceSynchronize();
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMemcpy to device msg: %s\n",rank,msg);
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
  hipError_t ierr = hipMemcpy(hostptr, devptr, bytes, hipMemcpyDeviceToHost); 
  hipErrCheck(ierr);
  if (ierr != hipSuccess) {
    hipPointerAttribute_t src_attr, dst_attr;
    char hosttype[32];
    char devtype[32];
    int hostid, devid;
    int err=0;
    const char *msg = hipGetErrorString(ierr);
    hipErrCheck(hipPointerGetAttributes(&dst_attr, hostptr));
    if (dst_attr.memoryType == hipMemoryTypeDevice) {
	    strcpy(hosttype,"device");
	    hostid = dst_attr.device;
    } else {
	    strcpy(hosttype,"host");
	    hostid = -1;
    }
    hipErrCheck(hipPointerGetAttributes(&src_attr, devptr));
    if (src_attr.memoryType == hipMemoryTypeDevice) {
	    strcpy(devtype,"device");
	    devid = src_attr.device;
    } else {
	    strcpy(devtype,"host");
	    devid = -1;
    }
    int rank = MPI_Wrapper_world_rank();
    printf("p[%d] hipMemcpy to host dev: %d on %s host: %d on %s msg: %s\n",rank,hostid,hosttype,devid,devtype,msg);
    MPI_Wrapper_abort(err);
  }
  hipDeviceSynchronize();
}

/* copy data between buffers on same device
 * dstptr: destination pointer
 * srcptr: source pointer
 * bytes: number of bytes to copy
 */
void copyDevToDev(void *dstptr, void *srcptr, int bytes)
{
  hipError_t ierr;
  ierr = hipMemcpy(dstptr, srcptr, bytes, hipMemcpyDeviceToDevice); 
  hipErrCheck(ierr);
  hipDeviceSynchronize();
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMemcpy dev to dev msg: %s\n",rank,msg);
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
  hipError_t ierr;
  ierr = hipMemcpyPeer(dstptr,dstID,srcptr,srcID,bytes);
  hipErrCheck(ierr);
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMemcpyPeer dev to dev msg: %s\n",rank,msg);
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
  hipError_t ierr = hipMemset(ptr, val, bytes);
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMemset ptr: %p bytes: %zu msg: %s\n",rank,ptr,bytes,msg);
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
  hipError_t ierr;
  dim3 nblocks((n+1023)/1024);
  dim3 ttf(1024);
  hipLaunchKernelGGL(iaxpy_kernel, nblocks, ttf, 0, 0, dst, src, *scale, n);
  hipDeviceSynchronize();
  ierr = hipGetLastError();
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    hipPointerAttribute_t src_attr;
    hipPointerAttribute_t dst_attr;
    hipError_t  perr = hipPointerGetAttributes(&src_attr, src);
    //TODO: no equivalent for cudaMemoryTypeUnregistered
    if (perr != hipSuccess || src_attr.memoryType == hipMemoryTypeHost
      /*||  src_attr.memoryType == cudaMemoryTypeUnregistered*/) {
      printf("p[%d] deviceIaxpy src pointer is on host\n",rank);
    } else if (src_attr.memoryType == hipMemoryTypeDevice)  {
      printf("p[%d] deviceIaxpy src pointer is on device %d\n",rank,src_attr.device);
    }
    perr = hipPointerGetAttributes(&dst_attr, src);
    //TODO: no equivalent for cudaMemoryTypeUnregistered
    if (perr != hipSuccess || dst_attr.memoryType == hipMemoryTypeHost 
      /* || dst_attr.memoryType == cudaMemoryTypeUnregistered */) {
      printf("p[%d] deviceIaxpy src pointer is on host\n",rank);
    } else if (dst_attr.memoryType == hipMemoryTypeDevice)  {
      printf("p[%d] deviceIaxpy dst pointer is on device %d\n",rank,dst_attr.device);
    }
    printf("p[%d] deviceIaxpy dst: %p src: %p scale: %d n: %d msg: %s\n",rank,dst,src,*scale,n,msg);
    MPI_Wrapper_abort(err);
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
  dim3 nblocks((n+1023)/1024);
  dim3 ttf(1024);
  hipLaunchKernelGGL(laxpy_kernel, nblocks, ttf, 0, 0, dst, src, *scale, n);
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
  hipError_t ierr;
  ierr = hipIpcGetMemHandle(&handle->handle, memory);
  hipErrCheck(ierr);
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipGetMemHandle msg: %s\n",rank,msg);
    MPI_Wrapper_abort(err);
  }
  return ierr;
}

int deviceOpenMemHandle(void **memory, devMemHandle_t handle)
{
  hipError_t ierr;
  ierr = hipIpcOpenMemHandle(memory, handle.handle, hipIpcMemLazyEnablePeerAccess);
  hipErrCheck(ierr);
  if (ierr != hipSuccess) {
    int err=0;
    int rank = MPI_Wrapper_world_rank();
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipIpcOpenMemHandle msg: %s\n",rank,msg);
    MPI_Wrapper_abort(err);
  }
  return ierr;
}

int deviceCloseMemHandle(void *memory)
{
 return hipIpcCloseMemHandle(memory);
}

};

