
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>

#include <hip/hip_runtime_api.h>
#include "dev_mem_handle.h"


/* avoid name mangling by the HIP compiler */
extern "C" {

/* return the number of GPUs visible from this processor*/
int numDevices()
{
  int ngpus;
  hipError_t ierr;
  ierr = hipGetDeviceCount(&ngpus);
  /*hipGetDeviceCount(&ngpus); */
  if (ierr != hipSuccess) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipSetDevice id: %d msg: %s\n",rank,id,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
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
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMalloc buf: %p size: %d msg: %s\n",rank,*buf,size,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

/* free unified memory
 * buf: pointer to memory allocation
 */
void freeDevice(void *buf)
{
  hipError_t ierr = hipFree(buf);
  if (ierr != hipSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipFree buf: %p msg: %s\n",rank,buf,msg);
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
  hipError_t ierr = hipMemcpy(devptr, hostptr, bytes, hipMemcpyHostToDevice);
  hipDeviceSynchronize();
  if (ierr != hipSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMemcpy to device msg: %s\n",rank,msg);
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
  hipError_t ierr = hipMemcpy(hostptr, devptr, bytes, hipMemcpyDeviceToHost); 
  if (ierr != hipSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMemcpy to host msg: %s\n",rank,msg);
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
  hipError_t ierr = hipMemcpy(dstptr, srcptr, bytes, hipMemcpyDeviceToDevice); 
  if (ierr != hipSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMemcpy dev to dev msg: %s\n",rank,msg);
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
  hipError_t ierr = hipMemset(ptr, val, bytes);
  if (ierr != hipSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipMemset ptr: %p bytes: %d msg: %s\n",rank,ptr,bytes,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

/* is pointer located on host?
 * return 1 data is located on host, 0 otherwise
 * ptr: pointer to data
 */
int isHostPointer(void *ptr)
{
  hipPointerAttribute_t attr;
  hipError_t tmp;
  hipError_t  err = hipPointerGetAttributes(&attr, ptr);
  /* Remove this error so that it doesn't trip up other error code */
  tmp = hipGetLastError();
  /* Assume that if Cuda doesn't know anything about the pointer, it is on the
   * host */
  if (err != hipSuccess) return 1;
  if (attr.devicePointer == NULL) {
    return  1;
  }
  return 0;
}

//TODO
// __global__ void iaxpy_kernel(int *dst, int *src, int scale, int n)
// {
//   int index = blockIdx.x*blockDim.x+threadIdx.x;
//   int stride = blockDim.x*gridDim.x;
//   int i;
//   for (i=index; i<n; i += stride) {
//     dst[i] = dst[i] + scale*src[i];
//   }
// }

void deviceIaxpy(int *dst, int *src, const int *scale, int n)
{
  hipError_t ierr;
  dim3 nblocks((n+1023)/1024);
  dim3 ttf(1024);
  // hipLaunchKernelGGL(iaxpy_kernel, nblocks, ttf, 0, 0, dst, src, *scale, n);
  hipDeviceSynchronize();
  ierr = hipGetLastError();
  if (ierr != hipSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] deviceIaxpy dst: %p src: %p scale: %d n: %d msg: %s\n",rank,dst,src,*scale,n,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
}

// TODO
// __global__ void laxpy_kernel(long *dst, long *src, long scale, int n)
// {
//   int index = blockIdx.x*blockDim.x+threadIdx.x;
//   int stride = blockDim.x*gridDim.x;
//   int i;
//   for (i=index; i<n; i += stride) {
//     dst[i] = dst[i] + scale*src[i];
//   }
// }

void deviceLaxpy(long *dst, long *src, const long *scale, int n)
{
  dim3 nblocks((n+1023)/1024);
  dim3 ttf(1024);
  // hipLaunchKernelGGL(laxpy_kernel, nblocks, ttf, 0, 0, dst, src, *scale, n);
}

//TODO
// __global__ void inc_int_kernel(int *target, const int *inc)
// {
//   int i = threadIdx.x;
//   target[i] += inc[i];
// }

void deviceAddInt(int *ptr, const int inc)
{
  int tmp;
  copyToHost(&tmp,ptr,sizeof(int));
  tmp += inc;
  copyToDevice(&tmp,ptr,sizeof(int));
}

//TODO
// __global__ void inc_long_kernel(long *target, const long *inc)
// {
//   int i = threadIdx.x;
//   target[i] += inc[i];
// }

void deviceAddLong(long *ptr, const long inc)
{
  long tmp;
  copyToHost(&tmp,ptr,sizeof(long));
  tmp += inc;
  copyToDevice(&tmp,ptr,sizeof(long));
}

int deviceGetMemHandle(devMemHandle_t *handle, void *memory)
{
  return hipIpcGetMemHandle(&handle->handle, memory);
}

int deviceOpenMemHandle(void **memory, devMemHandle_t handle)
{
  hipError_t ierr;
  ierr = hipIpcOpenMemHandle(memory, handle.handle, hipIpcMemLazyEnablePeerAccess);
  if (ierr != hipSuccess) {
    int rank, err=0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const char *msg = hipGetErrorString(ierr);
    printf("p[%d] hipIpcOpenMemHandle msg: %s\n",rank,msg);
    MPI_Abort(MPI_COMM_WORLD,err);
  }
  return ierr;
}

int deviceCloseMemHandle(void *memory)
{
 return hipIpcCloseMemHandle(memory);
}

};

