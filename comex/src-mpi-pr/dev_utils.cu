#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include "dev_mem_handle.h"

#include "comex.h"

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
  cudaError_t  err = cudaPointerGetAttributes(&attr, ptr);
  /* Remove this error so that it doesn't trip up other error code */
  cudaGetLastError();
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
  cudaError_t  err = cudaPointerGetAttributes(&attr, ptr);
  /* Remove this error so that it doesn't trip up other error code */
  cudaGetLastError();
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

#define MAXDIM 7
struct strided_kernel_arg {
  void *dst;
  void *src;
  int dst_strides[MAXDIM];  /* smallest strides are first */
  int src_strides[MAXDIM];
  int dims[MAXDIM];         /* dimensions of block being transferred */
  int stride_levels;        /* dimension of array minus 1 */
  int elem_size;            /* size of array elements */
  int totalCopyElems;       /* total constructs to copy */
  int elements_per_block;   /* number of elements copied by each thread */
  int op;                   /* accumulate operation (if applicable) */
  char scale[64];           /* accumulate scale parameter */
};

__global__ void strided_memcpy_kernel(strided_kernel_arg arg) {
  int index = threadIdx.x;
  int stride = blockIdx.x;

  int i;
  int idx[MAXDIM];
  int currElem = 0;
  int elements_per_block = arg.elements_per_block;
  int bytes_per_thread = arg.elem_size*elements_per_block;
  int stride_levels = arg.stride_levels;
  int src_block_offset; /* Offset based on chunk_index */
  int dst_block_offset; /* Offset based on chunk_index */

  /* Determine location of chunk_index in array based
    on the thread id and the block id */
  index = index + stride * blockDim.x;
  /* If the thread index is bigger than the total transfer 
    entities then this thread does not participate in the 
    copy */
  if(index >= arg.totalCopyElems) {
     return;
  }
  /* Find the indices that mark the location of this element within
     the block of data that will be moved */
  index *= elements_per_block;
  // Calculate the index starting points 
  for (i=0; i<=stride_levels; i++) {
    idx[i] = index%arg.dims[i];
    index = (index-idx[i])/arg.dims[i];
  }
  /* Calculate the block offset for this thread */
  src_block_offset = bytes_per_thread*idx[0];
  dst_block_offset = bytes_per_thread*idx[0];
  for (i=0; i<stride_levels; i++) {
    src_block_offset += arg.src_strides[i]*idx[i+1]*bytes_per_thread;
    dst_block_offset += arg.dst_strides[i]*idx[i+1]*bytes_per_thread;
  }

  /* Start copying element by element 
     TODO: Make it sure that it is continuous and replace the loop
     with a single memcpy */
  memcpy((char*)arg.dst + dst_block_offset + currElem * bytes_per_thread,
      (char*)arg.src + src_block_offset + currElem * bytes_per_thread, 
      elements_per_block*bytes_per_thread);
  /* Synchronize the threads before returning  */
  __syncthreads();
}

#define TTHREADS 1024
void parallelMemcpy(void *src,         /* starting pointer of source data */
                    int *src_stride,   /* strides of source data */
                    void *dst,         /* starting pointer of destination data */
                    int *dst_stride,   /* strides of destination data */
                    int *count,        /* dimensions of data block to be transfered */
                    int stride_levels) /* number of stride levels */
{
  int src_on_host = isHostPointer(src);
  int dst_on_host = isHostPointer(dst);
  void *msrc;
  void *mdst;
  int total_elems;
  int nblocks;
  strided_kernel_arg arg;
  int i;

  /* if src or dst is on host, map pointer to device */
  if (src_on_host) {
    /* Figure out how large segment of memory is. If this routine is being
       called, stride_levels must be at least 1 */
    int total = count[stride_levels]*src_stride[stride_levels-1];
    cudaHostRegister(src, total, cudaHostRegisterMapped); 
    /* Register the host pointer */
    cudaHostGetDevicePointer (&msrc, src, 0);
  } else {
    msrc = src;
  }
  if (dst_on_host) {
    /* Figure out how large segment of memory is. If this routine is being
       called, stride_levels must be at least 1 */
    int total = count[stride_levels]*dst_stride[stride_levels-1];
    cudaHostRegister(dst, total, cudaHostRegisterMapped); 
    /* Register the host pointer */
    cudaHostGetDevicePointer (&mdst, dst, 0);
  } else {
    mdst = dst;
  }
  
  /* total_elems = count[0]/elem_size; */
  total_elems = 1;
  for (i=0; i<stride_levels; i++) total_elems *= count[i+1];

  if(total_elems < TTHREADS){
    nblocks = 1;
  } else {
    nblocks = int(ceil(((float)total_elems)/((float)TTHREADS)));
  }

  arg.dst = mdst;
  arg.src = msrc;
  arg.elem_size = count[0];
  for (i=0; i<stride_levels; i++) {
    arg.dst_strides[i] = dst_stride[i]/arg.elem_size;
    arg.src_strides[i] = src_stride[i]/arg.elem_size;
  }
  for (i=0; i<=stride_levels; i++) arg.dims[i] = count[i];
  arg.dims[0] = 1;
  arg.stride_levels = stride_levels;
  /* arg.elem_size = elem_size; */
  arg.totalCopyElems = total_elems;
  arg.elements_per_block = 1;

  strided_memcpy_kernel<<<nblocks, TTHREADS>>>(arg);

  /*
  cudaDeviceSynchoronize();
  */
  if (src_on_host) {
    cudaHostUnregister(src);
  }
  if (dst_on_host) {
    cudaHostUnregister(dst);
  }
     

}

__global__ void strided_accumulate_kernel(strided_kernel_arg arg) {
  int index = threadIdx.x;
  int stride = blockIdx.x;

  int i;
  int idx[MAXDIM];
  int elements_per_block = arg.elements_per_block;
  int bytes_per_thread = arg.elem_size*elements_per_block;
  int stride_levels = arg.stride_levels;
  int src_block_offset; /* Offset based on chunk_index */
  int dst_block_offset; /* Offset based on chunk_index */
  void *src, *dst;
  int op;

  /* Determine location of chunk_index in array based
    on the thread id and the block id */
  index = index + stride * blockDim.x;
  /* If the thread index is bigger than the total transfer 
    entities then this thread does not participate in the 
    copy */
  if(index >= arg.totalCopyElems) {
     return;
  }
  /* Find the indices that mark the location of this element within
     the block of data that will be moved */
  // index *= elements_per_block;
  // Calculate the index starting points 
  for (i=0; i<=stride_levels; i++) {
    idx[i] = index%arg.dims[i];
    index = (index-idx[i])/arg.dims[i];
  }
  /* Calculate the block offset for this thread */
  src_block_offset = bytes_per_thread*idx[0];
  dst_block_offset = bytes_per_thread*idx[0];
  for (i=0; i<stride_levels; i++) {
    src_block_offset += arg.src_strides[i]*idx[i+1];
    dst_block_offset += arg.dst_strides[i]*idx[i+1];
  }

  /* Start copying element by element 
     TODO: Make it sure that it is continuous and replace the loop
     with a single memcpy */
  src = (void*)((char*)arg.src + src_block_offset);
  dst = (void*)((char*)arg.dst + dst_block_offset);
  op = arg.op;
  if (op == COMEX_ACC_INT) {
    int a = *((int*)src);
    int scale = *((int*)arg.scale);
    *((int*)dst) += a*scale;
  } else if (op == COMEX_ACC_LNG) {
    long a = *((long*)src);
    long scale = *((long*)arg.scale);
    *((long*)dst) += a*scale;
  } else if (op == COMEX_ACC_FLT) {
    float a = *((float*)src);
    float scale = *((float*)arg.scale);
    *((float*)dst) += a*scale;
  } else if (op == COMEX_ACC_DBL) {
    double a = *((double*)src);
    double scale = *((double*)arg.scale);
    *((double*)dst) += a*scale;
  } else if (op == COMEX_ACC_CPL) {
    float ar = *((float*)src);
    float ai = *(((float*)src)+1);
    float scaler = *((float*)arg.scale);
    float scalei = *(((float*)arg.scale)+1);
    *((float*)dst) += ar*scaler-ai*scalei;
    *(((float*)dst)+1) += ar*scalei+ai*scaler;
  } else if (op == COMEX_ACC_DCP) {
    double ar = *((double*)src);
    double ai = *(((double*)src)+1);
    double scaler = *((double*)arg.scale);
    double scalei = *(((double*)arg.scale)+1);
    *((double*)dst) += ar*scaler-ai*scalei;
    *(((double*)dst)+1) += ar*scalei+ai*scaler;
  }
  /* Synchronize the threads before returning  */
  __syncthreads();
}

void parallelAccumulate(int op,        /* accumulate operation */
                    void *src,         /* starting pointer of source data */
                    int *src_stride,   /* strides of source data */
                    void *dst,         /* starting pointer of destination data */
                    int *dst_stride,   /* strides of destination data */
                    int *count,        /* dimensions of data block to be transfered */
                    int stride_levels, /* number of stride levels */
                    void *scale)       /* scale factor in accumulate */
{
  int src_on_host = isHostPointer(src);
  int dst_on_host = isHostPointer(dst);
  void *msrc;
  void *mdst;
  int total_elems;
  int elem_size;
  int nblocks;
  strided_kernel_arg arg;
  int i;

  /* if src or dst is on host, map pointer to device */
  if (src_on_host) {
    /* Figure out how large segment of memory is. If this routine is being
       called, stride_levels must be at least 1 */
    int total = count[stride_levels]*src_stride[stride_levels-1];
    cudaHostRegister(src, total, cudaHostRegisterMapped); 
    /* Register the host pointer */
    cudaHostGetDevicePointer (&msrc, src, 0);
  } else {
    msrc = src;
  }
  if (dst_on_host) {
    /* Figure out how large segment of memory is. If this routine is being
       called, stride_levels must be at least 1 */
    int total = count[stride_levels]*dst_stride[stride_levels-1];
    cudaHostRegister(dst, total, cudaHostRegisterMapped); 
    /* Register the host pointer */
    cudaHostGetDevicePointer (&mdst, dst, 0);
  } else {
    mdst = dst;
  }
  
  /* total_elems = count[0]/elem_size; */
  if (op == COMEX_ACC_INT) {
    elem_size = sizeof(int);
    *((int*)arg.scale) = *((int*)scale);
  } else if (op == COMEX_ACC_LNG) {
    elem_size = sizeof(long);
    *((long*)arg.scale) = *((long*)scale);
  } else if (op == COMEX_ACC_FLT) {
    elem_size = sizeof(float);
    *((float*)arg.scale) = *((float*)scale);
  } else if (op == COMEX_ACC_DBL) {
    elem_size = sizeof(double);
    *((double*)arg.scale) = *((double*)scale);
  } else if (op == COMEX_ACC_CPL) {
    elem_size = 2*sizeof(float);
    *((float*)arg.scale) = *((float*)scale);
    *(((float*)arg.scale)+1) = *(((float*)scale)+1);
  } else if (op == COMEX_ACC_DCP) {
    elem_size = 2*sizeof(double);
    *((double*)arg.scale) = *((double*)scale);
    *(((double*)arg.scale)+1) = *(((double*)scale)+1);
  }

  total_elems = count[0]/elem_size;
  for (i=0; i<stride_levels; i++) total_elems *= count[i+1];

  if(total_elems < TTHREADS){
    nblocks = 1;
  } else {
    nblocks = int(ceil(((float)total_elems)/((float)TTHREADS)));
  }

  arg.src = msrc;
  arg.dst = mdst;
  arg.elem_size = elem_size;
  arg.op = op;
  for (i=0; i<stride_levels; i++) {
    arg.dst_strides[i] = dst_stride[i];
    arg.src_strides[i] = src_stride[i];
  }
  for (i=0; i<=stride_levels; i++) arg.dims[i] = count[i];
  arg.dims[0] = count[0]/elem_size;
  arg.stride_levels = stride_levels;
  /* arg.elem_size = elem_size; */
  arg.totalCopyElems = total_elems;
  arg.elements_per_block = 1;

  strided_accumulate_kernel<<<nblocks, TTHREADS>>>(arg);

  /*
  cudaDeviceSynchoronize();
  */
  if (src_on_host) {
    cudaHostUnregister(src);
  }
  if (dst_on_host) {
    cudaHostUnregister(dst);
  }
}
};
