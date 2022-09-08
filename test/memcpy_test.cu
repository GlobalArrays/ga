#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

#define N (1024)
#define M (512)
#define X 31
#define Y 33
#define MAXDIM 2

struct strided_memcpy_kernel_arg {
  char *dst;
  char *src;
  int dst_strides[MAXDIM];  // Smallest strides are first
  int src_strides[MAXDIM];
  int dims[MAXDIM]; // Dimension of block being transferred
  int stride_levels; // Dimension of array minus 1
  int size; // Size of array elements
  int bytes_per_block;
};

__global__ void strided_memcpy_kernel(strided_memcpy_kernel_arg arg) {
  //int index = blockIdx.x;
  int index = threadIdx.x;

#if SINGLE
  int bytes = 1, i;
  //for(i = 0 ; i < MAXDIM; ++i) bytes *= N;
  if (!index) memcpy(arg.dst, arg.src, M*M*sizeof(char));
  __syncthreads();
#else


  int i, thread_index = 0;
  int idx[MAXDIM];
  int bytes_per_thread = arg.size;
 // int thread_offset = bytes_per_thread * threadIdx.x;
  int stride_levels = arg.stride_levels;
  int src_block_offset; // Offset based on chunk_index
  int dst_block_offset; // Offset based on chunk_index

  // Determine location of chunk_index in array
  //index /= bytes_per_thread;
  for (i=0; i<=stride_levels; i++) {
    idx[i] = index%arg.dims[i];
    index = (index-idx[i])/arg.dims[i];
  }
  src_block_offset = bytes_per_thread*idx[0];
  dst_block_offset = bytes_per_thread*idx[0];
  for (i=0; i<stride_levels; i++) {
    src_block_offset += arg.src_strides[i]*idx[i+1];
    dst_block_offset += arg.dst_strides[i]*idx[i+1];
  }

  while (thread_index < arg.bytes_per_block)
  {
    //memcpy(arg.dst + dst_block_offset + thread_offset,
    memcpy(arg.dst + dst_block_offset,
    //       arg.src + src_block_offset + thread_offset, bytes_per_thread);
           arg.src + src_block_offset, bytes_per_thread);
    //thread_index += bytes_per_thread * blockDim.x;
    thread_index += bytes_per_thread;
    __syncthreads();
  }
#endif
}

int main() {
  int bytesN = 1, i;
   //for(i = 0; i < MAXDIM; ++i)
	//bytes = N * N;
  
  // ...Allocate memory

  char *block_A = NULL;
  char *block_B = NULL;


  // Test example: 
  // Block_A = i on CPU
  // Block_A -> Block_B on GPU
  // Check Block_B on CPU

  cudaHostAlloc(&block_A, N*N*sizeof(*block_A), 0);
  cudaHostAlloc(&block_B, M*M*sizeof(*block_B), 0);

  if(!block_A || !block_B ){
     printf("Allocation error:  %s %d\n", __FILE__, __LINE__);
     return 10;
  }

  int number_of_chunk = 16;
  int bytes_per_block = 1024;
  strided_memcpy_kernel_arg toarg;


  toarg.bytes_per_block = sizeof(*block_A); // Since we are copying 4 bytes per
                                // thread, bytes per block should be 
                                // a multiple of 4

  //for(i = 0; i < MAXDIM; ++i) toarg.dst_strides[i] = 1;
  //for(i = 0; i < MAXDIM; ++i) toarg.src_strides[i] = 1;
   /* dimension of array times size of element */
  toarg.dst_strides[0] = 512 * 1;
  toarg.dst_strides[1] = 512;

  toarg.src_strides[0] = 1024 * 1;
  toarg.src_strides[1] = 1024;

  // for(i = 0; i < MAXDIM; ++i) toarg.dims[i] = 256;
  
  toarg.dims[0] = Y;
  toarg.dims[1] = X;

  toarg.size = 1;
  toarg.stride_levels = MAXDIM-1;

  // Register the host pointer
  cudaHostGetDevicePointer (&(toarg.src), block_A, 0);
  cudaHostGetDevicePointer (&(toarg.dst), block_B, 0);

  for(i = 0; i < N*N; ++i){
     toarg.src[i] = i;
  }
  for(i = 0; i < M*M; ++i){
     toarg.dst[i] = 0;
  }

  strided_memcpy_kernel<<<1, X * Y>>>(toarg);
  cudaDeviceSynchronize();

  int err = 0, j;

  for(i = 0; i < X; ++i){
  for(j = 0; j < Y; ++j){
     printf("%d ", block_B[j + M*i]) ;
//     if(block_A[i] != block_B[i]) err++;
     if(block_A[j + N*i] != block_B[j + M*i]) err++;
  }
  }
  printf("Error: %d \n", err);
  fflush(stdout);

  cudaFreeHost(block_A);
  cudaFreeHost(block_B);
  return 0;
}
