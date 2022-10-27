#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#define TTHREADS 1024

#define N (1024)
#define M (512)
#define X 61
#define Y 65
#define MAXDIM 2
#define REPS 100

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#ifdef TYPE_INT
typedef uint32_t 	TEST_TYPE; 
#elif TYPE_DOUBLE
typedef double 		TEST_TYPE;
#elif TYPE_FLOAT
typedef float 		TEST_TYPE;
#elif TYPE_LONG
typedef uint64_t 	TEST_TYPE;
#elif SHORT
typedef uint16_t	TEST_TYPE;
#else
typedef uint8_t 	TEST_TYPE;
#endif

struct strided_memcpy_kernel_arg {
  char *dst;
  char *src;
  int dst_strides[MAXDIM];  // Smallest strides are first
  int src_strides[MAXDIM];
  int dims[MAXDIM]; // Dimension of block being transferred
  int stride_levels; // Dimension of array minus 1
  int size; // Size of array elements
  int totalTransferEnts; // Total constructs to copy
  int bytes_per_block;
};

__global__ void strided_memcpy_kernel(strided_memcpy_kernel_arg arg) {
  int index = threadIdx.x;
  int stride = blockIdx.x;

#if SINGLE
  int bytes = 1, i;
  if (!index) memcpy(arg.dst, arg.src, M*M*sizeof(char));
  __syncthreads();
#else
  int i;
  int idx[MAXDIM];
  int currElem = 0;
  int bytes_per_thread = arg.size;
  int elements_per_block = arg.bytes_per_block;
  int stride_levels = arg.stride_levels;
  int src_block_offset; // Offset based on chunk_index
  int dst_block_offset; // Offset based on chunk_index

  // Determine location of chunk_index in array
  index = index + stride * blockDim.x;
  if(index >= arg.totalTransferEnts) {
     return;
  }
  index *= elements_per_block;
  for (i=0; i<=stride_levels; i++) {
    idx[i] = index%arg.dims[i];
    index = (index-idx[i])/arg.dims[i];
  }
  src_block_offset = bytes_per_thread*idx[0];
  dst_block_offset = bytes_per_thread*idx[0];
  for (i=0; i<stride_levels; i++) {
    src_block_offset += arg.src_strides[i]*idx[i+1]*bytes_per_thread;
    dst_block_offset += arg.dst_strides[i]*idx[i+1]*bytes_per_thread;
  }

  //printf("[%d] BT/EB: %d %d [%d %d]\n", threadIdx.x, bytes_per_thread, elements_per_block, idx[0], idx[1]);
  while(currElem < elements_per_block){
     memcpy(arg.dst + dst_block_offset + currElem * bytes_per_thread,
            arg.src + src_block_offset + currElem * bytes_per_thread, 
            bytes_per_thread);
     //printf("[%d] Address: [%p] | %d - %d\n", threadIdx.x, (void *)(arg.dst + dst_block_offset + currElem * bytes_per_thread), currElem, elements_per_block);
     currElem += 1;
  }
  __syncthreads();
#endif
}

int check_errors(TEST_TYPE *block_A, TEST_TYPE *block_B){
  int err = 0, i, j;
  int *err_idx = (int *)calloc(X*Y, sizeof(*err_idx));
  int *err_jdx = (int *)calloc(X*Y, sizeof(*err_idx));

  for(i = 0 ; i < M * M; ++i){
     if(block_B[i] != 0){ 
        err ++;
        //printf("X V: %d %d\n", i, block_B[i]);
     }
  }

  printf("Set values vs expected: %d %d\n", err, X*Y);
  err = 0;

  for(i = 0; i < X; ++i){
     for(j = 0; j < Y; ++j){
        if(block_A[j + N*i] != block_B[j + M*i]){
           err_idx[err] = i + 1;
           err_jdx[err] = j + 1;
           err++;
        }
     }
  }
  printf("Error: %d \n", err);
  if(err){
     for(i = 0; i < err; ++i){
         int x = err_idx[i] - 1;
         int y = err_jdx[i] - 1;
         TEST_TYPE val = block_B[x* M + y];
         printf("I, J, V: %d %d %d\n", x, y, val);
     }
  }

  free(err_idx);
  free(err_jdx);
  return 0;
}

int check_errors_exit(TEST_TYPE *block_A, TEST_TYPE *block_B){
  int err = 0, i, j;
  int *err_idx = (int *)calloc(X*Y, sizeof(*err_idx));
  int *err_jdx = (int *)calloc(X*Y, sizeof(*err_idx));

  for(i = 0 ; i < M * M; ++i){
     if(block_B[i] != 0) err ++;
  }


  if(err != X*Y){
     printf("Set values are not the same: %d %d\n", err, X*Y);
     exit(8);
  }
  err = 0;

  for(i = 0; i < X; ++i){
     for(j = 0; j < Y; ++j){
        if(block_A[j + N*i] != block_B[j + M*i]){
           err_idx[err] = i + 1;
           err_jdx[err] = j + 1;
           err++;
        }
     }
  }
  if(err){
     printf("Error: %d \n", err);
     exit(9);
  }

  free(err_idx);
  free(err_jdx);
  return 0;
}



int main() {
  int i;
  float time;
  cudaEvent_t start, stop;
 
  // ...Allocate memory

  TEST_TYPE *block_A = NULL;
  TEST_TYPE *block_B = NULL;


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

  strided_memcpy_kernel_arg toarg;

  long blockDims = 1, totalSize = X*Y;

  if(totalSize < TTHREADS){
     blockDims = 1;
  }
  else{
     blockDims = int(ceil(totalSize / (float)TTHREADS));

  }

  printf("BLOCKS %d\n", blockDims);

  /* dimension of array times size of element */
  toarg.dst_strides[0] = M * 1;
  toarg.dst_strides[1] = M;

  toarg.src_strides[0] = N * 1;
  toarg.src_strides[1] = N;

  toarg.dims[0] = Y;
  toarg.dims[1] = X;

  toarg.stride_levels = MAXDIM-1;
  toarg.totalTransferEnts = X*Y;

  // Register the host pointer
  cudaHostGetDevicePointer (&(toarg.src), block_A, 0);
  cudaHostGetDevicePointer (&(toarg.dst), block_B, 0);

  for(i = 0; i < N*N; ++i){
     block_A[i] = i+1;
  }
  for(i = 0; i < M*M; ++i){
     block_B[i] = 0;
  }

  /* Cold call for the copying kernel */
  toarg.bytes_per_block = 1; /* Number of elements to copy by each thread */
  toarg.size = sizeof(*block_A); /* Size of elements */
  
  HANDLE_ERROR( cudaEventCreate(&start) );
  HANDLE_ERROR( cudaEventCreate(&stop) );
  HANDLE_ERROR( cudaEventRecord(start, 0) );

  strided_memcpy_kernel<<<blockDims, TTHREADS>>>(toarg);

  HANDLE_ERROR( cudaEventRecord(stop, 0) );
  HANDLE_ERROR( cudaEventSynchronize(stop) );
  HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
  cudaDeviceSynchronize();
  printf("Cold Call: Single Element Time for kernel: %3.4f ms \n", time);

  check_errors(block_A, block_B);

/* Cold call for the copying kernel */
  if(totalSize < TTHREADS){
     blockDims = 1;
  }
  else{
     blockDims = int(ceil(X*Y / (float)TTHREADS));

  }

  int ex = 0;
  double avgTime = 0.0, minT, maxT;

  for(ex = 0; ex < REPS; ++ex){

     for(i = 0; i < M*M; ++i){
        toarg.dst[i] = 0;
     }

     toarg.bytes_per_block = 1; /* Number of elements to copy by each thread */
     toarg.size = sizeof(*block_A); /* Size of elements */

     HANDLE_ERROR( cudaEventCreate(&start) );
     HANDLE_ERROR( cudaEventCreate(&stop) );
     HANDLE_ERROR( cudaEventRecord(start, 0) );

     strided_memcpy_kernel<<<blockDims, TTHREADS>>>(toarg);

     HANDLE_ERROR( cudaEventRecord(stop, 0) );
     HANDLE_ERROR( cudaEventSynchronize(stop) );
     HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
     cudaDeviceSynchronize();
     check_errors_exit(block_A, block_B);
     avgTime += time;
     if(ex == 0){minT = maxT = time;}
     else{
        if(time < minT) minT = time;
        else if(time > maxT) maxT = time;
     } 
  }
  printf("Hot Runs: Element Time for kernel: %3.4lf ms. Min: %3.4lf | Max: %3.4lf \n", avgTime/(double)REPS, minT, maxT);


/* Cold call for the copying kernel: complete rows */
  if(totalSize < TTHREADS){
     blockDims = 1;
  }
  else{
     blockDims = int(ceil(X / (float)TTHREADS));
  
  }

  for(i = 0; i < M*M; ++i){
     toarg.dst[i] = 0;
  }

  toarg.totalTransferEnts = X;
  toarg.bytes_per_block = Y; /* Number of elements to copy by each thread */
  toarg.size = sizeof(*block_A); /* Size of elements */

  HANDLE_ERROR( cudaEventCreate(&start) );
  HANDLE_ERROR( cudaEventCreate(&stop) );
  HANDLE_ERROR( cudaEventRecord(start, 0) );

  strided_memcpy_kernel<<<blockDims, TTHREADS>>>(toarg);

  HANDLE_ERROR( cudaEventRecord(stop, 0) );
  HANDLE_ERROR( cudaEventSynchronize(stop) );
  HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
  cudaDeviceSynchronize();
  printf("Cold Call: Last dimension Time for kernel: %3.4f ms \n", time);
  check_errors(block_A, block_B);

  avgTime = 0.0;
  for(ex = 0; ex < REPS; ++ex){

     for(i = 0; i < M*M; ++i){
        toarg.dst[i] = 0;
     }

     toarg.totalTransferEnts = X;
     toarg.bytes_per_block = Y; /* Number of elements to copy by each thread */
     toarg.size = sizeof(*block_A); /* Size of elements */

     HANDLE_ERROR( cudaEventCreate(&start) );
     HANDLE_ERROR( cudaEventCreate(&stop) );
     HANDLE_ERROR( cudaEventRecord(start, 0) );

     strided_memcpy_kernel<<<blockDims, TTHREADS>>>(toarg);

     HANDLE_ERROR( cudaEventRecord(stop, 0) );
     HANDLE_ERROR( cudaEventSynchronize(stop) );
     HANDLE_ERROR( cudaEventElapsedTime(&time, start, stop) );
     cudaDeviceSynchronize();
     check_errors_exit(block_A, block_B);
     avgTime += time;
     if(ex == 0){minT = maxT = time;}
     else{
        if(time < minT) minT = time;
        else if(time > maxT) maxT = time;
     }
  }
  printf("Hot Runs: Last Dimension Time for kernel: %3.4lf ms. Min: %3.4lf | Max: %3.4lf \n", avgTime/(double)REPS, minT, maxT);


  cudaFreeHost(block_A);
  cudaFreeHost(block_B);
  return 0;
}
