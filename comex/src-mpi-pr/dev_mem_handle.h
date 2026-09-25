#ifndef _DEV_MEM_HANDLE_H
#define _DEV_MEM_HANDLE_H


#if defined(ENABLE_DEVICE)

#if defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#include <rocblas.h>
// #include <hip/hip_runtime_api.h>
typedef struct {
  hipIpcMemHandle_t handle;
} devMemHandle_t;

#elif defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#include "cublas_v2.h"
typedef struct {
  cudaIpcMemHandle_t handle;
} devMemHandle_t;

#if defined(ENABLE_NVSHMEM)
#define CUDA_CHECK(stmt)                                  \
do {                                                      \
    cudaError_t result = (stmt);                          \
    if (cudaSuccess != result) {                          \
        fprintf(stderr, "[%s:%d] CUDA failed with %s \n", \
         __FILE__, __LINE__, cudaGetErrorString(result)); \
        exit(-1);                                         \
    }                                                     \
} while (0)
#endif

#endif

#endif


#endif /*_DEV_MEM_HANDLE_H*/
