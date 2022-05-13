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
#endif

#endif


#endif /*_DEV_MEM_HANDLE_H*/
