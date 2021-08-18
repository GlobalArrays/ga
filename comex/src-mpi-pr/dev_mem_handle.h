#ifndef _DEV_MEM_HANDLE_H
#define _DEV_MEM_HANDLE_H

#include <cuda_runtime.h>

#ifdef ENABLE_DEVICE
typedef struct {
  cudaIpcMemHandle_t handle;
} devMemHandle_t;
#endif


#endif /*_DEV_MEM_HANDLE_H*/
