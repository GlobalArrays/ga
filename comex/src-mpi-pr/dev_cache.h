#ifndef _DEV_CACHE_H_
#define _DEV_CACHE_H_

#include <cuda_runtime.h>

#include "groups.h"

/**
 * struct for passing around cudaMe
 */
typedef struct _dev_entry_t {
  struct _dev_entry_t *next;  /**< next memory region in list */
  int rank;                   /**< rank of processor holding memory region */
  void *ptr;                  /**< pointer to device memory segment */
  cudaIpcMemHandle_t handle;  /**< cuda memory handle for pointer */
} dev_entry_t;


/* functions
 *
 * documentation is in the *.c file
 */

#if 0
static void dev_cache_init(MPI_Comm comm);
static void dev_cache_destroy();
static void dev_cache_exchange(void *ptr, void ***ptrs, comex_group_t group);
static void dev_cache_insert(int rank, void *ptr);
#endif
#if 0
void dev_cache_launch(void *ptr, int op);
void dev_cache_open(int rank, void *ptr);
void dev_cache_close(int rank, void *ptr);
void dev_cache_remove(void *ptr);
#endif

#endif /* _DEV_CACHE_H_ */
