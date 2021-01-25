#ifndef _REG_WINDOW_H_
#define _REG_WINDOW_H_

#include <stddef.h>
#include <mpi.h>

/**
 * Enumerate the return codes for registration window functions.
 */
typedef enum _reg_return_t {
    RR_SUCCESS=0,   /**< success */
    RR_FAILURE      /**< non-specific failure */
} reg_return_t;

/**
 * A registered contiguous memory region.
 */
typedef struct _reg_entry_t {
    cmx_handle_t* hdl;
    int rank;                   /**< global rank where this region lives */
    void* buf;                  /**< starting address of region */
    size_t len;                 /**< length of region */
    struct _reg_entry_t *next;  /**< next memory region in list */
} reg_entry_t;

/* functions
 *
 * documentation is in the *.c file
 */

extern reg_return_t reg_entry_init(int nprocs);
extern reg_return_t reg_entries_destroy();
extern reg_entry_t *reg_entry_find(int rank, void *buf, int len);
extern reg_entry_t *reg_entry_insert(int world_rank, void *buf,
    int len, cmx_handle_t *hdl);
extern reg_return_t reg_entry_delete(int rank, void *buf);

#endif /* _REG_WINDOW_H_ */
