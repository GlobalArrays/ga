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
    comex_igroup_t *igroup;     /**< link to group that generated window */
    MPI_Win win;                /**< handle to MPI window containing this region */
    int rank;                   /**< rank where this region lives */
    void *buf;                  /**< starting address of region */
    size_t len;                 /**< length of region */
    struct _reg_entry_t *next;  /**< next memory region in list */
} reg_entry_t;

/* functions
 *
 * documentation is in the *.c file
 */

reg_return_t reg_win_init(int nprocs);
reg_return_t reg_win_destroy();
reg_entry_t *reg_win_find(int rank, void *buf, int len);
reg_entry_t *reg_win_insert(int rank, void *buf, int len, MPI_Win win,
    comex_igroup_t *group);
reg_return_t reg_win_delete(int rank, void *buf);

#endif /* _REG_WINDOW_H_ */
