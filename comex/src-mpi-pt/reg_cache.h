#ifndef _REG_CACHE_H_
#define _REG_CACHE_H_

#include <stddef.h>

/**
 * Enumerate the return codes for registration cache functions.
 */
typedef enum _reg_return_t {
    RR_SUCCESS=0,   /**< success */
    RR_FAILURE      /**< non-specific failure */
} reg_return_t;

/**
 * A registered contiguous memory region.
 */
typedef struct _reg_entry_t {
    int rank;                   /**< rank where this region lives */
    void *buf;                  /**< starting address of region */
    size_t len;                 /**< length of region */
    char name[SHM_NAME_SIZE];   /**< name of region */
    void *mapped;               /**< starting address of mmap'd region */
    struct _reg_entry_t *next;  /**< next memory region in list */
} reg_entry_t;

/* functions
 *
 * documentation is in the *.c file
 */

reg_return_t reg_cache_init(int nprocs);
reg_return_t reg_cache_destroy();
reg_entry_t *reg_cache_find(int rank, void *buf, size_t len);
reg_entry_t *reg_cache_insert(int rank, void *buf, size_t len, const char *name, void *mapped);
reg_return_t reg_cache_delete(int rank, void *buf);
reg_return_t reg_cache_nullify(reg_entry_t *entry);

#endif /* _REG_CACHE_H_ */
