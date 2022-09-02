#ifndef _REG_CACHE_H_
#define _REG_CACHE_H_

#if USE_SICM
#include <sicm_low.h>
//#include <sicm_impl.h>
#endif
#include <stddef.h>
// #define ENABLE_SYSV
#ifdef ENABLE_SYSV
#include <sys/ipc.h>
#endif

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
    struct _reg_entry_t *next;  /**< next memory region in list */
    void *buf;                  /**< starting address of region */
    size_t len;                 /**< length of region */
    void *mapped;               /**< starting address of mmap'd region */
    int rank;                   /**< rank where this region lives */
    int use_dev;                /**< memory is on a device */
#ifdef ENABLE_SYSV
    char name[2*SHM_NAME_SIZE];   /**< name of region */
    key_t key;
#else
    char name[SHM_NAME_SIZE];   /**< name of region */
#endif
#if USE_SICM
#if SICM_OLD
    sicm_device *device;         /**< pointer to SICM device */
#else
    sicm_device_list device;         /**< pointer to SICM device */
#endif
#endif
} reg_entry_t;

/* functions
 *
 * documentation is in the *.c file
 */

reg_return_t reg_cache_init(int nprocs);
reg_return_t reg_cache_destroy();
reg_entry_t *reg_cache_find(int rank, void *buf, size_t len);
reg_entry_t *reg_cache_insert(int rank, void *buf, size_t len,
#ifdef ENABLE_SYSV
    const char *name, key_t key, void *mapped, int use_dev
#else
    const char *name, void *mapped, int use_dev
#endif
#if USE_SICM
#if SICM_OLD
    ,sicm_device *device
#else
    ,sicm_device_list device
#endif
#endif
    );
reg_return_t reg_cache_delete(int rank, void *buf);
reg_return_t reg_cache_nullify(reg_entry_t *entry);

#endif /* _REG_CACHE_H_ */
