/**
 * @file nb.h
 * @brief Simple non-blocking request table for the SHMEM COMEX backend.
 *
 * This header provides a small fixed-size table used to track non-blocking
 * operations issued by the backend. It is intentionally simple: entries
 * are reserved with `comex_nb_reserve()` and released with
 * `comex_nb_release()`.
 */

#ifndef COMEX_NB_H
#define COMEX_NB_H

#include <stddef.h>

/** Maximum number of outstanding non-blocking operations tracked. */
#define COMEX_MAX_NB_OUTSTANDING 128

/**
 * Entry describing a non-blocking operation.
 *
 * Fields:
 * - `active`: non-zero when entry is in use
 * - `op`: operation type (0=put, 1=get)
 * - `src`, `dst`: pointers used for the operation
 * - `bytes`: size in bytes
 * - `target_pe`: target PE id
 */
typedef struct {
    int active;       /**< non-zero when entry is active */
    int op;           /**< 0=put, 1=get, others reserved */
    void *src;        /**< source pointer */
    void *dst;        /**< destination pointer */
    int bytes;        /**< size in bytes */
    int target_pe;    /**< target PE id */
    void *aux;        /**< backend auxiliary data (opaque) */
} comex_nb_entry_t;

/**
 * Reserve an index for a new non-blocking operation.
 *
 * @return index in the NB table, or -1 if no slot is available
 */
int comex_nb_reserve();

/**
 * Release a previously reserved NB table index.
 *
 * @param idx index returned by `comex_nb_reserve()`
 */
void comex_nb_release(int idx);

/**
 * Return a pointer to the NB table entry for a given index.
 *
 * @param idx index in the NB table
 * @return pointer to the entry or NULL on invalid index
 */
comex_nb_entry_t* comex_nb_get_entry(int idx);

#endif /* COMEX_NB_H */
