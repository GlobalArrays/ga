/* cmx header file */
#ifndef _CMX_H
#define _CMX_H

#include <mpi.h>

#include <stdlib.h>

#include "cmx_impl.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "c" {
#endif

typedef _cmx_handle cmx_handle_t;

typedef struct {
    void **loc; /**< array of local starting addresses */
    cmxInt *rem; /**< array of remote offsets */
    cmxInt count; /**< size of address arrays (src[count],dst[count]) */
    cmxInt bytes; /**< length in bytes for each src[i]/dst[i] pair */
} cmx_giov_t;

typedef _cmx_request cmx_request_t;

#define CMX_SUCCESS 0
#define CMX_FAILURE 1

#define CMX_SWAP 10
#define CMX_SWAP_LONG 11
#define CMX_FETCH_AND_ADD 12
#define CMX_FETCH_AND_ADD_LONG 13

#define CMX_ACC_OFF 36
#define CMX_ACC_INT (CMX_ACC_OFF + 1)
#define CMX_ACC_DBL (CMX_ACC_OFF + 2)
#define CMX_ACC_FLT (CMX_ACC_OFF + 3)
#define CMX_ACC_CPL (CMX_ACC_OFF + 4)
#define CMX_ACC_DCP (CMX_ACC_OFF + 5)
#define CMX_ACC_LNG (CMX_ACC_OFF + 6)

#define CMX_MAX_STRIDE_LEVEL 8

#define CMX_NOT_SET 0
#define CMX_INT     1
#define CMX_LONG    2
#define CMX_FLOAT   3
#define CMX_DOUBLE  4
#define CMX_COMPLEX 5
#define CMX_DCMPLX  6
#define CMX_USER    7

/**
 * Initialize cmx.
 *
 * @return CMX_SUCCESS on success
 */
extern int cmx_init();

/**
 * Initialize cmx with command line arguments.
 *
 * @return CMX_SUCCESS on success
 */
extern int cmx_init_args(int *argc, char ***argv);

/**
 * Test whether cmx has been initialized.
 *
 * @return CMX_SUCCESS if cmx has been initialized
 *         CMX_FAILURE if cmx has not
 */
extern int cmx_initialized();

/**
 * Terminate cmx and clean up resources.
 *
 * @return CMX_SUCCESS on success
 */
extern int cmx_finalize();

/**
 * Abort cmx, printing the msg, and exiting with code.
 *
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
extern void cmx_error(char *msg, int code);

/**
 * Create a new group from the given group and process ID list.
 *
 * The rank list selects the ranks from the given group to become members of
 * the new group. The ranks should be nonnegative and range from zero to the
 * size of the given group.
 *
 * This functions is collective only over the ranks within the rank list and
 * not over the entire original group.
 *
 * @param[in] n the number of ranks to select for the new group
 * @param[in] rank_list the list of ranks to select for the new group
 * @param[in] group the group to subset for the new group
 * @param[out] new_group the newly created group
 * @return CMX_SUCCESS on success
 *         CMX_FAILURE if a rank in the rank list is out of bounds
 */
extern int cmx_group_create(
        int n, int *pid_list, cmx_group_t group, cmx_group_t *new_group);

/**
 * Marks the group for deallocation.
 *
 * @param[in] group group to be destroyed
 * @return CMX_SUCCESS on success
 */
extern int cmx_group_free(cmx_group_t group);

/**
 * Determines the rank of the calling process in the given group.
 *
 * @param[in] group group handle
 * @param[out] rank rank of the calling process in the group
 * @return CMX_SUCCESS on success
 */
extern int cmx_group_rank(cmx_group_t group, int *rank);

/**
 * Determines the size of the given group.
 *
 * @param[in] group group handle
 * @param[out] size number of processes in the group
 * @return CMX_SUCCESS on success
 */
extern int cmx_group_size(cmx_group_t group, int *size);

/**
 * Returns the MPI_Comm object backing the given group.
 *
 * The actual MPI_Comm object is returned, therefore do not call
 * MPI_Comm_free() on the returned communicator. This function is for
 * convenience to be able to MPI_Comm_dup() the returned MPI_Comm instance.
 *
 * @param[in] group group handle
 * @param[out] comm the communicator handle
 * @return CMX_SUCCESS on success
 */
extern int cmx_group_comm(cmx_group_t group, MPI_Comm *comm);

/**
 * Translates the ranks of processes in one group to those in another group.
 *
 * @param[in] n the number of ranks in the ranks_from and ranks_to arrays
 * @param[in] group_from the group to translate ranks from 
 * @param[in] ranks_from array of zer or more valid ranks in group_from
 * @param[in] group_to the group to translate ranks to 
 * @param[out] ranks_to array of corresponding ranks in group_to
 * @return CMX_SUCCESS on success
 */
extern int cmx_group_translate_ranks(int n,
        cmx_group_t group_from, int *ranks_from,
        cmx_group_t group_to, int *ranks_to);

/**
 * Translate the given rank from its group to its corresponding rank in the
 * world group.
 *
 * Shorthand notation for common case.
 *
 * @param[in] group the group to translate from
 * @param[in] group_rank the rank to translate from
 * @param[out] world_rank the corresponding world rank
 * @return CMX_SUCCESS on success
 */
extern int cmx_group_translate_world(
        cmx_group_t group, int group_rank, int *world_rank);
/**
 * Extact group object from CMX allocation handle
 *
 * @param[in] handle CMX handle for data allocation
 * @param[out] group CMX group associated with CMX data allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_get_group_from_handle(cmx_handle_t handle, cmx_group_t *group);

/**
 * A collective communication and operations barrier.
 *
 * Ensures all cmx communication has completed prior to performing the
 * operations barrier.
 *
 * @param[in] group the group to perform the collective barrier over
 * @return CMX_SUCCESS on success
 */
extern int cmx_barrier(cmx_group_t group);

/**
 * Contiguous Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_put(
        void *src, cmxInt dst_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Strided Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride array of strides at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] dst_stride array of strides at destination
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_puts(
        void *src, cmxInt *src_stride,
        cmxInt dst_offset, cmxInt *dst_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Vector Put.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_putv(
        cmx_giov_t *darr, cmxInt len,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Nonblocking Contiguous Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
extern int cmx_nbput(
        void *src, cmxInt dst_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t* req);

/**
 * Nonblocking Strided Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride array of strides at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] dst_stride array of strides at destination
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
extern int cmx_nbputs(
        void *src, cmxInt *src_stride,
        cmxInt dst_offset, cmxInt *dst_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t* req);

/**
 * Nonblocking Vector Put.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
extern int cmx_nbputv(
        cmx_giov_t *darr, cmxInt len,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t* req);

/**
 * Contiguous Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_acc(
        int op, void *scale,
        void *src, cmxInt dst_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Strided Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride [stride_levels] array of strides at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] dst_stride [stride_levels] array of strides at destination
 * @param[in] count [stride_levels+1] number of units at each stride level
 *            count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_accs(
        int op, void *scale,
        void *src, cmxInt *src_stride,
        cmxInt dst_offset, cmxInt *dst_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Vector Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_accv(
        int op, void *scale,
        cmx_giov_t *darr, int len,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Nonblocking Contiguous Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
extern int cmx_nbacc(
        int op, void *scale,
        void *src, cmxInt dst_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *req);

/**
 * Strided Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride array of strides at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] dst_stride array of strides at destination
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
extern int cmx_nbaccs(
        int  op, void *scale,
        void *src, cmxInt *src_stride,
        cmxInt dst_offset, cmxInt *dst_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *req);

/**
 * Vector Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
extern int cmx_nbaccv(
        int op, void *scale,
        cmx_giov_t *darr, cmxInt len,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *req);

/**
 * Contiguous Get.
 *
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] src_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_get(
        void *dst, cmxInt src_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Strided Get.
 *
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] dst_stride array of strides at destination
 * @param[in] src_offset offset from start of data allocation on remote
 *            process
 * @param[in] src_stride array of strides at source
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_gets(
        void *dst, cmxInt *dst_stride,
        cmxInt src_offset, cmxInt *src_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Vector Get.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_getv(
        cmx_giov_t *darr, int len,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Nonblocking Contiguous Get.
 *
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] src_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
extern int cmx_nbget(
        void *dst, cmxInt src_offset, cmxInt bytes,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *req);

/**
 * Nonblocking Strided Get.
 *
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] dst_stride array of strides at destination
 * @param[in] src_offset offset from start of data allocation on remote
 *            process
 * @param[in] src_stride array of strides at source
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
extern int cmx_nbgets(
        void *dst, cmxInt *dst_stride,
        cmxInt src_offset, cmxInt *src_stride,
        cmxInt *count, int stride_levels,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *req);

/**
 * Nonblocking Vector Get.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation. The calling process
 *            and remote process must belong to the same group as the
 *            allocation
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
extern int cmx_nbgetv(
        cmx_giov_t *darr, cmxInt len,
        int proc, cmx_handle_t cmx_hdl,
        cmx_request_t *req);

/**
 * Collective allocation of registered memory and exchange of addresses.
 *
 * @param[out] cmx_hdl handle describing data allocation that should be
 *             used in all operations using this allocation
 * @param[in] bytes how many bytes to allocate locally
 * @param[in] group the group to which the calling process belongs
 * @return CMX_SUCCESS on success
 */
extern int cmx_malloc(
        cmx_handle_t *cmx_hdl, cmxInt bytes, cmx_group_t group);

/**
 * Access local buffer from CMX handle
 * @param[in] handle CMX handle for data allocation
 * @param buf[out] pointer to local buffer
 * @return CMX_SUCCESS on success
 */
extern int cmx_access(cmx_handle_t cmx_hdl, void **buf);

/**
 * Collective free of memory given the original local pointer.
 *
 * @param[in] cmx_hdl handle for data allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_free(cmx_handle_t cmx_hdl);

/**
 * Local (noncollective) allocation of registered memory.
 *
 * Using memory allocated here may have performance benefits when used as a
 * communication buffer.
 *
 * @param[in] bytes how many bytes to allocate locally
 * @return CMX_SUCCESS on success
 */
extern void* cmx_malloc_local(size_t bytes);

/**
 * Local (noncollective) free of memory allocated by cmx_malloc_local.
 *
 * @param[in] the original local memory allocated using cmx_malloc_local
 * @return CMX_SUCCESS on success
 */
extern int cmx_free_local(void *ptr);

/**
 * Flush all outgoing messages from me to the given proc.
 *
 * @param[in] proc the proc with which to flush outgoing messages
 * @return CMX_SUCCESS on success
 */
extern int cmx_fence_proc(int proc, cmx_group_t group);

/**
 * Flush all outgoing messages to all procs in group.
 *
 * @param[in] group flush operation if performed on all processors in group
 * @return CMX_SUCCESS on success
 */
extern int cmx_fence_all(cmx_group_t group);

/**
 * Collectively create num locks locally.
 *
 * Remote procs may create a different number of locks, including zero.
 *
 * This function is always collective on the world group.
 *
 * @param[in] num number of locks to create locally
 * @return CMX_SUCCESS on success
 */
extern int cmx_create_mutexes(int num);

/**
 * Collectively destroy all previously created locks.
 *
 * This function is always collective on the world group.
 *
 * @param[in] num number of locks to create locally
 * @return CMX_SUCCESS on success
 */
extern int cmx_destroy_mutexes();

/**
 * Lock the given mutex on the given proc.
 *
 * This function is always on the world group.
 *
 * @param[in] mutex the ID of the mutex to lock on proc
 * @param[in] the ID of the proc which owns the mutex
 *
 * @return CMX_SUCCESS on success
 *         CMX_FAILURE if given mutex or proc is out of range
 */
extern int cmx_lock(int mutex, int proc);

/**
 * Unlock the given mutex on the given proc.
 *
 * This function is always on the world group.
 *
 * @param[in] mutex the ID of the mutex to unlock on proc
 * @param[in] the ID of the proc which owns the mutex
 *
 * @return CMX_SUCCESS on success
 *         CMX_FAILURE if given mutex or proc is out of range
 */
extern int cmx_unlock(int mutex, int proc);

/**
 * Read-modify-write atomic operation.
 *
 * The operations may be one of
 *  - CMX_SWAP
 *  - CMX_SWAP_LONG
 *  - CMX_FETCH_AND_ADD
 *  - CMX_FETCH_AND_ADD_LONG
 *
 * For the swap operations, the extra parameter is not used. The values of the
 * ploc and prem locations are swapped.
 *
 * For the fetch and add operations, the extra parameter is also used to
 * indicate how much to increment the remote value. The original remove value
 * is returned in the ploc parameter.
 *
 * @param[in] op the operation to perform (see list above)
 * @param[in] ploc the value to update locally
 * @param[in] rem_offset offset to remote value
 * @param[in] extra for CMX_FETCH_AND_ADD and CMX_FETCH_AND_ADD_LONG, the
 *            amount to increment the remote value by
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation
 * @return CMX_SUCCESS on success
 */
extern int cmx_rmw(
        int op, void *ploc, cmxInt rem_offset, int extra,
        int proc, cmx_handle_t cmx_hdl);

/**
 * Waits for completion of non-blocking cmx operations with explicit handles.
 *
 * @param[in] req the request handle
 * @return CMX_SUCCESS on success
 */
extern int cmx_wait(cmx_request_t *req);

/**
 * Checks completion status of non-blocking cmx operations with explicit
 * handles.
 *
 * @param[in] req the request handle
 * @param[out] status 0-completed, 1-in progress
 * @return CMX_SUCCESS on success
 */
extern int cmx_test(cmx_request_t *req, int *status);

/**
 * Wait for all outstanding implicit non-blocking operations to finish.
 *
 * @param[in] group group handle
 * @return CMX_SUCCESS on success
 */
extern int cmx_wait_all(cmx_group_t group);

/**
 * Wait for all outstanding implicit non-blocking operations to a particular
 * process to finish.
 *
 * @param[in] proc proc for which all the outstanding non-blocking operations
 * have to be completed
 * @param[in] group group handle
 * @return CMX_SUCCESS on success
 */
extern int cmx_wait_proc(int proc, cmx_group_t group);

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* _CMX_H */
