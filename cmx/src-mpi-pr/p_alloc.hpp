/* p_cmx header file */
#ifndef _P_ALLOC_H
#define _P_ALLOC_H

#include <mpi.h>

#include <stdlib.h>
#include <vector>

#include "p_structs.hpp"
#include "node_config.hpp"
#include "shmem.hpp"
#include "group.hpp"
#include "environment.hpp"
#include "p_environment.hpp"

namespace CMX {

class p_Allocation {
public:

/**
 * Simple Constructor
 */
p_Allocation();

/**
 * Destructor
 */
~p_Allocation();

/**
 * This function does most of the setup and memory allocation.
 * @param bytes number of bytes being allocated on calling processor
 * @param group group of processors over which allocation is performed. If no
 * group is specified, assume allocation is on world group
 * @return SUCCESS if allocation is successful
 */
int malloc(int64_t bytes, Group *group=NULL);

/**
 * Free up a previous allocation
 * @return SUCCESS if free is successful
 */
int free();

/**
 * Access a list of pointers to data on remote processors
 * Note: this returns a standard vector, which may be slow. If needed,
 * access the internal vector of data using
 *   void **myptr = &ptrs[0];
 * @param ptrs a vector of pointers to data on all processors in the allocation
 * @return CMX_SUCCESS on success
 */
int access(std::vector<void*> &ptrs);

/**
 * Access the allocation on calling process
 * @return pointer to locally allocated data
 */
void* access();

/**
 * Access internal group
 * @return pointer to group 
 */
Group* group();

/**
 * A collective communication and operations barrier on the internal group.
 * This is equivalent to calling group()->barrier()
 * @return CMX_SUCCESS on success
 */
int barrier();

/**
 * Contiguous Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst pointer to destination buffer
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int put(void *src, void *dst, int64_t bytes, int proc);

/**
 * Strided Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride [stride_levels] array of strides at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] dst_stride [stride_levels] array of strides at destination
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int puts(void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
        int64_t *count, int stride_levels, int proc);

/**
 * Vector Put.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int putv(_cmx_giov_t *darr, int64_t len, int proc);

/**
 * Nonblocking Contiguous Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbput(void *src, void *dst, int64_t bytes, int proc, _cmx_request* req);

/**
 * Nonblocking Strided Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride [stride_levels] array of strides at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] dst_stride [stride_levels] array of strides at destination
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbputs(void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
        int64_t *count, int stride_levels, int proc, _cmx_request* req);

/**
 * Nonblocking Vector Put.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbputv(_cmx_giov_t *darr, int64_t len, int proc, _cmx_request* req);

/**
 * Contiguous Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int acc(int op, void *scale, void *src, void *dst, int64_t bytes, int proc);

/**
 * Strided Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride [stride_levels] array of strides at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] dst_stride [stride_levels] array of strides at destination
 * @param[in] count [stride_levels+1] number of units at each stride level
 *            count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int accs(int op, void *scale, void *src, int64_t *src_stride, void *dst,
    int64_t *dst_stride, int64_t *count, int stride_levels, int proc);

/**
 * Vector Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int accv(int op, void *scale, _cmx_giov_t *darr, int64_t len, int proc);

/**
 * Nonblocking Contiguous Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbacc(int op, void *scale, void *src, void *dst,
    int64_t bytes, int proc, _cmx_request *req);

/**
 * Nonblocking Strided Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride [stride_levels] array of strides at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] dst_stride [stride_levels] array of strides at destination
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbaccs(int op, void *scale, void *src, int64_t *src_stride,
    void *dst, int64_t *dst_stride, int64_t *count,
    int stride_levels, int proc, _cmx_request *req);

/**
 * Nonblocking Vector Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbaccv(int op, void *scale, _cmx_giov_t *darr, int64_t len, int proc, _cmx_request *req);

/**
 * Contiguous Get.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int get(void *src, void *dst, int64_t bytes, int proc);

/**
 * Strided Get.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride [stride_levels] array of strides at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] dst_stride [stride_levels] array of strides at destination
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int gets(void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc);

/**
 * Vector Get.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int getv(_cmx_giov_t *darr, int64_t len, int proc);

/**
 * Nonblocking Contiguous Get.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbget(void *src, void *dst, int64_t bytes, int proc, _cmx_request *req);

/**
 * Nonblocking Strided Get.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride [stride_levels] array of strides at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] dst_stride [stride_levels] array of strides at destination
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbgets(void *src, int64_t *src_stride, void *dst, int64_t *dst_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *req);

/**
 * Nonblocking Vector Get.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbgetv(_cmx_giov_t *darr, int64_t len, int proc, _cmx_request *req);

/**
 * Flush all outgoing messages from me to the given proc on the group of the
 * allocation.
 * @param[in] proc the proc with which to flush outgoing messages
 * @return CMX_SUCCESS on success
 */
int fenceProc(int proc);

/**
 * Flush all outgoing messages to all procs in group of the allocation
 * @return CMX_SUCCESS on success
 */
int fenceAll();

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
 * indicate how much to increment the remote value. The original remote value
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
int readModifyWrite(int op, void *ploc, int64_t rem_offset, int extra, int proc);

/**
 * Waits for completion of non-blocking cmx operations with explicit handles.
 * @param[in] req the request handle
 * @return CMX_SUCCESS on success
 */
int wait(_cmx_request *req);

/**
 * Checks completion status of non-blocking cmx operations with explicit
 * handles.
 * @param[in] req the request handle
 * @param[out] status true->completed, false->in progress
 * @return CMX_SUCCESS on success
 */
int test(_cmx_request *req, bool *status);

/**
 * Wait for all outstanding implicit non-blocking operations to finish on the
 * group of the allocation
 * @return CMX_SUCCESS on success
 */
int waitAll();

/**
 * Wait for all outstanding implicit non-blocking operations to a particular
 * process to finish. Proc is in the group of the allocation
 * @param[in] proc proc for which all the outstanding non-blocking operations
 * have to be completed
 * @return CMX_SUCCESS on success
 */
int waitProc(int proc);

private:

  Group *p_group; // Group associated with allocation

  p_Environment *p_impl_environment;

  Environment *p_environment;

  int p_datatype;   // enumeration describing data type of allocation

  std::vector<void*> p_list; //list list location of buffers on all processors

  int p_rank; // rank id

  void *p_buf; // pointer to allocation on rank p_rank

  size_t p_bytes; // size of allocation

  p_NodeConfig *p_config; // copy of the node configuration object
};

}; // CMX namespace

#endif /* _P_ALLOC_H */
