/* p_cmx header file */
#include <mpi.h>

#include <stdlib.h>
#include <vector>

#include "group.hpp"
#include "environment.hpp"
#include "alloc.hpp"

namespace CMX {

/**
 * Simple Constructor
 */
Allocation::Allocation()
{
  p_environment = Environment::instance();
  p_allocation = new p_Allocation();
}

/**
 * This function does most of the setup and memory allocation.
 * @param bytes number of bytes being allocated on calling processor
 * @param group group of processors over which allocation is performed. If no
 * group is specified, assume allocation is on world group
 * @return SUCCESS if allocation is successful
 */
int Allocation::malloc(int64_t bytes, Group *group)
{
  int ret = p_allocation->malloc(bytes,group);
  p_group = p_allocation->group();
  return ret;
}

/**
 * Destructor
 */
Allocation::~Allocation()
{
  delete p_allocation;
}

/**
 * Free up a previous allocation
 * @param ptr pointer to allocation on this processor
 * @return SUCCESS if free is successful
 */
int Allocation::free()
{
  return p_allocation->free();
}

/**
 * Access a list of pointers to data on remote processors
 * Note: this returns a standard vector, which may be slow. If needed,
 * access the internal vector of data using
 *   void **myptr = &ptrs[0];
 * @param ptrs a vector of pointers to data on all processors in the allocation
 * @return CMX_SUCCESS on success
 */
int Allocation::access(std::vector<void*> &ptrs)
{
  return p_allocation->access(ptrs);
}

/**
 * Access the allocation on calling process
 * @return pointer to locally allocated data
 */
void* Allocation::access()
{
  return p_allocation->access();
}

/**
 * Access internal group
 * @return pointer to group 
 */
Group* Allocation::group()
{
  return NULL;
}

/**
 * A collective communication and operations barrier on the internal group.
 * This is equivalent to calling group()->barrier()
 * @return CMX_SUCCESS on success
 */
int Allocation::barrier()
{
  return p_allocation->barrier();
}

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
int Allocation::put(void *src, void *dst, int64_t bytes, int proc)
{
  return p_allocation->put(src,dst,bytes,proc);
}

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
int Allocation::puts(void *src, int64_t *src_stride, void *dst,
    int64_t *dst_stride, int64_t *count, int stride_levels, int proc)
{
  return p_allocation->puts(src,src_stride,dst,dst_stride,count,
      stride_levels,proc);
}

/**
 * Vector Put.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int Allocation::putv(cmx_giov_t *darr, int64_t len, int proc)
{
  return p_allocation->putv(darr, len, proc);
}

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
int Allocation::nbput(void *src, void *dst, int64_t bytes,
    int proc, cmx_request* req)
{
  return p_allocation->nbput(src,dst,bytes,proc,req);
}

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
int Allocation::nbputs(void *src, int64_t *src_stride, void *dst,
    int64_t *dst_stride, int64_t *count, int stride_levels, int proc,
    cmx_request* req)
{
  return p_allocation->nbputs(src,src_stride,dst,dst_stride,count,
      stride_levels,proc,req);
}

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
int Allocation::nbputv(cmx_giov_t *darr, int64_t len, int proc, cmx_request* req)
{
  return p_allocation->nbputv(darr, len, proc, req);
}

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
int Allocation::acc(int op, void *scale, void *src, void *dst,
    int64_t bytes, int proc)
{
  int wrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
  return p_allocation->acc(op,scale,src,dst,bytes,proc);
}

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
int Allocation::accs(int op, void *scale, void *src, int64_t *src_stride,
    void *dst, int64_t *dst_stride, int64_t *count,
    int stride_levels, int proc)
{
  return p_allocation->accs(op,scale,src,src_stride,dst,dst_stride,count,
      stride_levels,proc);
}

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
int Allocation::accv(int op, void *scale, cmx_giov_t *darr, int64_t len, int proc)
{
  return p_allocation->accv(op,scale,darr,len,proc);
}

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
int Allocation::nbacc(int op, void *scale, void *src, void *dst,
    int64_t bytes, int proc, cmx_request *req)
{
  return p_allocation->nbacc(op,scale,src,dst,bytes,proc,req);
}

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
int Allocation::nbaccs(int op, void *scale, void *src, int64_t *src_stride,
    void *dst, int64_t *dst_stride, int64_t *count,
    int stride_levels, int proc, cmx_request *req)
{
  return p_allocation->nbaccs(op,scale,src,src_stride,dst,dst_stride,count,
      stride_levels,proc,req);
}

/**
 *  NonblockingVector Atomic Accumulate.
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
int Allocation::nbaccv(int op, void *scale, cmx_giov_t *darr, int64_t len,
    int proc, cmx_request *req)
{
  return p_allocation->nbaccv(op,scale,darr,len,proc,req);
}

/**
 * Contiguous Get.
 *
 * @param[in] src pointer to source buffer
 * @param[in] dst pointer to destination buffer
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int Allocation::get(void *src, void *dst, int64_t bytes, int proc)
{
  return p_allocation->get(src,dst,bytes,proc);
}

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
int Allocation::gets(void *src, int64_t *src_stride, void *dst,
    int64_t *dst_stride, int64_t *count, int stride_levels, int proc)
{
  return p_allocation->gets(src,src_stride,dst,dst_stride,count,stride_levels,proc);
}

/**
 * Vector Get.
 *
 * @param[in] darr descriptor array
 * @param[in] len length of descriptor array
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int Allocation::getv(cmx_giov_t *darr, int64_t len, int proc)
{
  return p_allocation->getv(darr, len, proc);
}

/**
 * Nonblocking Contiguous Get.
 *
 * @param[in] src pointer to source buffer
 * @param[in] dst pointer to destination buffer
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int Allocation::nbget(void *src, void *dst, int64_t bytes,
    int proc, cmx_request *req)
{
  return p_allocation->nbget(src,dst,bytes,proc,req);
}

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
int Allocation::nbgets(void *src, int64_t *src_stride, void *dst,
    int64_t *dst_stride, int64_t *count, int stride_levels, int proc,
    cmx_request *req)
{
  return p_allocation->nbgets(src,src_stride,dst,dst_stride,
      count,stride_levels,proc,req);
}

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
int Allocation::nbgetv(cmx_giov_t *darr, int64_t len, int proc, cmx_request *req)
{
  return p_allocation->nbgetv(darr, len, proc, req);
}

/**
 * Flush all outgoing messages from me to the given proc on the group of the
 * allocation.
 * @param[in] proc the proc with which to flush outgoing messages
 * @return CMX_SUCCESS on success
 */
int Allocation::fenceProc(int proc)
{
  return p_allocation->fenceProc(proc);
  return CMX_SUCCESS;
}

/**
 * Flush all outgoing messages to all procs in group of the allocation
 * @return CMX_SUCCESS on success
 */
int Allocation::fenceAll()
{
  p_allocation->fenceAll();
  return CMX_SUCCESS;
}

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
 * @param[in] prem pointer to remote data value
 * @param[in] extra for CMX_FETCH_AND_ADD and CMX_FETCH_AND_ADD_LONG, the
 *            amount to increment the remote value by
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation
 * @return CMX_SUCCESS on success
 */
int Allocation::readModifyWrite(int op, void *ploc, void *prem,
    int extra, int proc)
{
  return p_allocation->readModifyWrite(op,ploc,prem,extra,proc);
}

/**
 * Waits for completion of non-blocking cmx operations with explicit handles.
 * @param[in] req the request handle
 * @return CMX_SUCCESS on success
 */
int Allocation::wait(cmx_request *req)
{
  return p_allocation->wait(req);
}

/**
 * Checks completion status of non-blocking cmx operations with explicit
 * handles.
 * @param[in] req the request handle
 * @return true if operation completed
 */
bool Allocation::test(cmx_request *req)
{
  return p_allocation->test(req);
}

/**
 * Wait for all outstanding implicit non-blocking operations to finish on the
 * group of the allocation
 * @return CMX_SUCCESS on success
 */
int Allocation::waitAll()
{
  return CMX_SUCCESS;
}

/**
 * Wait for all outstanding implicit non-blocking operations to a particular
 * process to finish. Proc is in the group of the allocation
 * @param[in] proc proc for which all the outstanding non-blocking operations
 * have to be completed
 * @return CMX_SUCCESS on success
 */
int Allocation::waitProc(int proc)
{
  return CMX_SUCCESS;
}

}; // CMX namespace
