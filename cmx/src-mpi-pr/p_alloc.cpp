#include "group.hpp"
#include "p_environment.hpp"
#include "p_alloc.hpp"
#include "environment.hpp"
namespace CMX {

/**
 * Simple Constructor
 */
p_Allocation::p_Allocation()
{
  p_environment = Environment::instance();
  p_impl_environment = p_Environment::instance();
  p_config = NULL;
}

/**
 * This function does most of the setup and memory allocation. It is used to
 * simplify the writing of the actual constructor, which handles the data type.
 * @param bytes number of bytes being allocated on calling processor
 * @param group group of processors over which allocation is performed. If no
 * group is specified, assume allocation is on world group
 * @return SUCCESS if allocation is successful
 */
int p_Allocation::malloc(int64_t bytes, Group *group)
{
  // set group variable to world group if group not specified
  if (group == NULL) {
    p_group = p_environment->getWorldGroup();
  } else {
    p_group = group;
  }

  int nsize = group->size();

  p_list.resize(nsize);

  p_impl_environment->dist_malloc(&p_list[0], bytes, group);

  return CMX_SUCCESS;
}

/**
 * Free up a previous allocation
 * @return SUCCESS if free is successful
 */
int p_Allocation::free()
{
  int me = p_group->rank();
  p_impl_environment->dist_free(p_list[me], p_group);
  p_list.clear();
  return CMX_SUCCESS;
}

/**
 * Destructor
 */
p_Allocation::~p_Allocation()
{
  int me = p_group->rank();
  p_list.clear();
}

/**
 * Access the allocation on calling process
 * @return pointer to locally allocated data
 */
void* p_Allocation::access()
{
  return p_list[p_group->rank()];
}

/**
 * Access a list of pointers to data on remote processors
 * Note: this returns a standard vector, which may be slow. If needed,
 * access the internal vector of data using
 *   void **myptr = &ptrs[0];
 * @param ptrs a vector of pointers to data on all processors in the allocation
 * @return CMX_SUCCESS on success
 */
int p_Allocation::access(std::vector<void*> &ptrs)
{
  int i;
  int size = p_list.size();
  ptrs.clear();
  for (i=0; i<size; i++) {
    ptrs.push_back(p_list[i]);
  }
  return CMX_SUCCESS;
}

/**
 * Access internal group
 * @return pointer to group 
 */
Group* p_Allocation::group()
{
  return p_group;
}

/**
 * A collective communication and operations barrier on the internal group.
 * This is equivalent to calling group()->barrier()
 * @return CMX_SUCCESS on success
 */
int p_Allocation::barrier()
{
  return p_group->barrier();
}

/**
 * Contiguous Put.
 *
 * @param[in] src pointer to source buffer
 * @param[in] dst pointer to destination buffer
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int p_Allocation::put(void *src, void *dst, int64_t bytes, int proc)
{
  cmx_request request;
  int wrank;
  p_environment->translateWorld(1,p_group,&proc,&wrank);
  p_impl_environment->nb_register_request(&request);
  p_impl_environment->nb_put(src,dst,bytes,wrank,&request);
  p_impl_environment->nb_wait_for_all(&request);
  return CMX_SUCCESS;
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
int p_Allocation::puts(void *src, int64_t *src_stride, void *dst,
    int64_t *dst_stride, int64_t *count, int stride_levels, int proc)

{
  cmx_request request;
  int wrank;
  p_environment->translateWorld(1,p_group,&proc,&wrank);
  p_impl_environment->nb_register_request(&request);
  p_impl_environment->nb_puts(src,src_stride,dst,dst_stride,count,
      stride_levels,wrank,&request);
  p_impl_environment->nb_wait_for_all(&request);
  return CMX_SUCCESS;
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
int p_Allocation::putv(_cmx_giov_t *darr, int64_t len, int proc)
{
  return CMX_SUCCESS;
}

/**
 * Nonblocking Contiguous Put.
 *
 * @param[in] src pointer to source buffer
 * @param[in] dst pointer to destination buffer
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int p_Allocation::nbput(void *src, void *dst, int64_t bytes, int proc,
    _cmx_request* req)
{
  int wrank;
  p_environment->translateWorld(1,p_group,&proc,&wrank);
  p_impl_environment->nb_register_request(req);
  p_impl_environment->nb_put(src,dst,bytes,wrank,req);
  return CMX_SUCCESS;
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
int p_Allocation::nbputs(void *src, int64_t *src_stride, void *dst,
    int64_t *dst_stride, int64_t *count, int stride_levels, int proc,
    _cmx_request* req)
{
  int wrank;
  p_environment->translateWorld(1,p_group,&proc,&wrank);
  p_impl_environment->nb_register_request(req);
  p_impl_environment->nb_puts(src,src_stride,dst,dst_stride,count,
      stride_levels,wrank,req);
  return CMX_SUCCESS;
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
int p_Allocation::nbputv(_cmx_giov_t *darr, int64_t len, int proc, _cmx_request* req)
{
  return CMX_SUCCESS;
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
int p_Allocation::acc(int op, void *scale, void *src, void *dst,
    int64_t bytes, int proc)
{
  cmx_request request;
  int wrank;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  p_environment->translateWorld(1,p_group,&proc,&wrank);
  p_impl_environment->nb_register_request(&request);
  p_impl_environment->nb_acc(op,scale,src,dst,bytes,wrank,&request);
  p_impl_environment->nb_wait_for_all(&request);
  return CMX_SUCCESS;
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
int p_Allocation::accs(int op, void *scale, void *src, int64_t *src_stride,
    void *dst, int64_t *dst_stride, int64_t *count, int stride_levels,
    int proc)
{
  return CMX_SUCCESS;
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
int p_Allocation::accv(int op, void *scale, _cmx_giov_t *darr, int64_t len, int proc)
{
  return CMX_SUCCESS;
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
int p_Allocation::nbacc(int op, void *scale, void *src, void *dst,
    int64_t bytes, int proc, _cmx_request *req)
{
  int wrank;
  p_environment->translateWorld(1,p_group,&proc,&wrank);
  p_impl_environment->nb_register_request(req);
  p_impl_environment->nb_acc(op,scale,src,dst,bytes,wrank,req);
  return CMX_SUCCESS;
}

/**
 * Nonblocking Strided Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] src_stride [stride_levels] array of strides at source
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] dst_stride [stride_levels] array of strides at destination
 * @param[in] count number of units at each stride level count[0]=bytes
 * @param[in] stride_levels number of stride levels
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int p_Allocation::nbaccs(int op, void *scale, void *src, int64_t *src_stride,
    void *dst, int64_t *dst_stride, int64_t *count,
    int stride_levels, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}

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
int p_Allocation::nbaccv(int op, void *scale, _cmx_giov_t *darr, int64_t len,
    int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
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
int p_Allocation::get(void *src, void *dst, int64_t bytes, int proc)
{
  cmx_request request;
  int wrank;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  p_environment->translateWorld(1,p_group,&proc,&wrank);
  p_impl_environment->nb_register_request(&request);
  p_impl_environment->nb_get(src,dst,bytes,wrank,&request);
  p_impl_environment->nb_wait_for_all(&request);
  return CMX_SUCCESS;
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
int p_Allocation::gets(void *dst, int64_t *dst_stride, void *src,
    int64_t *src_stride, int64_t *count, int stride_levels, int proc)
{
  return CMX_SUCCESS;
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
int p_Allocation::getv(_cmx_giov_t *darr, int64_t len, int proc)
{
  return CMX_SUCCESS;
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
int p_Allocation::nbget(void *src, void *dst, int64_t bytes,
    int proc, _cmx_request *req)
{
  int wrank;
  p_environment->translateWorld(1,p_group,&proc,&wrank);
  p_impl_environment->nb_register_request(req);
  p_impl_environment->nb_get(src,dst,bytes,wrank,req);
  return CMX_SUCCESS;
}

/**
 * Nonblocking Strided Get.
 *
 * @param[in] src pointer to 1st segment at segment
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
int p_Allocation::nbgets( void *dst, int64_t *dst_stride, void *src,
    int64_t *src_stride, int64_t *count, int stride_levels, int proc,
    _cmx_request *req)
{
  return CMX_SUCCESS;
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
int p_Allocation::nbgetv(_cmx_giov_t *darr, int64_t len, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}

/**
 * Flush all outgoing messages from me to the given proc on the group of the
 * allocation.
 * @param[in] proc the proc with which to flush outgoing messages
 * @return CMX_SUCCESS on success
 */
int p_Allocation::fenceProc(int proc)
{
  return CMX_SUCCESS;
}

/**
 * Flush all outgoing messages to all procs in group of the allocation
 * @return CMX_SUCCESS on success
 */
int p_Allocation::fenceAll()
{
  p_environment->fence(p_group);
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
 * @param[in] rem_offset offset to remote value
 * @param[in] extra for CMX_FETCH_AND_ADD and CMX_FETCH_AND_ADD_LONG, the
 *            amount to increment the remote value by
 * @param[in] proc remote process(or) id
 * @param[in] cmx_hdl handle for data allocation
 * @return CMX_SUCCESS on success
 */
int p_Allocation::readModifyWrite(int op, void *ploc, int64_t rem_offset,
    int extra, int proc)
{
  return CMX_SUCCESS;
}

/**
 * Waits for completion of non-blocking cmx operations with explicit handles.
 * @param[in] req the request handle
 * @return CMX_SUCCESS on success
 */
int p_Allocation::wait(_cmx_request *req)
{
  p_impl_environment->nb_wait_for_all(req);
  return CMX_SUCCESS;
}

/**
 * Checks completion status of non-blocking cmx operations with explicit
 * handles.
 * @param[in] req the request handle
 * @param[out] status true->completed, false->in progress
 * @return CMX_SUCCESS on success
 */
int p_Allocation::test(_cmx_request *req, bool *status)
{
  return CMX_SUCCESS;
}

/**
 * Wait for all outstanding implicit non-blocking operations to finish on the
 * group of the allocation
 * @return CMX_SUCCESS on success
 */
int p_Allocation::waitAll()
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
int p_Allocation::waitProc(int proc)
{
  return CMX_SUCCESS;
}

}; // CMX namespace
