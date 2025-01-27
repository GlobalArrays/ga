#include <stdlib.h>
#include <mpi.h>

#include "environment.hpp"



namespace CMX {

Environment *Environment::p_instance = NULL;

/**
 * Initialize the environment
 */
Environment* Environment::instance()
{
  if (p_instance == NULL) {
    int flag;
    MPI_Initialized(&flag);
    if (!flag) {
      int argc;
      char **argv;
      MPI_Init(&argc, &argv);
    }
    p_instance = new Environment();
  }
  return p_instance;
}

/**
 * Initialize the environment with arguments
 * @param[in] argc number of arguments
 * @param[in] argv list of arguments
 */
Environment *Environment::instance(int *argc, char ***argv)
{
  if (p_instance == NULL) {
    int flag;
    MPI_Initialized(&flag);
    if (!flag) {
      MPI_Init(argc, argv);
    }
    p_instance = new Environment();
  }
  return p_instance;
}

/**
 * clean up environment and shut down libraries
 */
void Environment::finalize()
{
  p_Impl->finalize();
  delete p_cmx_world_group;
  p_cmx_world_group = NULL;
}

/**
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void Environment::wait(cmx_request *hdl)
{
  p_Impl->wait(hdl);
}

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void Environment::waitAll(Group *group)
{
  p_Impl->waitAll(group);
}

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool Environment::test(cmx_request *hdl)
{
  return p_Impl->test(hdl);
}

/**
 * Fence on all processes in group
 * @param group fence all process in group
 */
void Environment::fence(Group *group)
{
  p_Impl->fence(group);
}

/**
 * Get world group
 * @return pointer to world group
 */
Group* Environment::getWorldGroup()
{
  return p_cmx_world_group;
}

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
void Environment::error(char *msg, int code)
{
}

/**
 * Translates the ranks of processes in one group to those in another group.  The
 * group making the call is the "from" group, the group in the argument list is
 * the "to" group.
 *
 * @param[in] n the number of ranks in the ranks_from and ranks_to arrays
 * @param[in] group_from the group to translate ranks from 
 * @param[in] ranks_from array of zero or more valid ranks in group_from
 * @param[in] group_to the group to translate ranks to 
 * @param[out] ranks_to array of corresponding ranks in group_to
 * @return CMX_SUCCESS on success
 */
int Environment::translateRanks(int n, Group *group_from,
    int *ranks_from, Group *group_to, int *ranks_to)
{
  return p_Impl->translateRanks(n,group_from,ranks_from,group_to,ranks_to);
}

/**
 * Translate the given rank from its group to its corresponding rank in the
 * world group. Convenience function for common case.
 *
 * @param[in] n the number of ranks in the group_ranks and world_ranks arrays
 * @param[in] group the group to translate ranks from 
 * @param[in] group_ranks the ranks to translate from
 * @param[out] world_ranks the corresponding world rank
 * @return CMX_SUCCESS on success
 */
int Environment::translateWorld(int n, Group *group, int *group_ranks,
    int *world_ranks)
{
  return p_Impl->translateWorld(n,group,group_ranks,world_ranks);
}

#if 0
/**
 * This function does most of the setup and memory allocation of distributed
 * memory segments.
 * @param ptrs list of pointers to all allocations in group
 * @param bytes number of bytes being allocated on calling processor
 * @param group group of processors over which allocation is performed. If no
 * group is specified, assume allocation is on world group
 * @return SUCCESS if allocation is successful
 */
int Environment::dist_malloc(void** ptrs, int64_t bytes, Group *group)
{
  return p_Impl->dist_malloc(ptrs,bytes,group);
}

/**
 * Free allocation across all processors in group
 * @param ptr pointer to allocation
 * @param group group holding distributed allocation
 */
void Environment::free(void *ptr, Group *group)
{
  p_Impl->dist_free(ptr, group);
}

/**
 * Contiguous non-blocking put from local buffer to remote process
 * @param[in] src pointer to local buffer
 * @param[in] dst pointer to remote buffer
 * @param[in] bytes size of data transfer, in bytes
 * @param[in] proc rank of remote process
 * @param[in] nb non-blocking request handle for transfer
 */
void Environment::nb_put(void *src, void *dst, int bytes, int proc, request_t *nb)
{
  p_Impl->nb_put(src,dst,bytes,proc,nb);
}

/**
 * Contiguous non-blocking get from remote process to local buffer
 * @param[in] src pointer to remote buffer
 * @param[in] dst pointer to local buffer
 * @param[in] bytes size of data transfer, in bytes
 * @param[in] proc rank of remote process
 * @param[in] nb non-blocking request handle for transfer
 */
void Environment::nb_get(void *src, void *dst, int bytes, int proc, request_t *nb)
{
  p_Impl->nb_get(src,dst,bytes,proc,nb);
}

/**
 * Contiguous non-blocking accumulate from local buffer to remote process
 * @param[in] datatype enumerated type for data
 * @param[in] scale multiply data in source by scale before adding to destination
 * @param[in] src pointer to local buffer
 * @param[in] dst pointer to remote buffer
 * @param[in] bytes size of data transfer, in bytes
 * @param[in] proc rank of remote process
 * @param[in] nb non-blocking request handle for transfer
 */
void Environment::nb_acc(int datatype, void* scale, void *src, void *dst,
    int bytes, int proc, request_t *nb)
{
  p_Impl->nb_acc(datatype,scale,src,dst,bytes,proc,nb);
}

/**
 * Strided non-blocking put from local buffer to remote process
 * @param[in] src pointer to local buffer
 * @param[in] src_stride array of strides for source buffer
 * @param[in] dst pointer to remote buffer
 * @param[in] dst_stride array of strides for remote buffer
 * @param[in] count array containing length of stride along each dimension.
 * @param[in] stride_levels number of stride levels
 * @param[in] proc rank of remote process
 * @param[in] nb non-blocking request handle for transfer
 */
void Environment::nb_puts(void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, request_t *nb)
{
  p_Impl->nb_puts(src,src_stride,dst,dst_stride,count,stride_levels,proc,nb);
}

/**
 * Strided non-blocking get from remote process to local buffer
 * @param[in] src pointer to remote buffer
 * @param[in] src_stride array of strides for source buffer
 * @param[in] dst pointer to local buffer
 * @param[in] dst_stride array of strides for remote buffer
 * @param[in] count array containing length of stride along each dimension.
 * @param[in] stride_levels number of stride levels
 * @param[in] proc rank of remote process
 * @param[in] nb non-blocking request handle for transfer
 */
void Environment::nb_gets(void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, request_t *nb)
{
  p_Impl->nb_gets(src,src_stride,dst,dst_stride,count,stride_levels,proc,nb);
}

/**
 * Strided non-blocking accumulate from local buffer to remote process
 * @param[in] datatype enumerated type for data
 * @param[in] scale multiply data in source by scale before adding to destination
 * @param[in] src pointer to local buffer
 * @param[in] src_stride array of strides for source buffer
 * @param[in] dst pointer to remote buffer
 * @param[in] dst_stride array of strides for remote buffer
 * @param[in] count array containing length of stride along each dimension.
 * @param[in] stride_levels number of stride levels
 * @param[in] proc rank of remote process
 * @param[in] nb non-blocking request handle for transfer
 */
void Environment::nb_accs(int datatype, void* scale, void *src,
    int *src_stride, void *dst, int *dst_stride, int *count, int stride_levels,
    int proc, request_t *nb)
{
  p_Impl->nb_accs(datatype,scale,src,src_stride,dst,dst_stride,count,
      stride_levels,proc,nb);
}

/**
 * Vector non-blocking put of random elements from local buffer to random
 * locations in remote process
 * @param[in] iov data structure describing locations of elements in source
 *            and destination buffers
 * @param[in] iov_len number of elements to be transfered
 * @param[in] nb non-blocking request handle for transfer
 */
void Environment::nb_putv(cmx_giov_t *iov, int iov_len, int proc, request_t *nb)
{
  p_Impl->nb_putv(iov,iov_len,proc,nb);
}

/**
 * Vector non-blocking get of random elements from remote process to random
 * locations in local buffer
 * @param[in] iov data structure describing locations of elements in source
 *            and destination buffers
 * @param[in] iov_len number of elements to be transfered
 * @param[in] nb non-blocking request handle for transfer
 */
void Environment::nb_getv(cmx_giov_t *iov, int iov_len, int proc, request_t *nb)
{
  p_Impl->nb_getv(iov,iov_len,proc,nb);
}

/**
 * Vector non-blocking accumulate of random elements from local buffer to random
 * locations in remote process
 * @param[in] datatype enumerated type for data
 * @param[in] scale multiply data in source by scale before adding to destination
 * @param[in] iov data structure describing locations of elements in source
 *            and destination buffers
 * @param[in] iov_len number of elements to be transfered
 * @param[in] nb non-blocking request handle for transfer
 */
void Environment::nb_accv(int datatype, void* scale, cmx_giov_t *iov, int iov_len, int proc, request_t *nb)
{
  p_Impl->nb_accv(datatype,scale,iov,iov_len,proc,nb);
}
#endif

/**
 * Initialize CMX environment.
 */
Environment::Environment()
{
  p_Impl = p_Environment::instance();
  p_cmx_world_group = p_Impl->getWorldGroup();
}

/**
 * Terminate CMX environment and clean up resources.
 */
Environment::~Environment()
{
  if (p_instance) delete p_instance;
}

};
