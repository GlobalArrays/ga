/* cmx header file */
#ifndef _CMX_COMMON_ENVIRONMENT_H
#define _CMX_COMMON_ENVIRONMENT_H

#include <mpi.h>

#include <stdlib.h>

#include "defines.hpp"
#include "group.hpp"
#include "p_environment.hpp"

namespace CMX {

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
static void CMXError(char *msg, int code)
{
  printf("Environment Abort: %s code: %d\n",msg,code);
  MPI_Abort(MPI_COMM_WORLD,code);
}

class p_Environment;
class Group;

class Environment {

public:

/**
 * Return an instance of the p_Environment singleton
 * @return pointer to p_Environment singleton
 */
static Environment *instance(); 

/**
 * Return an instance of the p_Environment singleton. Initialize instance
 * with argc and argv if it does not already exist
 * @param[in] argc number of arguments
 * @param[in] argv list of arguments
 * @return pointer to p_Environment singleton
 */
static Environment *instance(int *argc, char ***argv); 

/**
 * wait for completion of non-blocking handle
 * @param hdl non-blocking request handle
 */
void wait(cmx_request *hdl);

/**
 * wait for completion of non-blocking handles associated with a particular group
 * @param group
 */
void waitAll(CMX::Group *group);

/**
 * clean up environment and shut down libraries
 */
void finalize();

/**
 * test for completion of non-blocking handle. If test is true, operation has
 * completed locally
 * @param hdl non-blocking request handle
 * @return true if operation is completed locally
 */
bool test(cmx_request *hdl);

/**
 * Translates the ranks of processes in one group to those in another group. The
 * group making the call is the "from" group, the group in the argument list is
 * the "to" group.
 *
 * @param[in] n the number of ranks in the ranks_from and ranks_to arrays
 * @param[in] ranks_from array of zero or more valid ranks in group_from
 * @param[in] group_to the group to translate ranks to 
 * @param[out] ranks_to array of corresponding ranks in group_to
 * @return CMX_SUCCESS on success
 */
int translateRanks(int n, int *ranks_from, Group *group_to, int *ranks_to);

/**
 * Translate the given rank from its group to its corresponding rank in the
 * world group. Convenience function for common case.
 *
 * @param[in] n the number of ranks in the group_ranks and world_ranks arrays
 * @param[in] group_ranks the rank to translate from
 * @param[out] world_ranks the corresponding world rank
 * @return CMX_SUCCESS on success
 */
int translateWorld(int n, int *group_ranks, int *world_ranks);

/**
 * Get world group
 * @return pointer to world group
 */
Group* getWorldGroup();

/**
 * Abort CMX, printing the msg, and exiting with code.
 * @param[in] msg the message to print
 * @param[in] code the code to exit with
 */
void error(char *msg, int code);

#if 0
/*n
 e* This function does most of the setup and memory allocation of distributed
 * memory segments.
 * @param ptrs list of pointers to all allocations in group
 * @param bytes number of bytes being allocated on calling processor
 * @param group group of processors over which allocation is performed. If no
 * group is specified, assume allocation is on world group
 * @return SUCCESS if allocation is successful
 */
int dist_malloc(void** ptrs, int64_t bytes, Group *group);

/**
 * Free allocation across all processors in group
 * @param ptr pointer to allocation
 * @param group group holding distributed allocation
 */
void free(void *ptr, Group *group);

/**
 * Contiguous non-blocking put from local buffer to remote process
 * @param[in] src pointer to local buffer
 * @param[in] dst pointer to remote buffer
 * @param[in] bytes size of data transfer, in bytes
 * @param[in] proc rank of remote process
 * @param[in] nb non-blocking request handle for transfer
 */
void nb_put(void *src, void *dst, int bytes, int proc, request_t *nb);

/**
 * Contiguous non-blocking get from remote process to local buffer
 * @param[in] src pointer to remote buffer
 * @param[in] dst pointer to local buffer
 * @param[in] bytes size of data transfer, in bytes
 * @param[in] proc rank of remote process
 * @param[in] nb non-blocking request handle for transfer
 */
void nb_get(void *src, void *dst, int bytes, int proc, request_t *nb);

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
void nb_acc(int datatype, void* scale, void *src, void *dst, int bytes,
    int proc, request_t *nb);

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
void nb_puts(void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, request_t *nb);

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
void nb_gets(void *src, int *src_stride, void *dst, int *dst_stride,
    int *count, int stride_levels, int proc, request_t *nb);

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
void nb_accs(int datatype, void * scale, void *src, int *src_stride,
    void *dst, int *dst_stride, int *count, int stride_levels, int proc,
    request_t *nb);

/**
 * Vector non-blocking put of random elements from local buffer to random
 * locations in remote process
 * @param[in] iov data structure describing locations of elements in source
 *            and destination buffers
 * @param[in] iov_len number of elements to be transfered
 * @param[in] nb non-blocking request handle for transfer
 */
void nb_putv(cmx_giov_t *iov, int iov_len, int proc, request_t *nb);

/**
 * Vector non-blocking get of random elements from remote process to random
 * locations in local buffer
 * @param[in] iov data structure describing locations of elements in source
 *            and destination buffers
 * @param[in] iov_len number of elements to be transfered
 * @param[in] nb non-blocking request handle for transfer
 */
void nb_getv(cmx_giov_t *iov, int iov_len, int proc, request_t *nb);

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
void nb_accv(int datatype, void* scale, cmx_giov_t *iov, int iov_len, int proc,
    request_t *nb);
#endif

protected:

/**
 * Initialize CMX environment.
 */
Environment();

/**
 * Terminate CMX environment and clean up resources.
 */
virtual ~Environment();

private:

p_Environment *p_Impl;

static Environment *p_instance;

Group *p_cmx_world_group;

//friend class p_Environment;

};
}

#endif /* _CMX_H */
