
#include "p_alloc.hpp"

namespace CMX {

/**
 * Constructor
 * @param bytes number of bytes being allocated on calling processor
 * @param group group of processors over which allocation is performed. If no
 * group is specified, assume allocation is on world group
 */
p_Allocation::p_Allocation(int bytes, p_Group *group=NULL)
{
}

p_Allocation::p_Allocation(int64_t bytes, p_Group *group=NULL)
{
  reg_entry_t *reg_entries = NULL;
  reg_entry_t my_reg;
  size_t size_entries = 0;
  int my_master = -1;
  int my_world_rank = -1;
  int i = 0;
  int is_notifier = 0;
  int reg_entries_local_count = 0;
  reg_entry_t *reg_entries_local = NULL;
  int status = 0;

  /* preconditions */
  CMX_ASSERT(group);

  /* is this needed? */
  group->barrier();

  p_environment = p_Environment::intance();

  my_world_rank = group->getWorldRank();
  p_state = p_environment->getGlobalState();
  my_master = p_state->master[my_world_rank];

  int smallest_rank_with_same_hostid, largest_rank_with_same_hostid; 
  int num_progress_ranks_per_node, is_node_ranks_packed;
  num_progress_ranks_per_node = get_num_progress_ranks_per_node();
  is_node_ranks_packed = get_progress_rank_distribution_on_node();
  smallest_rank_with_same_hostid = _smallest_world_rank_with_same_hostid(group);
  largest_rank_with_same_hostid = _largest_world_rank_with_same_hostid(group);
  is_notifier = p_state->rank == get_my_master_rank_with_same_hostid(p_state->rank,
      p_state->node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
      num_progress_ranks_per_node, is_node_ranks_packed);
  if (is_notifier) {
    reg_entries_local = malloc(sizeof(reg_entry_t)*p_state->node_size);
  }

  /* allocate space for registration cache entries */
  size_entries = sizeof(reg_entry_t) * group->size();
  reg_entries = malloc(size_entries);
  MAYBE_MEMSET(reg_entries, 0, sizeof(reg_entry_t)*group->size());

  /* allocate and register segment */
  MAYBE_MEMSET(&my_reg, 0, sizeof(reg_entry_t));
  if (0 == size) {
    reg_cache_nullify(&my_reg);
  }
  else {
    my_reg = *_cmx_malloc_local(sizeof(char)*size);
  }

  /* exchange buffer address via reg entries */
  reg_entries[group->rank()] = my_reg;
  status = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
      reg_entries, sizeof(reg_entry_t), MPI_BYTE, group->MPIComm());
  _translate_mpi_error(status, "cmx_malloc:MPI_Allgather");
  CMX_ASSERT(MPI_SUCCESS == status);

  /* insert reg entries into local registration cache */
  for (i=0; i<group->size(); ++i) {
    if (NULL == reg_entries[i].buf) {
      /* a proc did not allocate (size==0) */
    } else if (p_state->rank == reg_entries[i].rank) {
      /* we already registered our own memory, but PR hasn't */
      if (is_notifier) {
        /* does this need to be a memcpy?? */
        reg_entries_local[reg_entries_local_count++] = reg_entries[i];
      }
    } else if (p_state->master[reg_entries[i].rank] == 
        p_state->master[get_my_master_rank_with_same_hostid(p_state->rank,
          p_state->node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
          num_progress_ranks_per_node, is_node_ranks_packed)] ) {
      /* same SMP node, need to mmap */
      /* open remote shared memory object */
      void *memory = _shm_attach(reg_entries[i].name, reg_entries[i].len);
      (void)reg_cache_insert(
          reg_entries[i].rank,
          reg_entries[i].buf,
          reg_entries[i].len,
          reg_entries[i].name,
          memory,0);
      if (is_notifier) {
        /* does this need to be a memcpy?? */
        reg_entries_local[reg_entries_local_count++] = reg_entries[i];
      }
    }
  }

  /* assign the cmx handle to return to caller */
  cmx_alloc_t *prev = NULL;
  for (i=0; i<group->size(); ++i) {
    cmx_alloc_t *link = (cmx_alloc_t*)malloc(sizeof(cmx_alloc_t));
    p_list = link;
    link->buf = reg_entries[i].buf;
    link->size = (int64_t)reg_entries[i].len;
    link->rank = reg_entries[i].rank;
    link->next = prev;
    prev = link;
  }
  p_group = group;
  p_rank = group->rank();
  p_buf = my_reg.mapped;
  p_bytes = my_reg.len;

  /* send reg entries to my master */
  /* first non-master rank in an SMP node sends the message to master */
  if (is_notifier) {
    _cmx_request nb;
    int reg_entries_local_size = 0;
    int message_size = 0;
    char *message = NULL;
    header_t *header = NULL;

    nb_handle_init(&nb);
    reg_entries_local_size = sizeof(reg_entry_t)*reg_entries_local_count;
    message_size = sizeof(header_t) + reg_entries_local_size;
    message = malloc(message_size);
    CMX_ASSERT(message);
    header = (header_t*)message;
    header->operation = OP_MALLOC;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = 0;
    header->length = reg_entries_local_count;
    (void)memcpy(message+sizeof(header_t), reg_entries_local, reg_entries_local_size);
    nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
    nb_send_header(message, message_size, my_master, &nb);
    nb_wait_for_all(&nb);
    free(reg_entries_local);
  }

  free(reg_entries);

  group->barrier();

  return CMX_SUCCESS;
}

/**
 * Destructor
 */
p_Allocation::~p_Allocation();

/**
 * Access a list of pointers to data on remote processors
 * Note: this returns a standard vector, which may be slow. If needed,
 * access the internal vector of data using
 *   void **myptr = &ptrs[0];
 * @param ptrs a vector of pointers to data on all processors in the allocation
 * @return CMX_SUCCESS on success
 */
int p_Allocation::access(std::vector<void*> &ptrs);

/**
 * Access internal group
 * @return pointer to group 
 */
p_Group* p_Allocation::group();

/**
 * A collective communication and operations barrier on the internal group.
 * This is equivalent to calling group()->barrier()
 * @return CMX_SUCCESS on success
 */
int p_Allocation::barrier();

/**
 * Contiguous Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int p_Allocation::put(void *src, int dst_offset, int bytes, int proc);
int p_Allocation::put(void *src, int64_t dst_offset, int64_t bytes, int proc);

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
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int puts(void *src, int *src_stride, int dst_offset, int *dst_stride,
        int *count, int stride_levels, int proc);
int puts(void *src, int64_t *src_stride, int64_t dst_offset, int64_t *dst_stride,
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
int putv(cmx_giov_t *darr, int len, int proc);
int putv(cmx_giov_t *darr, int64_t len, int proc);

/**
 * Nonblocking Contiguous Put.
 *
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbput(void *src, int dst_offset, int bytes, int proc, cmx_request_t* req);
int nbput(void *src, int64_t dst_offset, int64_t bytes, int proc, cmx_request_t* req);

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
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbputs(void *src, int *src_stride, int dst_offset, int *dst_stride,
        int *count, int stride_levels, int proc, cmx_request_t* req);
int nbputs(void *src, int64_t *src_stride, int64_t dst_offset, int64_t *dst_stride,
        int64_t *count, int stride_levels, int proc, cmx_request_t* req);

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
int nbputv(cmx_giov_t *darr, int len, int proc, cmx_request_t* req);
int nbputv(cmx_giov_t *darr, int64_t len, int proc, cmx_request_t* req);

/**
 * Contiguous Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int acc(int op, void *scale, void *src, int dst_offset, int bytes, int proc);
int acc(int op, void *scale, void *src, int64_t dst_offset, int64_t bytes, int proc);

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
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int accs(int op, void *scale, void *src, int *src_stride, int dst_offset,
    int *dst_stride, int *count, int stride_levels, int proc);
int accs(int op, void *scale, void *src, int64_t *src_stride, int64_t dst_offset,
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
int accv(int op, void *scale, cmx_giov_t *darr, int len, int proc);
int accv(int op, void *scale, cmx_giov_t *darr, int64_t len, int proc);

/**
 * Nonblocking Contiguous Atomic Accumulate.
 *
 * @param[in] op operation
 * @param[in] scale factor x += scale*y
 * @param[in] src pointer to 1st segment at source
 * @param[in] dst_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbacc(int op, void *scale, void *src, int dst_offset,
    int bytes, int proc, cmx_request_t *req);
int nbacc(int op, void *scale, void *src, int64_t dst_offset,
    int64_t bytes, int proc, cmx_request_t *req);

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
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbaccs(int op, void *scale, void *src, int *src_stride,
    int dst_offset, int *dst_stride, int *count,
    int stride_levels, int proc, cmx_request_t *req);
int nbaccs(int op, void *scale, void *src, int64_t *src_stride,
    int64_t dst_offset, int64_t *dst_stride, int64_t *count,
    int stride_levels, int proc, cmx_request_t *req);

/**
 * Vector Atomic Accumulate.
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
int nbaccv(int op, void *scale, cmx_giov_t *darr, int len, int proc, cmx_request_t *req);
int nbaccv(int op, void *scale, cmx_giov_t *darr, int64_t len, int proc, cmx_request_t *req);

/**
 * Contiguous Get.
 *
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] src_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int get(void *dst, int src_offset, int bytes, int proc);
int get(void *dst, int64_t src_offset, int64_t bytes, int proc);

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
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @return CMX_SUCCESS on success
 */
int gets(void *dst, int *dst_stride, int src_offset, int *src_stride,
    int *count, int stride_levels, int proc);
int gets(void *dst, int64_t *dst_stride, int64_t src_offset, int64_t *src_stride,
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
int getv(cmx_giov_t *darr, int len, int proc);
int getv(cmx_giov_t *darr, int64_t len, int proc);

/**
 * Nonblocking Contiguous Get.
 *
 * @param[in] dst pointer to 1st segment at destination
 * @param[in] src_offset offset from start of data allocation on remote
 *            process
 * @param[in] bytes number of bytes to transfer
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbget(void *dst, int src_offset, int bytes, int proc, cmx_request_t *req);
int nbget(void *dst, int64_t src_offset, int64_t bytes, int proc, cmx_request_t *req);

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
 * @param[in] proc remote process(or) id. This processor must be in the same
 *            group as the allocation.
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbgets(void *dst, int *dst_stride, int src_offset, int *src_stride,
    int *count, int stride_levels, int proc, cmx_request_t *req);
int nbgets(void *dst, int64_t *dst_stride, int64_t src_offset, int64_t *src_stride,
    int64_t *count, int stride_levels, int proc, cmx_request_t *req);

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
int nbgetv(cmx_giov_t *darr, int len, int proc, cmx_request_t *req);
int nbgetv(cmx_giov_t *darr, int64_t len, int proc, cmx_request_t *req);

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
int readModifyWrite(int op, void *ploc, int rem_offset, int extra, int proc);
int readModifyWrite(int op, void *ploc, int64_t rem_offset, int extra, int proc);

/**
 * Waits for completion of non-blocking cmx operations with explicit handles.
 * @param[in] req the request handle
 * @return CMX_SUCCESS on success
 */
int wait(cmx_request_t *req);

/**
 * Checks completion status of non-blocking cmx operations with explicit
 * handles.
 * @param[in] req the request handle
 * @param[out] status true->completed, false->in progress
 * @return CMX_SUCCESS on success
 */
int test(cmx_request_t *req, bool *status);

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

#if 0
/* This should probably be a separate class (if implemented at all) */
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
#endif
}
