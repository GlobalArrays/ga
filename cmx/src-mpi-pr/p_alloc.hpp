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

#if USE_MEMSET_AFTER_MALLOC
#define MAYBE_MEMSET(a,b,c) (void)memset(a,b,c)
#else
#define MAYBE_MEMSET(a,b,c) ((void)0)
#endif

#define CMX_ASSERT(WHAT) ((void)(0))

namespace CMX {

template <typename _data_type> class p_Allocation {
public:

/**
 * Constructor
 * @param bytes number of bytes being allocated on calling processor
 * @param group group of processors over which allocation is performed.
 */
p_Allocation(int bytes, Group *group)
{
}

/**
 * Return a reg_entry_t struct segment of locally allocated memory
 * @param size length of allocation in bytes
 * @return reg_entry_t struct containing information on allocation
 */
reg_entry_t* _malloc_local(size_t size)
{
  char *name = NULL;
  void *memory = NULL;
  reg_entry_t *reg_entry = NULL;

  if (0 == size) {
    return NULL;
  }

  /* create my shared memory object */
  CMX::p_Shmem *shm = CMX::p_Shmem::instance();
  name = shm->generate_name(p_environment->getWorldGroup()->rank());
  memory = shm->create(name, size);
  /* register the memory locally */
  reg_entry = reg_cache_insert(
      p_state->rank, memory, size, name, memory, 0);

  if (NULL == reg_entry) {
    p_environment->error("_malloc_local: reg_cache_insert", -1);
  }

  free(name);

  return reg_entry;
}

void* malloc_local(size_t size)
{
    reg_entry_t *reg_entry;
    void *memory = NULL;

    if (size > 0) {
        reg_entry = _malloc_local(size);
        memory = reg_entry->mapped;
    }
    else {
        memory = NULL;
    }

    return memory;
}

/**
 * return the pointer to memory allocated on proc
 * @param proc rank of processor
 * @return pointer to allocated memory
 */
void* find_alloc_ptr(int proc)
{
  void *ret = NULL;
  cmx_alloc_t *next = p_list;
  while (next != NULL) {
    if (next->rank == proc) {
      ret = next->buf;
      break;
    }
    next = next->next;
  }
  return ret;
}



/**
 * This function does most of the setup and memory allocation. It is used to
 * simplify the writing of the actual constructor, which handles the data type.
 * @param bytes number of bytes being allocated on calling processor
 * @param group group of processors over which allocation is performed. If no
 * group is specified, assume allocation is on world group
 * @return SUCCESS if allocation is successful
 */
int malloc(int64_t bytes, Group *group=NULL)
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

  p_group = group;

  /* is this needed? */
  p_group->barrier();

  p_environment = Environment::instance();

  my_world_rank = p_group->getWorldRank(p_group->rank());
  my_master = p_state->master[my_world_rank];

  is_notifier = p_state->rank == p_node_config->get_my_master_rank_with_same_hostid(
      p_state->rank, p_state->node_size, group);
  if (is_notifier) {
    reg_entries_local = malloc(sizeof(reg_entry_t)*p_state->node_size);
  }

  /* allocate space for registration cache entries */
  size_entries = sizeof(reg_entry_t) * p_group->size();
  reg_entries = malloc(size_entries);
  MAYBE_MEMSET(reg_entries, 0, sizeof(reg_entry_t)*p_group->size());

  /* allocate and register segment */
  MAYBE_MEMSET(&my_reg, 0, sizeof(reg_entry_t));
  if (0 == bytes) {
    reg_cache_nullify(&my_reg);
  }
  else {
    my_reg = *_malloc_local(sizeof(char)*bytes);
  }

  CMX::p_Shmem *shm = CMX::p_Shmem::instance();

  /* exchange buffer address via reg entries */
  reg_entries[p_group->rank()] = my_reg;

  status = MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
      reg_entries, sizeof(reg_entry_t), MPI_BYTE, p_group->MPIComm());
  _translate_mpi_error(status, "cmx_malloc:MPI_Allgather");
//  CMX_ASSERT(MPI_SUCCESS == status);

  /* insert reg entries into local registration cache */
  for (i=0; i<p_group->size(); ++i) {
    if (NULL == reg_entries[i].buf) {
      /* a proc did not allocate (bytes==0) */
    } else if (p_state->rank == reg_entries[i].rank) {
      /* we already registered our own memory, but PR hasn't */
      if (is_notifier) {
        /* does this need to be a memcpy?? */
        reg_entries_local[reg_entries_local_count++] = reg_entries[i];
      }
    } else if (p_state->master[reg_entries[i].rank] == 
        p_state->master[my_master]) {
      /* same SMP node, need to mmap */
      /* open remote shared memory object */
      void *memory = shm->attach(reg_entries[i].name, reg_entries[i].len);
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
  for (i=0; i<p_group->size(); ++i) {
    cmx_alloc_t *link = (cmx_alloc_t*)malloc(sizeof(cmx_alloc_t));
    p_list = link;
    link->buf = reg_entries[i].buf;
    link->size = (int64_t)reg_entries[i].len;
    link->rank = reg_entries[i].rank;
    link->next = prev;
    prev = link;
  }
  p_rank = p_group->rank();
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

    p_environment->nb_handle_init(&nb);
    reg_entries_local_size = sizeof(reg_entry_t)*reg_entries_local_count;
    message_size = sizeof(header_t) + reg_entries_local_size;
    message = malloc(message_size);
//    CMX_ASSERT(message);
    header = (header_t*)message;
    header->operation = OP_MALLOC;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = 0;
    header->length = reg_entries_local_count;
    (void)memcpy(message+sizeof(header_t), reg_entries_local, reg_entries_local_size);
    p_environment->nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
    p_environment->nb_send_header(message, message_size, my_master, &nb);
    p_environment->nb_wait_for_all(&nb);
    free(reg_entries_local);
  }

  free(reg_entries);

  p_group->barrier();

  return CMX_SUCCESS;
}

/**
 * Destructor
 */
~p_Allocation()
{
  int my_world_rank = -1;
  int *world_ranks = NULL;
  int my_master = -1;
  void **ptrs = NULL;
  void *ptr;
  int i = 0;
  int is_notifier = 0;
  int reg_entries_local_count = 0;
  rank_ptr_t *rank_ptrs = NULL;
  int status = 0;
  cmx_alloc_t *list, *next;


#if DEBUG
  fprintf(stderr, "[%d] cmx_free(ptr=%p, group=%d)\n", p_state->rank, ptr, group);
#endif

  p_group->barrier();

  my_world_rank = p_group->getWorldRank(p_group->rank());
  my_master = p_state->master[my_world_rank];

#if DEBUG && DEBUG_VERBOSE
  fprintf(stderr, "[%d] cmx_free my_master=%d\n", p_state->rank, my_master);
#endif

  int num_progress_ranks_per_node = p_environment->get_num_progress_ranks_per_node();
  int is_node_ranks_packed = p_environment->get_progress_rank_distribution_on_node();
  int smallest_rank_with_same_hostid = p_environment->smallest_world_rank_with_same_hostid(p_group);
  int largest_rank_with_same_hostid = p_environment->largest_world_rank_with_same_hostid(p_group);
  is_notifier = p_state->rank == p_environment->get_my_master_rank_with_same_hostid(p_state->rank,
      p_state->node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
      num_progress_ranks_per_node, is_node_ranks_packed);
  if (is_notifier) {
    rank_ptrs = malloc(sizeof(rank_ptr_t)*p_state->node_size);
  }

  /* allocate receive buffer for exchange of pointers */
  ptrs = (void **)malloc(sizeof(void *) * p_group->size());
//  CMX_ASSERT(ptrs);
  ptrs[p_group->rank()] = find_alloc_ptr(my_world_rank);
  ptr = ptrs[p_group->rank()];

#if DEBUG && DEBUG_VERBOSE
  fprintf(stderr, "[%d] cmx_free ptrs allocated and assigned\n",
      p_state->rank);
#endif

  /* exchange of pointers */
  status = MPI_Allgather(MPI_IN_PLACE, sizeof(void *), MPI_BYTE,
      ptrs, sizeof(void *), MPI_BYTE, p_group->MPIComm());
  _translate_mpi_error(status, "cmx_free:MPI_Allgather");
//  CMX_ASSERT(MPI_SUCCESS == status);

#if DEBUG && DEBUG_VERBOSE
  fprintf(stderr, "[%d] cmx_free ptrs exchanged\n", p_state->rank);
#endif

  /* remove all pointers from registration cache */
  for (i=0; i<p_group->size(); ++i) {
    if (i == p_group->rank()) {
#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] cmx_free found self at %d\n", p_state->rank, i);
#endif
      if (is_notifier) {
        /* does this need to be a memcpy? */
        rank_ptrs[reg_entries_local_count].rank = world_ranks[i];
        rank_ptrs[reg_entries_local_count].ptr = ptrs[i];
        reg_entries_local_count++;
      }
    }
    else if (NULL == ptrs[i]) {
#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] cmx_free found NULL at %d\n", p_state->rank, i);
#endif
    }
    // else if (p_state->hostid[world_ranks[i]]
    //         == p_state->hostid[p_state->rank]) 
    else if (p_state->master[world_ranks[i]] == 
        p_state->master[p_environment->get_my_master_rank_with_same_hostid(p_state->rank,
          p_state->node_size, smallest_rank_with_same_hostid, largest_rank_with_same_hostid,
          num_progress_ranks_per_node, is_node_ranks_packed)] )
    {
      /* same SMP node */
      reg_entry_t *reg_entry = NULL;
      int retval = 0;

#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] cmx_free same hostid at %d\n", p_state->rank, i);
#endif

      /* find the registered memory */
      reg_entry = reg_cache_find(world_ranks[i], ptrs[i], 0);
//      CMX_ASSERT(reg_entry);

#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] cmx_free found reg entry\n", p_state->rank);
#endif

      /* unmap the memory */
      retval = munmap(reg_entry->mapped, reg_entry->len);
      if (-1 == retval) {
        perror("cmx_free: munmap");
        p_environment->error("cmx_free: munmap", retval);
      }

#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] cmx_free unmapped mapped memory in reg entry\n",
          p_state->rank);
#endif

      reg_cache_delete(world_ranks[i], ptrs[i]);

#if DEBUG && DEBUG_VERBOSE
      fprintf(stderr, "[%d] cmx_free deleted reg cache entry\n", p_state->rank);
#endif

      if (is_notifier) {
        /* does this need to be a memcpy? */
        rank_ptrs[reg_entries_local_count].rank = world_ranks[i];
        rank_ptrs[reg_entries_local_count].ptr = ptrs[i];
        reg_entries_local_count++;
      }
    } else {
    }
  }

  /* send ptrs to my master */
  /* first non-master rank in an SMP node sends the message to master */
  if (is_notifier) {
    _cmx_request nb;
    int rank_ptrs_local_size = 0;
    int message_size = 0;
    char *message = NULL;
    header_t *header = NULL;

    p_environment->nb_handle_init(&nb);
    rank_ptrs_local_size = sizeof(rank_ptr_t) * reg_entries_local_count;
    message_size = sizeof(header_t) + rank_ptrs_local_size;
    message = malloc(message_size);
//    CMX_ASSERT(message);
    header = (header_t*)message;
    header->operation = OP_FREE;
    header->remote_address = NULL;
    header->local_address = NULL;
    header->rank = 0;
    header->length = reg_entries_local_count;
    (void)memcpy(message+sizeof(header_t), rank_ptrs, rank_ptrs_local_size);
    p_environment->nb_recv(NULL, 0, my_master, &nb); /* prepost ack */
    p_environment->nb_send_header(message, message_size, my_master, &nb);
    p_environment->nb_wait_for_all(&nb);
    free(rank_ptrs);
  }

  /* free ptrs array */
  free(ptrs);
  free(world_ranks);

  /* remove my ptr from reg cache and free ptr */
  p_environment->free_local(ptr);

  /* Is this needed? */
  p_group->barrier();

  /* clean up the cmx_handle_t struct */
  list = p_list; 
  while (list) {
    next = list;
    list = next->next;
    free(next);
  }
}

/**
 * Access a list of pointers to data on remote processors
 * Note: this returns a standard vector, which may be slow. If needed,
 * access the internal vector of data using
 *   void **myptr = &ptrs[0];
 * @param ptrs a vector of pointers to data on all processors in the allocation
 * @return CMX_SUCCESS on success
 */
int access(std::vector<void*> &ptrs)
{
  return CMX_SUCCESS;
}

/**
 * Access internal group
 * @return pointer to group 
 */
Group* group()
{
  return (Group*)0;
}

/**
 * A collective communication and operations barrier on the internal group.
 * This is equivalent to calling group()->barrier()
 * @return CMX_SUCCESS on success
 */
int barrier()
{
  return CMX_SUCCESS;
}

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
int put(void *src, int dst_offset, int bytes, int proc)
{
  return CMX_SUCCESS;
}
int put(void *src, int64_t dst_offset, int64_t bytes, int proc)
{
  return CMX_SUCCESS;
}

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
        int *count, int stride_levels, int proc)
{
  return CMX_SUCCESS;
}
int puts(void *src, int64_t *src_stride, int64_t dst_offset, int64_t *dst_stride,
        int64_t *count, int stride_levels, int proc)
{
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
int putv(giov_t *darr, int len, int proc)
{
  return CMX_SUCCESS;
}
int putv(giov_t *darr, int64_t len, int proc)
{
  return CMX_SUCCESS;
}

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
int nbput(void *src, int dst_offset, int bytes, int proc, _cmx_request* req)
{
  return CMX_SUCCESS;
}
int nbput(void *src, int64_t dst_offset, int64_t bytes, int proc, _cmx_request* req)
{
  return CMX_SUCCESS;
}

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
        int *count, int stride_levels, int proc, _cmx_request* req)
{
  return CMX_SUCCESS;
}
int nbputs(void *src, int64_t *src_stride, int64_t dst_offset, int64_t *dst_stride,
        int64_t *count, int stride_levels, int proc, _cmx_request* req)
{
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
int nbputv(giov_t *darr, int len, int proc, _cmx_request* req)
{
  return CMX_SUCCESS;
}
int nbputv(giov_t *darr, int64_t len, int proc, _cmx_request* req)
{
  return CMX_SUCCESS;
}

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
int acc(int op, void *scale, void *src, int dst_offset, int bytes, int proc)
{
  return CMX_SUCCESS;
}
int acc(int op, void *scale, void *src, int64_t dst_offset, int64_t bytes, int proc)
{
  return CMX_SUCCESS;
}

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
    int *dst_stride, int *count, int stride_levels, int proc)
{
  return CMX_SUCCESS;
}
int accs(int op, void *scale, void *src, int64_t *src_stride, int64_t dst_offset,
    int64_t *dst_stride, int64_t *count, int stride_levels, int proc)
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
int accv(int op, void *scale, giov_t *darr, int len, int proc)
{
  return CMX_SUCCESS;
}
int accv(int op, void *scale, giov_t *darr, int64_t len, int proc)
{
  return CMX_SUCCESS;
}

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
    int bytes, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}
int nbacc(int op, void *scale, void *src, int64_t dst_offset,
    int64_t bytes, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}

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
    int stride_levels, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}
int nbaccs(int op, void *scale, void *src, int64_t *src_stride,
    int64_t dst_offset, int64_t *dst_stride, int64_t *count,
    int stride_levels, int proc, _cmx_request *req)
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
 * @param[out] req nonblocking request object
 * @return CMX_SUCCESS on success
 */
int nbaccv(int op, void *scale, giov_t *darr, int len, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}
int nbaccv(int op, void *scale, giov_t *darr, int64_t len, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}

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
int get(void *dst, int src_offset, int bytes, int proc)
{
  return CMX_SUCCESS;
}
int get(void *dst, int64_t src_offset, int64_t bytes, int proc)
{
  return CMX_SUCCESS;
}

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
    int *count, int stride_levels, int proc)
{
  return CMX_SUCCESS;
}
int gets(void *dst, int64_t *dst_stride, int64_t src_offset, int64_t *src_stride,
    int64_t *count, int stride_levels, int proc)
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
int getv(giov_t *darr, int len, int proc)
{
  return CMX_SUCCESS;
}
int getv(giov_t *darr, int64_t len, int proc)
{
  return CMX_SUCCESS;
}

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
int nbget(void *dst, int src_offset, int bytes, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}
int nbget(void *dst, int64_t src_offset, int64_t bytes, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}

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
    int *count, int stride_levels, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}
int nbgets(void *dst, int64_t *dst_stride, int64_t src_offset, int64_t *src_stride,
    int64_t *count, int stride_levels, int proc, _cmx_request *req)
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
int nbgetv(giov_t *darr, int len, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}
int nbgetv(giov_t *darr, int64_t len, int proc, _cmx_request *req)
{
  return CMX_SUCCESS;
}

/**
 * Flush all outgoing messages from me to the given proc on the group of the
 * allocation.
 * @param[in] proc the proc with which to flush outgoing messages
 * @return CMX_SUCCESS on success
 */
int fenceProc(int proc)
{
  return CMX_SUCCESS;
}

/**
 * Flush all outgoing messages to all procs in group of the allocation
 * @return CMX_SUCCESS on success
 */
int fenceAll()
{
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
int readModifyWrite(int op, void *ploc, int rem_offset, int extra, int proc)
{
  return CMX_SUCCESS;
}
int readModifyWrite(int op, void *ploc, int64_t rem_offset, int extra, int proc)
{
  return CMX_SUCCESS;
}

/**
 * Waits for completion of non-blocking cmx operations with explicit handles.
 * @param[in] req the request handle
 * @return CMX_SUCCESS on success
 */
int wait(_cmx_request *req)
{
  return CMX_SUCCESS;
}

/**
 * Checks completion status of non-blocking cmx operations with explicit
 * handles.
 * @param[in] req the request handle
 * @param[out] status true->completed, false->in progress
 * @return CMX_SUCCESS on success
 */
int test(_cmx_request *req, bool *status)
{
  return CMX_SUCCESS;
}

/**
 * Wait for all outstanding implicit non-blocking operations to finish on the
 * group of the allocation
 * @return CMX_SUCCESS on success
 */
int waitAll()
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
int waitProc(int proc)
{
  return CMX_SUCCESS;
}

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
extern int cmx_create_mutexes(int num)
{
}

/**
 * Collectively destroy all previously created locks.
 *
 * This function is always collective on the world group.
 *
 * @param[in] num number of locks to create locally
 * @return CMX_SUCCESS on success
 */
extern int cmx_destroy_mutexes()
{
}

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
extern int cmx_lock(int mutex, int proc)
{
}

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
extern int cmx_unlock(int mutex, int proc)
{
}
#endif
private:
  Group *p_group; // Group associated with allocation

  int p_datatype;   // enumeration describing data type of allocation

  std::vector<p_Allocation> p_list; // list describing allocation on other processors

  int p_rank; // rank id

  void *p_buf; // pointer to allocation on rank p_rank

  size_t p_bytes; // size of allocation
};

}; // CMX namespace

#endif /* _P_ALLOC_H */
