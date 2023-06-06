/**
 * Registration cache.
 *
 * Defensive programming via copious CMX_ASSERT statements is encouraged.
 */

/* C headers */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* 3rd party headers */
#include <mpi.h>

/* our headers */
#include "defines.hpp"
#include "reg_cache.hpp"

#define CMX_ASSERT(WHAT) ((void)(0))

namespace CMX {

#define TEST_FOR_INTERSECTION 0
#define TEST_FOR_CONTAINMENT 1

/**
 * Simple constructor
 */
p_Register::p_Register()
{
}

/**
 * Simple destructor
 */
p_Register::~p_Register()
{
}

/**
 * Create internal data structures for the registration cache.
 *
 * @param[in] config    pointer to node config object
 * @param[in] shmem     pointer to shmem object
 *
 * @pre this function is called once to initialize the internal data
 * structures and cannot be called again until reg_cache_destroy() has been
 * called
 *
 * @see reg_cache_destroy()
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
p_Register::init(p_NodeConfig *config, p_Shmem *shmem)
{
  int i = 0;
  p_config = config;
  p_shmem = shmem;

  p_list.resize(p_config->world_size());

  return RR_SUCCESS;
}


/**
 * Deregister and destroy all cache entries and associated buffers.
 *
 * @pre this function is called once to destroy the internal data structures
 * and cannot be called again until reg_cache_init() has been called
 *
 * @see reg_cache_init()
 *
 * @return RR_SUCCESS on success
 */
reg_return_t p_Register::destroy()
{
  int i = 0;

  int nprocs = p_list.size();

  for (i = 0; i < nprocs; ++i) {
    std::map<void*,reg_entry_t*>::iterator it = p_list[i].begin();
    while (it != p_list[i].end()) {
      if (it->second->rank == p_config->rank()) {
        p_shmem->free(it->second->name, it->second->buf, it->second->len);
      } else {
        p_shmem->unmap(it->second->buf, it->second->len);
      }
      delete it->second;
      it++;
    }
    p_list[i].clear();
  }

  p_list.clear();

  return RR_SUCCESS;
}

/**
 * Create a memory segment and return a reg_entry_t object that describes
 * it *
 * @param[in] bytes size of memory segment in bytes
 *
 * @return a reg_entry_t object that describes allocation
 */
reg_entry_t* p_Register::malloc(size_t bytes)
{
  char *name;
  void *memory = NULL;
  reg_entry_t *reg_entry = NULL;

  if (0 == bytes) {
    return NULL;
  }

  /* create my shared memory object */
  name = p_shmem->generateName(p_config->rank());
  memory = p_shmem->create(name, bytes);

  /* register the memory locally */
  reg_entry = insert(
      p_config->rank(), memory, bytes, name, memory, 0);

  if (NULL == reg_entry) {
    printf("Create memory segment failes\n");
    CMX_ASSERT(0);
  }
  delete [] name;

  return reg_entry;
}

/**
 * Locate a registration cache entry which contains the given segment
 * completely.
 *
 * @param[in] rank  world rank of the process
 * @param[in] buf   starting address of the buffer
 * @parma[in] len   length of the buffer
 * 
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre reg_cache_init() was previously called
 *
 * @return the reg cache entry, or NULL on failure
 */
reg_entry_t*
p_Register::find(int rank, void *buf, size_t len)
{
  reg_entry_t *entry = NULL;

  CMX_ASSERT(0 <= rank && rank < p_list.size());

  std::map<void*,reg_entry_t*>::iterator it = p_list[rank].begin();

  while (it != p_list[rank].end()) {
    if (RR_SUCCESS == contains(it->second, buf, len)) {
      entry = it->second;
      it++;
      break;
    }
    it++;
  }

  /* we CMX_ASSERT that the found entry was unique */
  while (it != p_list[rank].end()) {
    if (RR_SUCCESS == contains(it->second, buf, len)) {
      CMX_ASSERT(0);
    }
    it++;
  }

  return entry;
}


/**
 * Locate a registration cache entry which intersects the given segment.
 *
 * @param[in] rank  rank of the process
 * @param[in] buf   starting address of the buffer
 * @parma[in] len   length of the buffer
 * 
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre reg_cache_init() was previously called
 *
 * @return the reg cache entry, or NULL on failure
 */
reg_entry_t*
p_Register::find_intersection(int rank, void *buf, size_t len)
{
  reg_entry_t *entry = NULL;

  /* preconditions */
  CMX_ASSERT(0 <= rank && rank < list.size());


  std::map<void*,reg_entry_t*>::iterator it = p_list[rank].begin();
  while (it != p_list[rank].end()) {
    if (RR_SUCCESS == intersects(it->second, buf, len)) {
      entry = it->second;
    }
    it++;
  }

  /* we CMX_ASSERT that the found entry was unique */
  while (it != p_list[rank].end()) {
    if (RR_SUCCESS == intersects(it->second, buf, len)) {
      CMX_ASSERT(0);
    }
    it++;
  }

  return entry;
}


/**
 * Create a new registration entry based on the given members.
 *
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre NULL != buf
 * @pre 0 <= len
 * @pre reg_cache_init() was previously called
 * @pre NULL == reg_cache_find(rank, buf, 0)
 * @pre NULL == reg_cache_find_intersection(rank, buf, 0)
 *
 * @return RR_SUCCESS on success
 */
reg_entry_t*
p_Register::insert(int rank, void *buf, size_t len,
    const char *name, void *mapped, int use_dev)
{
  reg_entry_t *node = NULL;

  /* preconditions */
  CMX_ASSERT(0 <= rank && rank < p_list.size());
  CMX_ASSERT(NULL != buf);
  CMX_ASSERT(len >= 0);
  CMX_ASSERT(NULL == find(rank, buf, len));
  CMX_ASSERT(NULL == find_intersection(rank, buf, len));

  /* allocate the new entry */
  node = new reg_entry_t;
  CMX_ASSERT(node);

  /* initialize the new entry */
  node->rank = rank;
  node->buf = buf;
  node->len = len;
  node->use_dev = use_dev;
  (void)memcpy(node->name, name, SHM_NAME_SIZE);
  node->mapped = mapped;

  /* push new entry to tail of linked list */
  p_list[rank].insert(std::pair<void*,reg_entry_t*>(buf,node));

  return node;
}


/**
 * Removes the reg cache entry associated with the given rank and buffer.
 * Note that this does not actually remove the buffer.
 *
 * @param[in] rank
 * @param[in] buf
 *
 * @pre 0 <= rank && rank < reg_nprocs
 * @pre NULL != buf
 * @pre reg_cache_init() was previously called
 * @pre NULL != reg_cache_find(rank, buf, 0)
 *
 * @return RR_SUCCESS on success
 *         RR_FAILURE otherwise
 */
reg_return_t
p_Register::remove(int rank, void *buf)
{
  reg_return_t status = RR_FAILURE;

  CMX_ASSERT(0 <= rank && rank < p_list.size());
  CMX_ASSERT(NULL != buf);
  CMX_ASSERT(NULL != find(rank, buf, 0));

  std::map<void*,reg_entry_t*>::iterator it = p_list[rank].find(buf);
  /* this is more restrictive than reg_cache_find() in that we locate
   * exactlty the same region starting address */
  /* we should have found an entry */
  if (it == p_list[rank].end()) {
    CMX_ASSERT(0);
    return RR_FAILURE;
  }

  reg_entry_t *entry = it->second;
  p_list[rank].erase(it);
  delete entry;

  return status;
}


/**
 * initialize all attributes of reg_entry_t struct
 * @param node pointer to reg_entry_t struct
 */
reg_return_t p_Register::nullify(reg_entry_t *node)
{
  node->buf = NULL;
  node->len = 0;
  node->mapped = NULL;
  node->rank = -1;
  node->use_dev = 0;
  (void)memset(node->name, 0, SHM_NAME_SIZE);

  return RR_SUCCESS;
}

/**
 * Detects whether two memory segments intersect or one contains the other.
 *
 * @param[in] reg_addr  starting address of original segment
 * @param[in] reg_len   length of original segment
 * @param[in] oth_addr  starting address of other segment
 * @param[in] oth_len   length of other segment
 * @param[in] op        op to perform, either TEST_FOR_INTERSECTION or
 *                      TEST_FOR_CONTAINMENT
 *
 * @pre NULL != reg_beg
 * @pre NULL != oth_beg
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
p_Register::seg_cmp(void *reg_addr, size_t reg_len,
    void *oth_addr, size_t oth_len, int op)
{
  ptrdiff_t reg_beg = 0;
  ptrdiff_t reg_end = 0;
  ptrdiff_t oth_beg = 0;
  ptrdiff_t oth_end = 0;
  int result = 0;

  /* preconditions */
  CMX_ASSERT(NULL != reg_addr);
  CMX_ASSERT(NULL != oth_addr);

  /* casts to ptrdiff_t since arithmetic on void* is undefined */
  reg_beg = (ptrdiff_t)(reg_addr);
  reg_end = reg_beg + (ptrdiff_t)(reg_len);
  oth_beg = (ptrdiff_t)(oth_addr);
  oth_end = oth_beg + (ptrdiff_t)(oth_len);

  /* hack? we had problems with adjacent registered memory regions and
   * when the length of the query region was 0 */
  if (oth_beg == oth_end) {
    oth_end += 1;
  }

  switch (op) {
    case TEST_FOR_INTERSECTION:
      result = (reg_beg >= oth_beg && reg_beg <  oth_end) ||
        (reg_end >  oth_beg && reg_end <= oth_end);
#if DEBUG
      printf("[%d] TEST_FOR_INTERSECTION "
          "(%td >= %td [%d] && %td < %td [%d]) ||"
          "(%td > %td [%d] && %td <= %td [%d])\n",
          g_state.rank,
          reg_beg, oth_beg, (reg_beg >= oth_beg),
          reg_beg, oth_end, (reg_beg < oth_end),
          reg_end, oth_beg, (reg_end > oth_beg),
          reg_end, oth_end, (reg_end <= oth_end));
#endif
      break;
    case TEST_FOR_CONTAINMENT:
      result = reg_beg <= oth_beg && reg_end >= oth_end;
#if DEBUG
      printf("[%d] TEST_FOR_CONTAINMENT "
          "%td <= %td [%d] && %td >= %td [%d]\n",
          g_state.rank,
          reg_beg, oth_beg, (reg_beg <= oth_beg),
          reg_end, oth_end, (reg_end >= oth_end));
#endif
      break;
    default:
      CMX_ASSERT(0);
  }

  if (result) {
    return RR_SUCCESS;
  }
  else {
    return RR_FAILURE;
  }
}


/**
 * Detects whether two memory segments intersect.
 *
 * @param[in] reg_addr starting address of original segment
 * @param[in] reg_len  length of original segment
 * @param[in] oth_addr starting address of other segment
 * @param[in] oth_len  length of other segment
 *
 * @pre NULL != reg_beg
 * @pre NULL != oth_beg
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
p_Register::seg_intersects(void *reg_addr, size_t reg_len,
    void *oth_addr, size_t oth_len)
{
  /* preconditions */
  CMX_ASSERT(NULL != reg_addr);
  CMX_ASSERT(NULL != oth_addr);

  return seg_cmp(
      reg_addr, reg_len,
      oth_addr, oth_len,
      TEST_FOR_INTERSECTION);
}


/**
 * Detects whether the first memory segment contains the other.
 *
 * @param[in] reg_addr starting address of original segment
 * @param[in] reg_len  length of original segment
 * @param[in] oth_addr starting address of other segment
 * @param[in] oth_len  length of other segment
 *
 * @pre NULL != reg_beg
 * @pre NULL != oth_beg
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
p_Register::seg_contains(void *reg_addr, size_t reg_len,
    void *oth_addr, size_t oth_len)
{
  /* preconditions */
  CMX_ASSERT(NULL != reg_addr);
  CMX_ASSERT(NULL != oth_addr);

  return seg_cmp(
      reg_addr, reg_len,
      oth_addr, oth_len,
      TEST_FOR_CONTAINMENT);
}


/**
 * Detects whether two memory segments intersect.
 *
 * @param[in] reg_entry the registration entry
 * @param[in] buf       starting address for the contiguous memory region
 * @param[in] len       length of the contiguous memory region
 *
 * @pre NULL != reg_entry
 * @pre NULL != buf
 * @pre len >= 0
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
p_Register::intersects(reg_entry_t *reg_entry, void *buf, size_t len)
{
  /* preconditions */
  CMX_ASSERT(NULL != reg_entry);
  CMX_ASSERT(NULL != buf);
  CMX_ASSERT(len >= 0);

  return seg_intersects(
      reg_entry->buf, reg_entry->len,
      buf, len);
}


/**
 * Detects whether the first memory segment contains the other.
 *
 * @param[in] reg_entry the registration entry
 * @param[in] buf       starting address for the contiguous memory region
 * @param[in] len       length of the contiguous memory region
 *
 * @pre NULL != reg_entry
 * @pre NULL != buf
 * @pre len >= 0
 *
 * @return RR_SUCCESS on success
 */
reg_return_t
p_Register::contains(reg_entry_t *reg_entry, void *buf, size_t len)
{

  /* preconditions */
  CMX_ASSERT(NULL != reg_entry);
  CMX_ASSERT(NULL != buf);
  CMX_ASSERT(len >= 0);

  return seg_contains(
      reg_entry->buf, reg_entry->len,
      buf, len);
}

} // CMX
