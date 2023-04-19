#ifndef _REG_CACHE_H_
#define _REG_CACHE_H_

#include "shmem.hpp"
#include "node_config.hpp"

#include <stddef.h>
#include <vector>
#include <map>

#define SHM_NAME_SIZE 31

namespace CMX {
/**
 * Enumerate the return codes for registration cache functions.
 */
typedef enum _reg_return_t {
    RR_SUCCESS=0,   /**< success */
    RR_FAILURE      /**< non-specific failure */
} reg_return_t;

/**
 * A registered contiguous memory region.
 */
typedef struct {
    void *buf;                  /* starting address of region */
    size_t len;                 /* length of region */
    void *mapped;               /* starting address of mmap'd region */
    int rank;                   /* rank where this region lives */
    char name[SHM_NAME_SIZE];   /* name of region */
    int use_dev;                /* memory is on a device */
} reg_entry_t;


#define TEST_FOR_INTERSECTION 0
#define TEST_FOR_CONTAINMENT 1

class p_Register {

  public:

  /**
   * Simple constructor
   */
  p_Register();

  /**
   * Simple destructor
   */
  ~p_Register();

  /**
   * Create internal data structures for the registration cache.
   *
   * @param[in] config    pointer to node config object
   * @param[in] shmem     pointer to shmem object
   *
   * @pre this function is called once to initialize the internal data
   * structures and cannot be called again until destroy() has been
   * called
   *
   * @see destroy()
   *
   * @return RR_SUCCESS on success
   */
  reg_return_t init(p_NodeConfig *config, p_Shmem *shmem);

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
  reg_return_t destroy();

  /**
   * Create a memory segment and return a reg_entry_t object that describes
   * it
   *
   * @param[in] bytes size of memory segment in bytes
   *
   * @return a reg_entry_t object that describes allocation
   */
  reg_entry_t* malloc(size_t bytes);

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
  reg_entry_t* find(int rank, void *buf, size_t len);

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
  reg_entry_t* insert(int rank, void *buf, size_t len,
      std::string name, void *mapped, int use_dev);


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
  reg_return_t remove(int rank, void *buf);

  /**
   * initialize all attributes of reg_entry_t struct
   * @param node pointer to reg_entry_t struct
   */
  reg_return_t nullify(reg_entry_t *node);

  private:
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
  seg_cmp(void *reg_addr, size_t reg_len, void *oth_addr,
      size_t oth_len, int op);

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
  seg_intersects(void *reg_addr, size_t reg_len,
      void *oth_addr, size_t oth_len);

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
    seg_contains(void *reg_addr, size_t reg_len,
        void *oth_addr, size_t oth_len);

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
    intersects(reg_entry_t *reg_entry,
        void *buf, size_t len);

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
    contains(reg_entry_t *reg_entry, void *buf, size_t len);

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
    find_intersection(int rank, void *buf, size_t len);


  std::vector<std::map<void*,reg_entry_t*> > p_list;

  p_Shmem *p_shmem;

  p_NodeConfig *p_config;

};
} // CMX
#endif /* _REG_CACHE_H_ */
