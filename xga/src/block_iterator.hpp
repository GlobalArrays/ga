/* Object for managing the distribution of data on a processor */
#ifndef _BLOCK_ITERATOR_H
#define _BLOCK_ITERATOR_H

#include "xga_types.hpp"
#include "xga_private.hpp";

namespace XGA {

class BlockIterator {

public:
  /**
   * Basic constructor
   */
  BlockIterator();

  /**
   * Basic destructor
   */
  ~BlockIterator();

  /**
   * Initialize block iterator
   * @param[in] ga pointer to global array creating block iterator
   * @param[in] lo,hi bounding indices of requested patch in global array
   * @param[in] block_dims
   * @param[in] nblock
   * @param[in] dims
   * @param[in] distr data layout in global array
   */
  void init(GlobalArray *ga, int64_t *lo, int64_t *hi, data_distribution distr);

   /**
    * Reset iterator to initial condition
    */
  void reset();

  /**
   * @param[out] proc processor containing block
   * @param[out] plo,phi bounding indices of next block
   * @param[out] prem pointer to data in next block
   * @param[out] ldrem array of strides for next block
   * @return true if there is a next block, false otherwise
   */
  bool nextBlock(int *proc, int64_t *plo, int64_t *phi,
      void *prem, int64_t *ldrem);

  /**
   * Check if this is the last block
   * @return true if this is the last block
   */
  bool last();

private:

  data_distribution distr_type; /* Distribution type */
  int     ndim;                 /* dimension of array */
  int64_t *lo;                  /* lower corner of block in array */
  int64_t *hi;                  /* upper corner of block in array */
  int64_t count;                /* counter to keep track of blocks */
  const int64_t *map;           /* bounds of individual blocks */
  const int     *proclist;      /* list of procs containing data */
  const int64_t *mapc;          /* pointer to GA mapc dat */
  int     nproc;                /* number of processors */
  int64_t offset;               /* offset to start of block in data segment */
  int     iblock;               /* counter tracking blocks on processor */
  int64_t *lobuf;               /* lower corner of sub-block */
  int64_t *hibuf;               /* upper corner of sub-block */
  int64_t *blk_num;             /* number of blocks in each direction */
  int64_t *blk_size;            /* maximum dimensions of block */
  int64_t *blk_inc;             /* dimensions of partial blocks */
  int64_t *blk_ld;              /* stride between blocks */
  int64_t *hlf_blk;             /* ??? */
  int64_t *blk_dims;            /* dimensions of total data on processor */
  int     *proc_index;          /* location of processor in proc grid */
  int     *index;               /* location of current sub-block */

  GlobalArray *p_GA;
};
}
#endif
