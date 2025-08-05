/* XGA private header file */
#ifndef _XGA_PRIVATE_HEADER_H
#define _XGA_PRIVATE_HEADER_H
#include "cmx_group.hpp"
#include "cmx_environment.hpp"
#include "cmx_alloc.hpp"
#include "xga_types.hpp"
#include "xga_macros.hpp"
#include "xga_environment.hpp"
#include "xga_group.hpp"

#include <complex>

#define MAXDIM 7

#define MAX_NUM_NB_HDLS 256

namespace XGA {

class p_GA {

public:

  /**
   * Constructor
   * @param[in] group home group for global array
   * @param[in] ndim dimension of global array
   * @param[in] dims dimensions of global array
   * @param[in] type data type of array
   */
  p_GA(Group *group, int ndim, int64_t *dims, xga_types type);

  /**
   * Basic destructor
   */
  ~p_GA();

  /**
   * Set data distribution type
   * @param[in] distr data distribution type
   */
  void setDataDistribution(XGA::data_distribution distr);

  /**
   * @param[in] mapc array containing partitions along each axis
   * @param[in] nblock array containing processor decomposition
   */
  void setIrregularDistribution(int64_t *mapc, int *nblock);

  /**
   * Allocate resources to create global array
   */
  void allocate();

  /**
   * Find block owned by processor proc
   * @param[in] proc processor being queried
   * @param[out] lo,hi lower and upper bounding indices of block
   *             owned by processor proc
   */
  void distribution(const int proc, int64_t *lo, int64_t *hi);

  /**
   * @param[in] lo,hi lower and upper indices of patch in global array
   * @param[out] map list of lower and upper indices for portion of
   *             patch the exists on each processor containing a portion
   *             of the patch. The map is constructed so that for a D
   *             dimensional global array, the first D elements are the
   *             lower indices on the first processor in proclist, the
   *             next D elements are the upper indices of the first
   *             processor in proclist, the next D elements are the
   *             lower indices for the second processor in proclist and
   *             so on.
   * @param[out] proclist list of processors containing some portion of
   *             patch
   * @param[out] np total number of processors containing some portion
   *             of the patch
   * @return false if bounds of patch are invalid
   *
   * For a block cyclic data distribution, this function returns a list
   * of blocks that cover the region, along with the lower and upper
   * indices of each block.
   */
  bool locateRegion(const int64_t *lo, const int64_t *hi,
      std::vector<int64_t> &map, std::vector<int> &proclist, int *np);

  /**
   * Locate process that owns element corresponding to subscript
   * @param[in] subscript n-tuple identifiying element in array
   * @param[out] ownere process that owns element
   * @return false if element out of bounds
   */
  bool locate(const int64_t *subscript, int *owner);

  /**
   * Access data corresponding to a specific patch
   * @param[in] plo,phi lower and upper indices of patch
   * @param[out] rptr pointer to data
   * @param[out] ld array of strides for data
   */
  void accessPtr(int64_t *plo, int64_t *phi, void **rptr, int64_t *ld);

  /**
   * Access data corresponding to a specific block
   * @param[in] index indices of block in proc grid or block cyclic layout
   * @param[out] rptr pointer to data
   * @param[out] ld array of strides for block
   */
  void accessBlockGridPtr(int *index, void **rptr, int64_t *ld);

  /**
   * Copy data from local buffer to global array
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   */
  void put(int64_t *lo, int64_t *hi, void* buf, int64_t *ld);

private:

  /**
   * Routine to create data distribution
   * @param[in] ndim number of dimensions in the data array and the process
   *            grid. There is no provision for requesting a process grid
   *            with fewer dimensions than the data array
   * @param[in] dims extents of the each dimension of the data array. This
   *            array is of size ndim and is destroyed by the routine
   * @param[in] nproc number of processors onto which the distribution takes
   *            place
   * @param[in] threshold minimum acceptable value of the load balance ratio
   * @param[in] bias when set to a positive value, the rightmost axes of the
   *            data array are preferentially distributed, similarly when bias
   *            is negative. When bias is zero, the heuristic attempts to deal
   *            processes equally among the axes
   * @param[in/out] blk granularity of data mapping. The number of consecutive
   *            elements along each dimension of the array. Upon output: the
   *            extents of the local array assigned to the process
   * @param[out] pedims number of processors along each dimension of the data
   *            array
   */
  void ddb_h2(int ndim, int64_t *dims, int nproc, double threshold, int bias,
      int64_t *blk, int *pedims);

  /* Routines to set up internal iterator of data blocks */

  /**
   * Initialize block iterator
   * @param[in] lo,hi bounding indices of requested patch in global array
   */
  void initIterator(const int64_t *lo, const int64_t *hi);

   /**
    * Reset iterator to initial condition
    */
  void resetIterator();

  /**
   * @param[out] proc processor containing block
   * @param[out] plo,phi bounding indices of next block
   * @param[out] prem pointer to data in next block
   * @param[out] ldrem array of strides for next block
   * @return true if there is a next block, false otherwise
   */
  bool nextBlock(int *proc, int64_t *plo[], int64_t *phi[],
      char **prem, int64_t *ldrem);

  /**
   * Check if this is the last block
   * @return true if this is the last block
   */
  bool lastBlock();

  /**
   * clean up iterator
   */
  void destroyIterator();

  /**
   * Functions that iterate over locally held blocks in a global array. These
   * are used in some routines such as copy
   */

  /**
   * Initialize a local iterator
   */
  void localInit();

  /**
   * Get the next sub-block from local portion of global array
   * @param plo indices for lower corner of block
   * @param phi indices for upper corner of block
   * @param ptr pointer to local buffer
   * @param ld array of strides for local block
   * @return returns false if there is no new block, true otherwise
   */
  bool nextLocalBlock(int64_t *plo, int64_t *phi, char **ptr, int64_t *ld);

  /**
   * Internal implementation of put call that handles both blocking and
   * non-blocking variants
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   * @param[out] req non-blocking request handle
   */
  void putCommon(int64_t *lo, int64_t *hi, void* buf, int64_t *ld,
      xga_request **req);

private:

  int p_datatype = XGA_UNKNOWN; /* data type */
  data_distribution p_distr;    /* data distribution */
  int     p_ndim;               /* dimension of array */
  int64_t p_dims[MAXDIM];       /* dimensions of array */
  int64_t chunk[MAXDIM];        /* chunking array */
  int     nblock[MAXDIM];       /* number of blocks per dimension */
  int64_t *p_mapc;              /* block distribution map */
  std::vector<int64_t> map;     /* distribution map for iterator */
  int     nproc;                /* number of processors */
  std::vector<int> proclist;    /* list of procs containing data */
  int     proc_grid[MAXDIM];    /* processor array */
  double  scale[MAXDIM];        /* nblock/dim (precomputed) */
  int64_t p_size;               /* size of local data, in bytes */
  int64_t p_elemsize;           /* size of data element */
  bool    ghosts;               /* flag indicate ghost cells */
  int64_t width[MAXDIM];        /* boundary cells per dimension */
  int64_t p_lo[MAXDIM];         /* lower indices of local block */
  void    **ptr;                /* array of pointers to remoted data */
  bool    p_active;             /* data has been allocated to array */


  /* iterator parameters */
  int64_t it_lo[MAXDIM];        /* lower corner of block in array */
  int64_t it_hi[MAXDIM];        /* upper corner of block in array */
  int64_t count;                /* counter to keep track of blocks */
  int64_t offset;               /* offset to start of block in data segment */
  int     iblock;               /* counter tracking blocks on processor */
  int64_t lobuf[MAXDIM];        /* lower corner of sub-block */
  int64_t hibuf[MAXDIM];        /* upper corner of sub-block */
  int64_t blk_num[MAXDIM];      /* number of blocks in each direction */
  int64_t blk_size[MAXDIM];     /* maximum dimensions of block */
  int64_t blk_inc[MAXDIM];      /* dimensions of partial blocks */
  int64_t blk_ld[MAXDIM];       /* stride between blocks */
  int64_t hlf_blk[MAXDIM];      /* ??? */
  int64_t blk_dims[MAXDIM];     /* dimensions of total data on processor */
  int     proc_index[MAXDIM];   /* location of processor in proc grid */
  int     index[MAXDIM];        /* location of current sub-block */
  int64_t block_total;          /* total number of blocks in array */

  Environment *p_env;
  Group *p_group;
  CMX::Allocation *p_alloc;
};
}
#endif
