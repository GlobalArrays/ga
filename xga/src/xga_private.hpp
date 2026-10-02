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

namespace XGA {

class p_GA {

public:

  enum{XGA_GATHER, XGA_SCATTER, XGA_SCATTERACC};

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
   * Set block sizes and processor grid for ScaLAPACK-style data
   * distribution. This is only strictly an ScaLAPACK distribution in
   * 2 dimensions but the generalization to higher dimensions is
   * straightforward
   * @param[in] dims dimensions of individual blocks
   * @param[in] prod_grid dimension of processor grid
   */
  void setBlockLayout(int64_t *dims, int *proc_grid);

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
   * Duplicate a global array. New array has same datatype, size and
   * data partition but individual values are not initialized.
   * @return pointer to new global array
   */
  p_GA* duplicate();

  /**
   * Copy contents of array B into calling array. Arrays must be same size and
   * datatype
   * @param g_b source array
   */
  void copy(p_GA *g_b);

  /**
   * Copy a patch of array B to a patch in the calling array. Array must be the
   * same datatype.
   * @param trans flag signifying whether to transpose data when copying
   * @param alo, ahi bounding indices of target patch
   * @param g_b source array
   * @param blo, bhi bounding indices of source patch
   */
  void copyPatch(char trans, int64_t *alo, int64_t *ahi,
      p_GA *g_b, int64_t *blo, int64_t *bhi);

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
   * @param[in] index indices of block in proc grid
   * @param[out] rptr pointer to data
   * @param[out] ld array of strides for block
   */
  void accessBlockGridPtr(int *index, void **rptr, int64_t *ld);

  /**
   * Return pointer to data corresponding to block indexed by idx.
   * Assume C-style ordering
   * @param[in] idx index of block
   * @param[out] rptr pointer to data
   * @param[out] ld array of strides for block
   */
  void accessBlockPtr(int idx, void **rptr, int64_t *ld);

  /**
   * Return pointer to data owned by this processors
   * @param[out] rptr pointer to local data
   * @param[out] nelem number of elements owned by this processor
   */
  void accessSegmentPtr(void **rptr, int64_t *nelem);

  /**
   * Release data corresponding to a specific patch
   * @param[in] plo,phi lower and upper indices of patch
   */
  void releasePtr(int64_t *plo, int64_t *phi);
  void releaseUpdatePtr(int64_t *plo, int64_t *phi);

  /**
   * Release data corresponding to a specific block
   * in the proc grid array
   * @param[in] index indices of block in proc grid
   */
  void releaseBlockGridPtr(int *index);
  void releaseUpdateBlockGridPtr(int *index);

  /**
   * Release data corresponding to a specific block indexed
   * using a C-style indexing convention
   * @param[in] index index of block
   */
  void releaseBlockPtr(int index);
  void releaseUpdateBlockPtr(int index);

  /**
   * Release data corresponding to this process
   */
  void releaseSegmentPtr();
  void releaseUpdateSegmentPtr();

  /**
   * Synchronize global array across all processor that are hosting
   * the array
   */
  void sync();

  /**
   * Copy data from local buffer to global array
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   */
  void put(int64_t *lo, int64_t *hi, void* buf, int64_t *ld);

  /**
   * Copy data from global array to local buffer
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   */
  void get(int64_t *lo, int64_t *hi, void* buf, int64_t *ld);

  /**
   * Accumulate data from local buffer to global array
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   * @param[in] alpha scale factor for adding contents of buffer
   *            to global array
   */
    void acc(int64_t *lo, int64_t *hi, void* buf, int64_t *ld, void* alpha);

    /**
     * Scatter values to random locations in a global array
     * @param[in] v array containing values to be scattered to array. The
     *            type of values in v must match the type of values
     *            in the global array
     * @param[in] subscript array of indices representing locations of
     *            values in global array. Each ndim locations represents
     *            the index location of one value
     * @param[in] nv number of values to scattered
     * @param[in] idxtype flag indicating size of index type (0 for int,
     *            1 for int64_t)
     */
    void scatter(void *v, void *subscript, int64_t nv, int idxtype);

    /**
     * Gather values from random locations in a global array
     * @param[in] v array containing values gathered from array. The
     *            type of values in v must match the type of values
     *            in the global array
     * @param[in] subscript array of indices representing locations of
     *            values in global array. Each ndim locations represents
     *            the index location of one value
     * @param[in] nv number of values to gathered
     * @param[in] idxtype flag indicating size of index type (0 for int,
     *            1 for int64_t)
     */
    void gather(void *v, void *subscript, int64_t nv, int idxtype);

    /**
     * Accumulate values to random locations in a global array
     * @param[in] v array containing values to be accumulated to array. The
     *            type of values in v must match the type of values
     *            in the global array
     * @param[in] subscript array of indices representing locations of
     *            values in global array. Each ndim locations represents
     *            the index location of one value
     * @param[in] nv number of values to accumulated
     * @param[in] scale scale factor to multiply each value by before being
     *            accumulated
     * @param[in] idxtype flag indicating size of index type (0 for int,
     *            1 for int64_t);
     */
    void scatterAcc(void *v, void *subscript, int64_t nv, void *alpha,
        int idxtype);

    /**
     * Read the value at location indicated by subscript and increment b
     * the amount inc. This operation is atomic with respect to other read
     * increment operations.
     * @param[in] subscript locate of element to be read and incremented
     * @param[in] inc amount to increment element
     * @param[out] pointer to variable containing the current value of element
     */
    void readInc(int64_t *subscript, void *inc, void *result);

    /**
     * Set all values in the array to zero
     */
    void zero();

    /**
     * Fill array with a single value
     * @param value pointer to value being filled
     */
    void fill(void *value);

    /**
     * Scale all elements of array
     * @param value scale factor for all elements
     */
    void scale(void *value);

    /**
     * Scale all elements in a patch of an array
     * @param lo, hi bounding indices of patch
     * @param value scale factor for elements in patch
     */
    void scalePatch(int64_t *lo, int64_t *hi, void *value);

    /**
     * Add two global arrays to get a third array. The calling array must be the
     * same size and dimension of the two arrays in the argument list, all three
     * arrays must also be the same data type. The calling array can also be the
     * same as one of the two arrays in the argument list. The parameters alpha
     * and beta can be used to scale the arrays before performing the sum.
     *   C = alpha*A + beta*B
     * If alpha or beta are NULL, assume that the are set to 1
     * @param alpha parameter to scale array A
     * @param g_a first array (A) in sum
     * @param beta parameter to scale array B
     * @param g_b second array (B) in sum
     */
    void add(void *alpha, p_GA *g_a, void *beta, p_GA *g_b);

    /**
     * Add patches of two global arrays to get a new patch in a third array.
     * The calling array must be the same type as the other two arrays and
     * the dimensions of the patches must be compatible. The calling array
     * can also be the same as one of the two arrays in the argument list.
     * The parameters alpha and beta can be used to scale the patches before
     * performing the sum.
     *   C = alpha*A + beta*B
     * If alpha or beta are NULL, assume that the are set to 1
     * @param alpha parameter to scale array A
     * @param g_a first array (A) in sum
     * @param alo, ahi bounding indices of patch in array A
     * @param beta parameter to scale array B
     * @param g_b second array (B) in sum
     * @param blo, bhi bounding indices of patch in array B
     * @param clo, chi bounding indices of patch in array C
     */
    void addPatch(void* alpha, p_GA *g_a, int64_t *alo, int64_t *ahi,
        void* beta,  p_GA *g_b, int64_t *blo, int64_t *bhi,
        int64_t *clo, int64_t *chi);

private:

  /**
   * Routine to create data distribution
   * @param[in] ndim number of dimensions in the data array and the process
   *            grid. There is no provision for requesting a process grid
   *            with fewer dimensions than the data array
   * @param[in] dims extents of the each dimension of the data array. This
   *            array is of size ndim and is destroyed by the routine
   * @param[in] nprocs number of processors onto which the distribution takes
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
  void ddb_h2(int ndim, int64_t *dims, int nprocs, double threshold, int bias,
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
      xga_request *req);

  /**
   * Internal implementation of get call that handles both blocking and
   * non-blocking variants
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   * @param[out] req non-blocking request handle
   */
  void getCommon(int64_t *lo, int64_t *hi, void* buf, int64_t *ld,
      xga_request *req);

  /**
   * Accumulate data from local buffer to global array
   * @param[in] lo,hi bounding indices of block in global array
   * @param[in] buf pointer to first element in local buffer
   * @param[in] ld strides in local buffer
   * @param[in] alpha scale factor for adding contents of buffer
   *            to global array
   * @param[out] req non-blocking request handle
   */
  void accCommon(int64_t *lo, int64_t *hi, void* buf, int64_t *ld,
      void* alpha, xga_request *req);

  /**
   * Generic routine for implementing gather, scatter and scatter-accumulate
   * operations
   * @param[in] op enum indicating which operation is being performed
   * @param[in] v pointer to array containing values to be scattered
   * @param[in] subscript array containing indices of values to be moved
   * @param[in] idxtype flag indicating size of index (0 int, 1 int64_t)
   * @param[in] nv number of values being moved
   * @param[in] alpha scale factor that is used in scatter-accumulate operation
   * @param[out] req non-blocking request handle
   */
  void gatscatCommon(int op, void *v, void *subscript, int idxtype,
      int64_t nv, void *alpha, xga_request *req);

  /**
   * Utility function to print subscripts
   * @param[in] pre character string before subscript
   * @param[in] ndim dimension of subscript
   * @param[in] subscript array containing subscript values
   * @param[in] post character string after subscript
   */
  void printSubscript(const char *pre, const int ndim, const int64_t *subscript,
      const char *post);

  /**
   * count number of elements in map array
   * @return sum of number of partitions in each dimension
   */
  int calc_maplen();

  /**
   * compare data distribution of two arrays
   * @param[i] g_a comparision array
   * @return true if arrays have the same data distribution, false otherwise
   */
  bool compare_distr(p_GA *g_a);

  /**
   * Compare two patches to see if they are identical
   * @param andim dimension of patch A
   * @param alo, ahi lower and upper dimensions of patch A
   * @param andim dimension of patch B
   * @param alo, ahi lower and upper dimensions of patch B
   * @return true if patches match
   */
  bool comp_patch(int andim, int64_t *alo, int64_t *ahi,
                  int bndim, int64_t *blo, int64_t *bhi);

  /**
   * Check if two patches intersect and return the intersection
   * in second patch
   * @param lo, hi bounding indices for first block
   * @param lop, hip bounding indices for second block
   * @param ndim number of dimensions for both blocks
   */
  bool patch_intersect(int64_t *lo, int64_t *hi,
      int64_t *lop, int64_t *hip, int ndim);

  /**
   * compute index from subscript and convert it back to subscript
   * in another array
   * @param ndims number of dimensions in index of source block
   * @param los lower index of current block
   * @param blos lower index of source block
   * @param dimss array of strides for source block
   * @param ndimsd number of dimensions in index of destination block
   * @param blos lower index of destination block
   * @param dimss array of strides for destination block
   */
  void dest_indices(int ndims, int64_t *los, int64_t *blos, int64_t *dimss,
      int ndimd, int64_t *lod, int64_t *blod, int64_t *dimsd);

  /**
   * Utility function to add patch values together
   * @param alpha, beta parameters multiplying individual arrays
   * @param ndim dimension of array
   * @param loC, hiC lower and upper bounding indices of patch
   * @param ldC array of strides for patch
   * @param A_ptr, B_ptr, C_ptr pointer to individual chunks of data
   */
  template <typename _type>
    void add_patch_values(void *alpha, void *beta, int64_t *loC, int64_t *hiC,
        int64_t *ldC, void *A_ptr, void *B_ptr, void *C_ptr)
    {
      int64_t bvalue[MAXDIM], bunit[MAXDIM], baseldC[MAXDIM];
      int64_t idx, n1dim;
      int64_t i, j;
      int ndim = p_ndim;
      _type talpha = *(reinterpret_cast<_type*>(alpha));
      _type tbeta = *(reinterpret_cast<_type*>(beta));
      _type *aptr = reinterpret_cast<_type*>(A_ptr);
      _type *bptr = reinterpret_cast<_type*>(B_ptr);
      _type *cptr = reinterpret_cast<_type*>(C_ptr);
      /* compute "local" add */

      /* number of n-element of the first dimension */
      n1dim = 1; for(i=p_ndim-2; i>=0; i--) n1dim *= (hiC[i] - loC[i] + 1);

      /* calculate the destination indices */
      bvalue[ndim-1] = 0;
      bunit[ndim-1] = 1;
      if (ndim > 1) {
        bvalue[ndim-2] = 0;
        bunit[ndim-2] = 1;
      }
      /* baseld[ndim-2] = ld[ndim-1]
       * baseld[ndim-3] = ld[ndim-1] * ld[ndim-2]
       * baseld[ndim-4] = ld[ndim-1] * ld[ndim-2] * ld[ndim-3] .....
       */
      baseldC[ndim-1] = ldC[ndim-1]; 
      if (ndim > 1) {
        baseldC[ndim-2] = baseldC[ndim-1] *ldC[ndim-2];
      }
      for(i=ndim-3; i>=0; i--) {
        bvalue[i] = 0;
        bunit[i] = bunit[i+1] * (hiC[i+1] - loC[i+1] + 1);
        baseldC[i] = baseldC[i+1] * ldC[i];
      }
      for (i=0; i<n1dim; i++) {
        idx = 0;
        for (j=ndim-2; j>=0; j--) {
          idx += bvalue[j]*baseldC[j+1];
          if (((i+1)%bunit[j]) == 0) bvalue[j]++;
          if (bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
        }
        for (j=0; j<(hiC[ndim-1]-loC[ndim-1]+1); j++) {
          cptr[idx+j] = talpha*aptr[idx+j]+tbeta*bptr[idx+j];
        }
      }
    }

  /**
   * Utility function to accumulate one patch into another
   * @param alpha scale factor for patch
   * @param ndim dimension of array
   * @param loC, hiC lower and upper bounding indices of patch
   * @param ldC array of strides for patch
   * @param A_ptr, C_ptr pointer to individual chunks of data
   */
  template <typename _type>
    void acc_patch_values(void *alpha, int64_t *loC, int64_t *hiC,
        int64_t *ldC, void *A_ptr, void *C_ptr)
    {
      int64_t bvalue[MAXDIM], bunit[MAXDIM], baseldC[MAXDIM];
      int64_t idx, n1dim;
      int64_t i, j;
      int ndim = p_ndim;
      _type talpha = *(reinterpret_cast<_type*>(alpha));
      _type *aptr = reinterpret_cast<_type*>(A_ptr);
      _type *cptr = reinterpret_cast<_type*>(C_ptr);
      /* compute "local" add */

      /* number of n-element of the first dimension */
      n1dim = 1; for(i=ndim-2; i>=0; i--) n1dim *= (hiC[i] - loC[i] + 1);

      /* calculate the destination indices */
      bvalue[ndim-1] = 0;
      bunit[ndim-1] = 1;
      if (ndim > 1) {
        bvalue[ndim-2] = 0;
        bunit[ndim-2] = 1;
      }
      /* baseld[ndim-2] = ld[ndim-1]
       * baseld[ndim-3] = ld[ndim-1] * ld[ndim-2]
       * baseld[ndim-4] = ld[ndim-1] * ld[ndim-2] * ld[ndim-3] .....
       */
      baseldC[ndim-1] = ldC[ndim-1]; 
      if (ndim > 1) {
        baseldC[ndim-2] = baseldC[ndim-1] *ldC[ndim-2];
      }
      for(i=ndim-3; i>=0; i--) {
        bvalue[i] = 0;
        bunit[i] = bunit[i+1] * (hiC[i+1] - loC[i+1] + 1);
        baseldC[i] = baseldC[i+1] * ldC[i];
      }
      for (i=0; i<n1dim; i++) {
        idx = 0;
        for (j=ndim-2; j>=0; j--) {
          idx += bvalue[j]*baseldC[j+1];
          if (((i+1)%bunit[j]) == 0) bvalue[j]++;
          if (bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
        }
        for (j=0; j<(hiC[ndim-1]-loC[ndim-1]+1); j++) {
          cptr[idx+j] += talpha*aptr[idx+j];
        }
      }
    }

  /**
   * Utility function to scale values of a patch
   * @param value scale factor for patch
   * @param lo, hi lower and upper bounding indices of patch
   * @param ld array of strides for patch
   * @param ptr pointer to individual chunks of data
   */
  template <typename _type> void scale_patch_values(void *value,
      int64_t *lo, int64_t *hi, int64_t *ld, void *ptr)
  {
    int64_t n1dim, i, j, idx;
    int64_t bvalue[MAXDIM], bunit[MAXDIM], baseld[MAXDIM];
    int ndim = p_ndim;
    /* number of n-element of the first dimension */
    n1dim = 1; for(i=ndim-2; i>=0; i--) n1dim *= (hi[i] - lo[i] + 1);

    bvalue[ndim-1] = 0;
    bunit[ndim-1] = 1;
    if (ndim > 1) {
      bvalue[ndim-2] = 0;
      bunit[ndim-2] = 1;
    }
    /* baseld[ndim-2] = ld[ndim-1]
     * baseld[ndim-3] = ld[ndim-1] * ld[ndim-2]
     * baseld[ndim-4] = ld[ndim-1] * ld[ndim-2] * ld[ndim-3] .....
     */
    baseld[ndim-1] = ld[ndim-1]; 
    if (ndim > 1) {
      baseld[ndim-2] = baseld[ndim-1] *ld[ndim-2];
    }
    for(i=ndim-3; i>=0; i--) {
      bvalue[i] = 0;
      bunit[i] = bunit[i+1] * (hi[i+1] - lo[i+1] + 1);
      baseld[i] = baseld[i+1] * ld[i];
    }

    /* scale local part of array */
    for(i=0; i<n1dim; i++) {
      idx = 0;
      for(j=1; j<ndim; j++) {
        idx += bvalue[j] * baseld[j-1];
        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
        if(bvalue[j] > (hi[j]-lo[j])) bvalue[j] = 0;
      }

      for(j=0; j<(hi[0]-lo[0]+1); j++)
        (reinterpret_cast<_type*>(ptr))[idx+j]  *=
          *reinterpret_cast<_type*>(value);
    }
  }

  /**
   * Wrapper for error function in environment class
   * @param msg message to print with error
   * @param code error code to exit with
   */
  void error(const char *msg, int code);

  /**
   * Utility function to convert XGA datatype into an actual size
   * @param type XGA datatype
   */
  int xga_sizeof(int type);

  /**
   * Utility function to allocate n XGA datatype elements
   * @param n number of elements
   * @param type XGA datatype
   * @return pointer to allocated data
   */
  void* xga_malloc(int64_t n, int type);

  /**
   * Utility function to free memory allocated by xga_malloc
   * @param ptr void pointer allocated by xga_malloc
   * @param type XGA datatype
   */
  void xga_free(void *ptr, int type);

  template <typename _type>
  friend class GlobalArray;

private:

  xga_types p_datatype = XGA_UNKNOWN; /* data type */
  data_distribution p_distr;          /* data distribution */
  int     p_ndim;                     /* dimension of array */
  int64_t p_dims[MAXDIM];             /* dimensions of array */
  int64_t chunk[MAXDIM];              /* chunking array */
  int64_t *p_mapc;                    /* block distribution map */
  std::vector<int64_t> map;           /* distribution map for iterator */
  int     nproc;                      /* number of processors */
  std::vector<int> proclist;          /* list of procs containing data */
  int     p_proc_grid[MAXDIM];        /* processor array */
  double  p_scale[MAXDIM];            /* nblock/dim (precomputed) */
  int64_t p_size;                     /* size of local data, in bytes */
  int64_t p_elemsize;                 /* size of data element */
  bool    ghosts;                     /* flag indicate ghost cells */
  int64_t width[MAXDIM];              /* boundary cells per dimension */
  int64_t p_lo[MAXDIM];               /* lower indices of local block */
  void    **ptr;                      /* array of pointers to remoted data */
  bool    p_active;                   /* data has been allocated to array */


  /* iterator parameters */
  int64_t it_lo[MAXDIM];        /* lower corner of block in array */
  int64_t it_hi[MAXDIM];        /* upper corner of block in array */
  int64_t count;                /* counter to keep track of blocks */
  int64_t p_offset;             /* offset to start of block in data segment */
  int     p_iblock;             /* counter tracking blocks on processor */
  int64_t lobuf[MAXDIM];        /* lower corner of sub-block */
  int64_t hibuf[MAXDIM];        /* upper corner of sub-block */
  int64_t num_blks[MAXDIM];     /* number of blocks in each dimension */
  int64_t blk_num[MAXDIM];      /* number of whole blocks in each direction */
  int64_t blk_dims[MAXDIM];     /* maximum dimensions of block */
  int64_t blk_inc[MAXDIM];      /* dimensions of partial blocks */
  int64_t blk_ld[MAXDIM];       /* stride between blocks */
  int64_t hlf_blk[MAXDIM];      /* flag indicating whether or not to add
                                 * an extra block or partial block to
                                 * process in this dimension */
  int     p_np;                 /* number of processors with data */
  int64_t blk_size[MAXDIM];     /* blk_dims*p_proc_grid */
  int64_t blk_ngrd[MAXDIM];     /* number of whole blocks of size blk_size */
  int     proc_index[MAXDIM];   /* location of processor in proc grid */
  int     index[MAXDIM];        /* location of current sub-block */
  int64_t block_total;          /* total number of blocks in array */

  Environment *p_env;
  Group *p_group;
  CMX::Allocation *p_alloc;

  int64_t block_count;
};
}
#endif
