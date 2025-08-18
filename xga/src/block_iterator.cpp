#include "xga_private.hpp"

#define XGA_GETBLOCKPATCH_M(plo,phi,lo,hi,blo,bhi,ndim) {                  \
  int _d;                                                                  \
  for (_d=0; _d<ndim; _d++) {                                              \
    if (lo[_d] <= phi[_d] && lo[_d] >= plo[_d]) blo[_d] = lo[_d];          \
    else blo[_d] = plo[_d];                                                \
    if (hi[_d] <= phi[_d] && hi[_d] >= plo[_d]) bhi[_d] = hi[_d];          \
    else bhi[_d] = phi[_d];                                                \
  }                                                                        \
}

namespace XGA {
  /**
   * Initialize block iteration
   * @param[in] lo,hi bounding indices of requested patch in global array
   */
void p_GA::initIterator(const int64_t *lo, const int64_t *hi)
{
  int i;
  count = 0;
  for (i=0; i<p_ndim; i++) {
    it_lo[i] = lo[i];
    it_hi[i] = hi[i];
  }
  /* Standard GA distribution */
  if (p_distr == REGULAR) {
    /* Locate the processors containing some portion of the patch
     * specified by lo and hi and return the results in map,
     * proclist, and np. proclist contains a list of processors
     * containing some portion of the patch, map contains
     * the lower and upper indices of the portion of the patch held
     * by a given processor, and np contains the total number of
     * processors that contain some portion of the patch.
     */
    if(!locateRegion(lo, hi, map, proclist, &nproc))
      XGA_REGIONERROR_M(p_ndim, lo, hi, -1);
  } else if (p_distr == SCALAPACK)  {
    /* XGA uses ScaLAPACK block cyclic data distribution */
    int j;
    /* Calculate some properties associated with data distribution */
    for (j=0; j<p_ndim; j++)  {
      blk_dims[j] = blk_size[j]*nblock[j];
      blk_num[j] = p_dims[j]/blk_dims[j];
      blk_inc[j] = p_dims[j]-blk_num[j]*blk_dims[j];
      blk_ld[j] = blk_num[j]*blk_size[j];
      hlf_blk[j] = blk_inc[j]/blk_size[j];
    }
    iblock = 0;
    offset = 0;
    /* Initialize proc_index and index arrays */
    XGA_FIND_PROC_INDICES_M(iblock, proc_index);
    XGA_FIND_PROC_INDICES_M(iblock, index);
  } else if (p_distr == TILED || p_distr == TILED_IRREG)  {
    iblock = 0;
    offset = 0;
    /* Initialize proc_index and index arrays */
    XGA_FIND_TILE_PROC_INDICES_M(iblock, proc_index);
    XGA_FIND_TILE_PROC_INDICES_M(iblock, index);
  }
}

/**
 * Reset an iterator back to the start
 */
void p_GA::resetIterator()
{
  if (p_distr == REGULAR) {
    /* Regular data distribution */
    count = 0;
  } else if (p_distr == SCALAPACK) {
    iblock = 0;
    offset = 0;
    /* Initialize proc_index and index arrays */
    XGA_FIND_PROC_INDICES_M(iblock, proc_index);
    XGA_FIND_PROC_INDICES_M(iblock, index);
  } else if (p_distr == TILED || p_distr == TILED_IRREG)  {
    iblock = 0;
    offset = 0;
    /* Initialize proc_index and index arrays */
    XGA_FIND_TILE_PROC_INDICES_M(iblock, proc_index);
    XGA_FIND_TILE_PROC_INDICES_M(iblock, index);
  }
}

#define XGA_GETRANGEFROMMAP_M(_p, _ndim, _plo, _phi){  \
  int64_t   _mloc = _p* _ndim *2;                      \
  *_plo  = &map[_mloc];                                \
  *_phi  = *_plo + _ndim;                              \
}

/**
 * Get the next sub-block from the larger block defined when the iterator was
 * initialized
 * @param proc processor on which the next block resides
 * @param plo indices for lower corner of remote block
 * @param phi indices for upper corner of remote block
 * @param prem pointer to remote buffer
 * @return returns false if there is no new block, true otherwise
 */
bool p_GA::nextBlock(int *proc, int64_t *plo[],
    int64_t *phi[], char **prem, int64_t ldrem[])
{
  int64_t idx, i, p;
  bool ok;
  if (p_distr == REGULAR) {
    int64_t *blo, *bhi;
    int64_t nelems;
    idx = count;

    /* Check to see if range is valid (it may not be valid if user has
     * created an irregular distribution in which some processors do not have
     * data). If invalid, skip this block and go to the next one
     */
    ok = false;
    while(!ok) {
      /* no blocks left, so return */
      if (count>=nproc) return false;
      *proc = (int)proclist[idx];
      *proc = (int)p_group->getLocalRank(*proc);
      /* Find  visible portion of patch held by processor p and
       * return the result in plo and phi. Also get actual processor
       * index corresponding to p and store the result in proc.
       */
      XGA_GETRANGEFROMMAP_M(count, p_ndim, plo, phi);
      ok = true;
      for (i=0; i<p_ndim; i++) {
        if ((*phi)[i]<(*plo)[i]) {
          ok = false;
          break;
        }
      }
      if (ok) {
        *proc = proclist[idx];
        blo = *plo;
        bhi = *phi;

        XGA_LOCATION_M(*proc, blo, prem, ldrem);
        *proc = proclist[idx];
        *proc = (int)p_group->getLocalRank(*proc);
      }
      count++;
      idx = count;
    }
    return true;
  } else {
    int64_t offset, l_offset, last, pinv;
    int64_t blk_tot = block_total;
    int64_t blo[MAXDIM], bhi[MAXDIM];
    int64_t idx, j, jtot, iproc;
    bool chk, check1, check2;
    if (p_distr == SCALAPACK ||
        p_distr == TILED ||
        p_distr == TILED_IRREG) {
      /* Scalapack-type data distribution */
      int64_t proc_index[MAXDIM], index[MAXDIM];
      int64_t itmp;
      int64_t blk_jinc;
      /* Return false at the end of the iteration */
      if (iblock >= nproc) return false;
      chk = false;
      /* loop over blocks until a block with data is found */
      while (!chk) {
        /* get bounds for current block */
        if (p_distr == SCALAPACK || p_distr == TILED) {
          for (j = 0; j < p_ndim; j++) {
            blo[j] = blk_size[j]*(index[j])+1;
            bhi[j] = blk_size[j]*(index[j]+1);
            if (bhi[j] > p_dims[j]) bhi[j] = p_dims[j];
          }
        } else {
          offset = 0;
          for (j = 0; j < p_ndim; j++) {
            blo[j] = p_mapc[offset+index[j]];
            if (index[j] == blk_num[j]-1) {
              bhi[j] = p_dims[j];
            } else {
              bhi[j] = p_mapc[offset+index[j]+1]-1;
            }
            offset += nblock[j];
          }
        }
        /* check to see if this block overlaps with requested block
         * defined by lo and hi */
        chk = true;
        for (j=0; j<p_ndim; j++) {
          /* check to see if at least one end point of the interval
           * represented by blo and bhi falls in the interval
           * represented by lo and hi */
          check1 = ((blo[j] >= it_lo[j] && blo[j] <= it_hi[j]) ||
              (bhi[j] >= it_lo[j] && bhi[j] <= it_hi[j]));
          /* check to see if interval represented by lo and hi
           * falls entirely within interval represented by blo and bhi */
          check2 = ((it_lo[j] >= blo[j] && it_lo[j] <= bhi[j]) &&
              (it_hi[j] >= blo[j] && it_hi[j] <= bhi[j]));
          /* If there is some data, move to the next section of code,
           * otherwise, check next block */
          if (!check1 && !check2) {
            chk = false;
          }
        }
        
        if (!chk) {
          /* update offset for block */
          itmp = 1;
          for (j=0; j<p_ndim; j++) {
            itmp *= bhi[j]-blo[j]+1;
          }
          offset += itmp;

          /* increment to next block */
          index[0] += nblock[0];
          for (j=0; j<p_ndim; j++) {
            if (index[j] >= blk_num[j] && j < p_ndim-1) {
              index[j] = proc_index[j];
              index[j+1] += nblock[j+1];
            }
          }
          if (index[p_ndim-1] >= blk_num[p_ndim-1]) {
            /* last iteration has been completed on current processor. Go
             * to next processor */
            iblock++;
            if (iblock >= nproc) return false;
            offset = 0;
            if (p_distr == TILED || p_distr == TILED_IRREG) {
              XGA_FIND_TILE_PROC_INDICES_M(iblock, proc_index);
              XGA_FIND_TILE_PROC_INDICES_M(iblock, index);
            } else if (p_distr == SCALAPACK) {
              XGA_FIND_PROC_INDICES_M(iblock, proc_index);
              XGA_FIND_PROC_INDICES_M(iblock, index);
            }
          }
        }
      }
      if (chk) {
        int64_t *clo, *chi;
        *plo = lobuf;
        *phi = hibuf;
        clo = *plo;
        chi = *phi;
        /* get the patch of block that overlaps requested region */
        XGA_GETBLOCKPATCH_M(blo,bhi,it_lo,it_hi,clo,chi,p_ndim);

        /* evaluate offset within block */
        last = p_ndim - 1;
        if (p_distr == TILED || p_distr == TILED_IRREG) {
          jtot = 1;
          if (last == 0) ldrem[0] = bhi[0] - blo[0] + 1;
          l_offset = 0;
          for (j=0; j<last; j++) {
            l_offset += (clo[j]-blo[j])*jtot;
            ldrem[j] = bhi[j]-blo[j]+1;
            jtot *= ldrem[j];
          }
          l_offset += (clo[last]-blo[last])*jtot;
          l_offset += offset;
        } else if (p_distr == SCALAPACK) {
          l_offset = 0;
          jtot = 1;
          for (j=0; j<last; j++)  {
            ldrem[j] = blk_ld[j];
            blk_jinc = p_dims[j]%blk_size[j];
            if (blk_inc[j] > 0) {
              if (proc_index[j]<hlf_blk[j]) {
                blk_jinc = blk_size[j];
              } else if (proc_index[j] == hlf_blk[j]) {
                blk_jinc = blk_inc[j]%blk_size[j];
              } else {
                blk_jinc = 0;
              }
            }
            ldrem[j] += blk_jinc;
            l_offset += (clo[j]-blo[j]
                + ((blo[j]-1)/blk_dims[j])*blk_size[j])*jtot;
            jtot *= ldrem[j];
          }
          l_offset += (clo[last]-blo[last]
              + ((blo[last]-1)/blk_dims[j])*blk_size[last])*jtot;
        }
        /* get pointer to data on remote block */
        pinv = (iblock)%nproc;
        pinv = p_group->getLocalRank(pinv);
        *prem =  static_cast<char*>(ptr[pinv])+l_offset*p_elemsize;
        *proc = pinv;

        /* evaluate new offset for block */
        itmp = 1;
        for (j=0; j<p_ndim; j++) {
          itmp *= bhi[j]-blo[j]+1;
        }
        offset += itmp;
        /* increment to next block */
        index[0] += nblock[0];
        for (j=0; j<p_ndim; j++) {
          if (index[j] >= blk_num[j] && j < p_ndim-1) {
            index[j] = proc_index[j];
            index[j+1] += nblock[j+1];
          }
        }
        if (index[p_ndim-1] >= blk_num[p_ndim-1]) {
          iblock++;
          offset = 0;
          if (p_distr == TILED || p_distr == TILED_IRREG) {
            XGA_FIND_TILE_PROC_INDICES_M(iblock, proc_index);
            XGA_FIND_TILE_PROC_INDICES_M(iblock, index);
          } else if (p_distr == SCALAPACK) {
            XGA_FIND_PROC_INDICES_M(iblock, proc_index);
            XGA_FIND_PROC_INDICES_M(iblock, index);
          }
        }
      }
    }
    return true;
  }
  return false;
}

#undef XGA_GETRANGEFROMAP_M

/**
 * Check if this is the last block
 * @return true if this is the last block
 */
bool p_GA::lastBlock()
{
  int64_t idx;
  if (p_distr == REGULAR) {
    idx = count;
    /* no blocks left after this iteration */
    if (idx>=nproc) return true;
  } else {
    if (p_distr == SCALAPACK ||
        p_distr == TILED ||
        p_distr == TILED_IRREG) {
      if (iblock >= nproc) return true;
    }
  }
  return false;
}

/**
 * Clean up iterator
 */
void p_GA::destroyIterator()
{
    map.clear();
    proclist.clear();
}

/**
 * Functions that iterate over locally held blocks in a global array. These
 * are used in some routines such as copy
 */

/**
 * Initialize a local iterator
 */
void p_GA::localInit()
{
  count = 0;
  int64_t me = p_group->rank();
  /* If standard GA distribution then no additional action needs to be taken */
  if (p_distr == SCALAPACK) {
    /* GA uses ScaLAPACK block cyclic data distribution */
    /* Initialize proc_index and index arrays */
    XGA_FIND_PROC_INDICES_M(me, proc_index);
    XGA_FIND_PROC_INDICES_M(me, index);
  } else if (p_distr == TILED) {
    /* GA uses tiled distribution */
    /* Initialize proc_index and index arrays */
    XGA_FIND_TILE_PROC_INDICES_M(me, proc_index);
    XGA_FIND_TILE_PROC_INDICES_M(me, index);
  } else if (p_distr == TILED_IRREG) {
    /* GA uses irregular tiled distribution */
    /* Initialize proc_index and index arrays */
    XGA_FIND_TILE_PROC_INDICES_M(me, proc_index);
    XGA_FIND_TILE_PROC_INDICES_M(me, index);
  }
}

/**
 * Get the next sub-block from local portion of global array
 * @param plo indices for lower corner of block
 * @param phi indices for upper corner of block
 * @param ptr pointer to local buffer
 * @param ld array of strides for local block
 * @return returns false if there is no new block, true otherwise
 */
bool p_GA::nextLocalBlock(int64_t plo[], int64_t phi[],
    char **ptr, int64_t ld[])
{
  int64_t i;
  int me = p_group->rank();
  if (p_distr == REGULAR) {
    int64_t nelems;
    /* no blocks left, so return */
    if (count>0) return 0;

    /* Find  visible portion of patch held by this processor and
     * return the result in plo and phi. Return pointer to local
     * data as well
     */
    distribution(me, plo, phi);
    /* Check to see if this process has any data. Return 0 if
     * it does not */
    for (i=0; i<p_ndim; i++) {
      if (phi[i]<plo[i]) return 0;
    }
    void *vptr;
    accessPtr(plo,phi,&vptr,ld);
    *ptr = static_cast<char*>(vptr);
    count++;
  } else if (p_distr == SCALAPACK || p_distr == TILED) {
    /* Scalapack-type data distribution */
    if (index[p_ndim-1] >= blk_num[p_ndim-1]) return 0;
    /* Find coordinates of bounding block */
    for (i=0; i<p_ndim; i++) {
      plo[i] = index[i]*blk_size[i]+1;
      phi[i] = (index[i]+1)*blk_size[i];
      if (phi[i] > blk_dims[i]) phi[i] = blk_dims[i];
    }
    void *vptr;
    accessBlockGridPtr(index,&vptr,ld);
    *ptr = static_cast<char*>(vptr);
    index[0] += blk_inc[0];
    for (i=0; i<p_ndim; i++) {
      if (index[i] >= blk_num[i] && i<p_ndim-1) {
        index[i] = proc_index[i];
        index[i+1] += blk_inc[i+1];
      }
    }
  } else if (p_distr == TILED_IRREG) {
    /* Irregular tiled data distribution */
    int64_t t_offset = 0;
    if (index[p_ndim-1] >= blk_num[p_ndim-1]) return 0;
    /* Find coordinates of bounding block */
    for (i=0; i<p_ndim; i++) {
      plo[i] = p_mapc[t_offset+index[i]];
      if (index[i] < blk_num[i]-1) {
        phi[i] = p_mapc[t_offset+index[i]+1]-1;
      } else {
        phi[i] = p_dims[i];
      }
      t_offset += blk_num[i];
    }
    void *vptr;
    accessBlockGridPtr(index,&vptr,ld);
    *ptr = static_cast<char*>(vptr);
    index[0] += blk_inc[0];
    for (i=0; i<p_ndim; i++) {
      if (index[i] >= blk_num[i] && i<p_ndim-1) {
        index[i] = proc_index[i];
        index[i+1] += blk_inc[i+1];
      }
    }
  }
  return 1;
}
}
