#include "xga_private.hpp"

#define XGA_GETBLOCKPATCH_M(_plo,_phi,_lo,_hi,_blo,_bhi,ndim) {           \
  int _d;                                                                  \
  for (_d=0; _d<ndim; _d++) {                                              \
    if (_lo[_d] <= _phi[_d] && _lo[_d] >= _plo[_d]) _blo[_d] = _lo[_d];    \
    else _blo[_d] = _plo[_d];                                              \
    if (_hi[_d] <= _phi[_d] && _hi[_d] >= _plo[_d]) _bhi[_d] = _hi[_d];    \
    else _bhi[_d] = _phi[_d];                                              \
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
//  printf("p[%d] (initIterator) lo[%ld:%ld:%ld] hi[%ld:%ld:%ld]\n",
//      p_group->rank(),lo[0],lo[1],lo[2],hi[0],hi[1],hi[2]);
    for (i=0; i<p_ndim; i++)  {
      blk_size[i] = blk_dims[i]*p_proc_grid[i];
      blk_num[i] = p_dims[i]/blk_size[i];
      blk_inc[i] = p_dims[i]-blk_num[i]*blk_size[i];
      blk_ld[i] = blk_num[i]*blk_dims[i];
      hlf_blk[i] = blk_inc[i]/blk_dims[i];
    }
    /* Need to check all remote processors so start at 0 */
    p_iblock = 0;
    p_offset = 0;
    /* Initialize proc_index and index arrays */
    XGA_FIND_PROC_INDICES_M(p_iblock, proc_index);
    XGA_FIND_PROC_INDICES_M(p_iblock, index);
  } else if (p_distr == TILED || p_distr == TILED_IRREG)  {
    p_iblock = 0;
    p_offset = 0;
    /* Initialize proc_index and index arrays */
    XGA_FIND_TILE_PROC_INDICES_M(p_iblock, proc_index);
    XGA_FIND_TILE_PROC_INDICES_M(p_iblock, index);
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
    p_iblock = 0;
    p_offset = 0;
    /* Initialize proc_index and index arrays */
    XGA_FIND_PROC_INDICES_M(p_iblock, proc_index);
    XGA_FIND_PROC_INDICES_M(p_iblock, index);
  } else if (p_distr == TILED || p_distr == TILED_IRREG)  {
    p_iblock = 0;
    p_offset = 0;
    /* Initialize proc_index and index arrays */
    XGA_FIND_TILE_PROC_INDICES_M(p_iblock, proc_index);
    XGA_FIND_TILE_PROC_INDICES_M(p_iblock, index);
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
 * @param ldrem array of strides on remote buffer
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
      *proc = static_cast<int>(proclist[idx]);
      *proc = static_cast<int>(p_group->getLocalRank(*proc));
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
    int64_t l_offset, last, pinv;
    int64_t blk_tot = block_total;
    int64_t blo[MAXDIM], bhi[MAXDIM];
    int64_t idx, j, jtot, iproc;
    bool chk, check1, check2;
    if (p_distr == SCALAPACK ||
        p_distr == TILED ||
        p_distr == TILED_IRREG) {
      /* Scalapack-type data distribution */
      int64_t blk_jinc;
      /* Return false at the end of the iteration */
      if (p_iblock >= nproc) {
        return false;
      }
      chk = false;
      /* loop over blocks until a block with data is found */
//          printf("p[%d] (nextBlock) index[%ld:%ld:%ld] prem: %d\n",
//              p_group->rank(),index[0],index[1],index[2],p_iblock);
      while (!chk) {
        /* get bounds for current block */
        if (p_distr == SCALAPACK || p_distr == TILED) {
//          printf("p[%d] (nextBlock) blk_dims[%ld:%ld:%ld]\n",p_group->rank(),
//              blk_dims[0],blk_dims[1],blk_dims[2]);
          for (j = 0; j < p_ndim; j++) {
            blo[j] = blk_dims[j]*(index[j]);
            bhi[j] = blk_dims[j]*(index[j]+1)-1;
            if (bhi[j] >= p_dims[j]) bhi[j] = p_dims[j]-1;
          }
//          printf("p[%d] (nextBlock) blo[%ld:%ld:%ld] bhi[%ld:%ld:%ld]\n",
//              p_group->rank(),blo[0],blo[1],blo[2],bhi[0],bhi[1],bhi[2]);
        } else {
          p_offset = 0;
          for (j = 0; j < p_ndim; j++) {
            blo[j] = p_mapc[p_offset+index[j]];
            if (index[j] == blk_num[j]-1) {
              bhi[j] = p_dims[j];
            } else {
              bhi[j] = p_mapc[p_offset+index[j]+1]-1;
            }
            p_offset += p_proc_grid[j];
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
          /* This block has no data that overlaps with the
           * requested region */
          int64_t itmp = 1;
          for (j=0; j<p_ndim; j++) {
            itmp *= bhi[j]-blo[j]+1;
          }
          p_offset += itmp;

          /* increment to next block */
          index[p_ndim-1] += p_proc_grid[p_ndim-1];
          for (j=p_ndim-1; j>=0; j--) {
            if (index[j] >= num_blks[j] && j > 0) {
              index[j] = proc_index[j];
              index[j-1] += p_proc_grid[j-1];
            }
          }
          if (index[0] >= num_blks[0]) {
            /* last iteration has been completed on current processor. Go
             * to next processor */
            p_iblock++;
            if (p_iblock >= nproc) {
              return false;
            }
            p_offset = 0;
            if (p_distr == TILED || p_distr == TILED_IRREG) {
              XGA_FIND_TILE_PROC_INDICES_M(p_iblock, proc_index);
              XGA_FIND_TILE_PROC_INDICES_M(p_iblock, index);
            } else if (p_distr == SCALAPACK) {
              XGA_FIND_PROC_INDICES_M(p_iblock, proc_index);
              XGA_FIND_PROC_INDICES_M(p_iblock, index);
            }
          }
        }
      }
      if (chk) {
        int64_t clo[MAXDIM], chi[MAXDIM];
        *plo = lobuf;
        *phi = hibuf;
        /* get the patch of block that overlaps requested region */
        XGA_GETBLOCKPATCH_M(blo,bhi,it_lo,it_hi,clo,chi,p_ndim);
        for (i=0; i<p_ndim; i++) {
          (*plo)[i] = clo[i];
          (*phi)[i] = chi[i];
        }

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
          l_offset += p_offset;
        } else if (p_distr == SCALAPACK) {
          l_offset = 0;
          jtot = 1;
          for (j=last; j>0; j--)  {
            ldrem[j-1] = blk_ld[j];
            /* initialize this so that it works if first block is partial
             * block */
            blk_jinc = p_dims[j]%blk_dims[j];
//            printf("p[%d] j: %ld blk_jinc: %ld ldrem: %ld\n",
//                p_group->rank(),j,blk_jinc,ldrem[j-1]);
            if (blk_inc[j] > 0) {
              /* may need to add an extra block or a partial block to stride */
              if (proc_index[j]<hlf_blk[j]) {
                /* add a full block */
                blk_jinc = blk_dims[j];
              } else if (proc_index[j] == hlf_blk[j]) {
                /* add a partial block */
                blk_jinc = blk_inc[j]%blk_dims[j];
              } else {
                /* add nothing */
                blk_jinc = 0;
              }
            }
//            printf("p[%d] j: %ld blk_inc: %ld proc_index: %ld hlf_blk: %ld dims: %ld\n",
//                p_group->rank(),j,blk_inc[j],proc_index[j],hlf_blk[j],blk_dims[j]);
//            printf("p[%d] j: %ld blk_jinc: %ld\n",p_group->rank(),j,blk_jinc);
            ldrem[j-1] += blk_jinc;
            l_offset += (clo[j]-blo[j]
                + ((blo[j])/blk_size[j])*blk_dims[j])*jtot;
//            printf("p[%d]   j: %d clo: %ld blo: %ld size: %ld dims: %ld offset: %ld\n",
//                p_group->rank(),j,clo[j],blo[j],blk_size[j],blk_dims[j],l_offset);
            jtot *= ldrem[j-1];
          }
          l_offset += (clo[0]-blo[0]
              + ((blo[0])/blk_size[0])*blk_dims[0])*jtot;
//            printf("p[%d]   j: 0 clo: %ld blo: %ld size: %ld dims: %ld offset: %ld\n",
//                p_group->rank(),clo[0],blo[0],blk_size[0],blk_dims[0],l_offset);
        }
        /* get pointer to data on remote block */
        pinv = (p_iblock)%nproc;
        //pinv = p_group->getLocalRank(pinv);
        *prem =  static_cast<char*>(ptr[pinv])+l_offset*p_elemsize;
        *proc = pinv;
//          printf("p[%d] (nextBlock) ldrem: [%ld:%ld] proc: %d\n",p_group->rank(),
//              ldrem[0],ldrem[1],pinv);

        /* evaluate new offset for block */
        int64_t itmp = 1;
        for (j=0; j<p_ndim; j++) {
          itmp *= bhi[j]-blo[j]+1;
        }
        p_offset += itmp;
        /* increment to next block */
        index[p_ndim-1] += p_proc_grid[p_ndim-1];
        for (j=p_ndim-1; j>=0; j--) {
          if (index[j] >= num_blks[j] && j > 0) {
            index[j] = proc_index[j];
            index[j-1] += p_proc_grid[j-1];
          }
        }
        if (index[0] >= num_blks[0]) {
          /* last iteration has been completed on current processor. Go
           * to next processor */
          p_iblock++;
//          if (p_iblock >= nproc) return false;
          p_offset = 0;
          if (p_distr == TILED || p_distr == TILED_IRREG) {
            XGA_FIND_TILE_PROC_INDICES_M(p_iblock, proc_index);
            XGA_FIND_TILE_PROC_INDICES_M(p_iblock, index);
          } else if (p_distr == SCALAPACK) {
            XGA_FIND_PROC_INDICES_M(p_iblock, proc_index);
            XGA_FIND_PROC_INDICES_M(p_iblock, index);
          }
        }
      }
//      printf("p[%d] (nextBlock) plo[%ld:%ld:%ld] phi[%ld:%ld:%ld] prem: %d offset: %ld\n",
//          p_group->rank(),lobuf[0],lobuf[1],lobuf[2],
//          hibuf[0],hibuf[1],hibuf[2],*proc,l_offset);
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
      if (p_iblock >= nproc) return true;
    }
  }
  return false;
}

/**
 * Clean up iterator
 */
void p_GA::destroyIterator()
{
  if (p_distr == REGULAR) {
    map.clear();
    proclist.clear();
  }
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
