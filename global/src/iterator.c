/* $Id: iterator.c,v 1.80.2.18 2007/12/18 22:22:27 d3g293 Exp $ */
/* 
 * module: iterator.c
 * author: Jarek Nieplocha
 * description: implements an iterator that can be used for looping over blocks
 *              in a GA operation. This functionality is designed to hide
 *              the details of the data layout from the operation
 * 
 * DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */
#if HAVE_CONFIG_H
#   include "config.h"
#endif
 
/*#define PERMUTE_PIDS */

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#if HAVE_STDINT_H
#   include <stdint.h>
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif
#if HAVE_ASSERT_H
#   include <assert.h>
#endif
#if HAVE_STDDEF_H
#include <stddef.h>
#endif

#include "global.h"
#include "globalp.h"
#include "base.h"
#include "armci.h"
#include "macdecls.h"
#include "ga-papi.h"
#include "ga-wapi.h"
#include "thread-safe.h"

#define DEBUG 0
#ifdef PROFILE_OLD
#include "ga_profile.h"
#endif



/*\ prepare permuted list of processes for remote ops
\*/
#define gaPermuteProcList(nproc)                                      \
{                                                                     \
  if((nproc) ==1) ProcListPerm[0]=0;                                  \
  else{                                                               \
    int _i, iswap, temp;                                              \
    if((nproc) > GAnproc) pnga_error("permute_proc: error ", (nproc));\
                                                                      \
    /* every process generates different random sequence */           \
    (void)srand((unsigned)GAme);                                      \
                                                                      \
    /* initialize list */                                             \
    for(_i=0; _i< (nproc); _i++) ProcListPerm[_i]=_i;                 \
                                                                      \
    /* list permutation generated by random swapping */               \
    for(_i=0; _i< (nproc); _i++){                                     \
      iswap = (int)(rand() % (nproc));                                \
      temp = ProcListPerm[iswap];                                     \
      ProcListPerm[iswap] = ProcListPerm[_i];                         \
      ProcListPerm[_i] = temp;                                        \
    }                                                                 \
  }                                                                   \
}

#define gam_GetRangeFromMap(p, ndim, plo, phi){  \
  Integer   _mloc = p* ndim *2;                  \
            *plo  = (Integer*)_ga_map + _mloc;   \
            *phi  = *plo + ndim;                 \
}

/*\ Return pointer (ptr_loc) to location in memory of element with subscripts
 *  (subscript). Also return physical dimensions of array in memory in ld.
\*/
#define gam_Location(proc, g_handle,  subscript, ptr_loc, ld)              \
{                                                                          \
  Integer _offset=0, _d, _w, _factor=1, _last=GA[g_handle].ndim-1;         \
  Integer _lo[MAXDIM], _hi[MAXDIM], _pinv, _p_handle;                      \
                                                                           \
  ga_ownsM(g_handle, proc, _lo, _hi);                                      \
  gaCheckSubscriptM(subscript, _lo, _hi, GA[g_handle].ndim);               \
  if(_last==0) ld[0]=_hi[0]- _lo[0]+1+2*(Integer)GA[g_handle].width[0];    \
  __CRAYX1_PRAGMA("_CRI shortloop");                                       \
  for(_d=0; _d < _last; _d++)            {                                 \
    _w = (Integer)GA[g_handle].width[_d];                                  \
    _offset += (subscript[_d]-_lo[_d]+_w) * _factor;                       \
    ld[_d] = _hi[_d] - _lo[_d] + 1 + 2*_w;                                 \
    _factor *= ld[_d];                                                     \
  }                                                                        \
  _offset += (subscript[_last]-_lo[_last]                                  \
      + (Integer)GA[g_handle].width[_last])                                \
  * _factor;                                                               \
  _p_handle = GA[g_handle].p_handle;                                       \
  if (_p_handle != 0) {                                                    \
    if (GA[g_handle].num_rstrctd == 0) {                                   \
      _pinv = proc;                                                        \
    } else {                                                               \
      _pinv = GA[g_handle].rstrctd_list[proc];                             \
    }                                                                      \
  } else {                                                                 \
    _pinv = PGRP_LIST[_p_handle].inv_map_proc_list[proc];                  \
  }                                                                        \
  *(ptr_loc) =  GA[g_handle].ptr[_pinv]+_offset*GA[g_handle].elemsize;     \
}

void gam_LocationF(int proc, Integer g_handle,  Integer subscript[],
    char **ptr_loc, Integer ld[])
{                                                                          
  Integer _offset=0, _d, _w, _factor=1, _last=GA[g_handle].ndim-1;         
  Integer _lo[MAXDIM], _hi[MAXDIM], _pinv, _p_handle;                     
                                                                           
  ga_ownsM(g_handle, proc, _lo, _hi);                                      
  gaCheckSubscriptM(subscript, _lo, _hi, GA[g_handle].ndim);               
  if(_last==0) ld[0]=_hi[0]- _lo[0]+1+2*(Integer)GA[g_handle].width[0];   
  __CRAYX1_PRAGMA("_CRI shortloop");                                      
  for(_d=0; _d < _last; _d++)            {                                 
    _w = (Integer)GA[g_handle].width[_d];                                  
    _offset += (subscript[_d]-_lo[_d]+_w) * _factor;                       
    ld[_d] = _hi[_d] - _lo[_d] + 1 + 2*_w;                                 
    _factor *= ld[_d];                                                     
  }                                                                        
  _offset += (subscript[_last]-_lo[_last]                                  
      + (Integer)GA[g_handle].width[_last])                                
  * _factor;                                                               
  _p_handle = GA[g_handle].p_handle;                                       
  if (_p_handle != 0) {                                                    
    if (GA[g_handle].num_rstrctd == 0) {                                   
      _pinv = proc;                                                        
    } else {                                                               
      _pinv = GA[g_handle].rstrctd_list[proc];                             
    }                                                                      
  } else {                                                                 
    _pinv = PGRP_LIST[_p_handle].inv_map_proc_list[proc];                  
  }                                                                       
  *(ptr_loc) =  GA[g_handle].ptr[_pinv]+_offset*GA[g_handle].elemsize;    
}

#define gam_GetBlockPatch(plo,phi,lo,hi,blo,bhi,ndim) {                    \
  Integer _d;                                                              \
  for (_d=0; _d<ndim; _d++) {                                              \
    if (lo[_d] <= phi[_d] && lo[_d] >= plo[_d]) blo[_d] = lo[_d];          \
    else blo[_d] = plo[_d];                                                \
    if (hi[_d] <= phi[_d] && hi[_d] >= plo[_d]) bhi[_d] = hi[_d];          \
    else bhi[_d] = phi[_d];                                                \
  }                                                                        \
}

/**
 * Initialize an iterator handle
 * @param g_a global array handle
 * @param lo indices for lower corner of block in global array
 * @param hi indices for upper corner of block in global array
 * @param hdl handle for iterator
 */
void gai_iterator_init(Integer g_a, Integer lo[], Integer hi[],
                       _iterator_hdl *hdl)
{
  Integer handle = GA_OFFSET + g_a;
  Integer ndim = GA[handle].ndim;
  Integer i;
  hdl->g_a = g_a;
  hdl->count = 0;
  /*
  hdl->map = (Integer*)malloc((size_t)(GAnproc*2*ndim+1)*sizeof(Integer));
  hdl->proclist = (Integer*)malloc((size_t)(GAnproc)*sizeof(Integer));
  */
  hdl->map = _ga_map;
  hdl->proclist = GA_proclist;
  for (i=0; i<ndim; i++) {
    hdl->lo[i] = lo[i];
    hdl->hi[i] = hi[i];
  }
  /* Standard GA distribution */
  if (!GA[handle].block_flag) {
    /* Locate the processors containing some portion of the patch
     * specified by lo and hi and return the results in _ga_map,
     * GA_proclist, and np. GA_proclist contains a list of processors
     * containing some portion of the patch, _ga_map contains
     * the lower and upper indices of the portion of the patch held
     * by a given processor, and np contains the total number of
     * processors that contain some portion of the patch.
     */
    if(!pnga_locate_region(g_a, lo, hi, hdl->map, hdl->proclist, &hdl->nproc ))
      ga_RegionError(pnga_ndim(g_a), lo, hi, g_a);

    gaPermuteProcList(hdl->nproc);
    /* Block-cyclic distribution */
  } else {
    if (GA[handle].block_sl_flag == 0) {
      /* GA uses simple block cyclic data distribution */
      hdl->iproc = 0;
      hdl->iblock = pnga_nodeid();
    } else {
      /* GA uses ScaLAPACK block cyclic data distribution */
      int *proc_gid;
      int j;
#if COMPACT_SCALAPACK
#else
      C_Integer *block_dims;
      /* Scalapack-type block-cyclic data distribution */
      /* Calculate some properties associated with data distribution */
      int *proc_grid = GA[handle].nblock;
      /*num_blocks = GA[handle].num_blocks;*/
      block_dims = GA[handle].block_dims;
      for (j=0; j<ndim; j++)  {
        hdl->blk_size[j] = block_dims[j];
        hdl->blk_dim[j] = block_dims[j]*proc_grid[j];
        hdl->blk_num[j] = GA[handle].dims[j]/hdl->blk_dim[j];
        hdl->blk_inc[j] = GA[handle].dims[j]-hdl->blk_num[j]*hdl->blk_dim[j];
        hdl->blk_ld[j] = hdl->blk_num[j]*block_dims[j];
        hdl->hlf_blk[j] = hdl->blk_inc[j]/block_dims[j];
      }
      hdl->iproc = 0;
      hdl->offset = 0;
      /* Initialize proc_index and index arrays */
      gam_find_proc_indices(handle, hdl->iproc, hdl->proc_index);
      gam_find_proc_indices(handle, hdl->iproc, hdl->index);
    }
  }
#endif
}

/**
 * Reset an iterator back to the start
 * @param hdl handle for iterator
 */
void gai_iterator_reset(_iterator_hdl *hdl)
{
  Integer handle = GA_OFFSET + hdl->g_a;
  if (!GA[handle].block_flag) {
    /* Regular data distribution */
    hdl->count = 0;
  } else {
    if (GA[handle].block_sl_flag == 0) {
      /* simple block cyclic data distribution */
      hdl->iproc = 0;
      hdl->iblock = 0;
      hdl->offset = 0;
    } else {
      hdl->iproc = 0;
      hdl->offset = 0;
      /* Initialize proc_index and index arrays */
      gam_find_proc_indices(handle, hdl->iproc, hdl->proc_index);
      gam_find_proc_indices(handle, hdl->iproc, hdl->index);
    }
  }
}

/**
 * Get the next sub-block from the larger block defined when the iterator was
 * initialized
 * @param hdl handle for iterator
 * @param proc processor on which the next block resides
 * @param plo indices for lower corner of remote block
 * @param phi indices for upper corner of remote block
 * @param prem pointer to remote buffer
 * @return returns false if there is no new block, true otherwise
 */
int gai_iterator_next(_iterator_hdl *hdl, Integer *proc, Integer *plo[],
    Integer *phi[], char **prem, Integer ldrem[])
{
  Integer idx, p;
  Integer handle = GA_OFFSET + hdl->g_a;
  Integer p_handle = GA[handle].p_handle;
  Integer n_rstrctd = GA[handle].num_rstrctd;
  Integer *rank_rstrctd = GA[handle].rank_rstrctd;
  int ndim;
  ndim = GA[handle].ndim;
  if (!GA[handle].block_flag) {
    Integer *blo, *bhi;
    idx = hdl->count;
    /* no blocks left, so return */
    if (idx>=hdl->nproc) return 0;
    p = (Integer)ProcListPerm[idx];
    *proc = (int)GA_proclist[p];
    if (p_handle >= 0) {
      *proc = PGRP_LIST[p_handle].inv_map_proc_list[*proc];
    }
#ifdef PERMUTE_PIDS
    if (GA_Proc_list) *proc = GA_inv_Proc_list[*proc];
#endif
    /* Find  visible portion of patch held by processor p and
     * return the result in plo and phi. Also get actual processor
     * index corresponding to p and store the result in proc.
     */
    gam_GetRangeFromMap(p, ndim, plo, phi);
    *proc = (int)GA_proclist[p];

    blo = *plo;
    if (n_rstrctd == 0) {
      gam_Location(*proc, handle, blo, prem, ldrem);
    } else {
      gam_Location(rank_rstrctd[*proc], handle, blo, prem, ldrem);
    }
    if (p_handle >= 0) {
      *proc = (int)GA_proclist[p];
      /* BJP */
      *proc = PGRP_LIST[p_handle].inv_map_proc_list[*proc];
    }
    hdl->count++;
    return 1;
  } else {
    Integer offset, l_offset, last, pinv;
    Integer blk_tot = GA[handle].block_total;
    Integer blo[MAXDIM], bhi[MAXDIM];
    Integer idx, j, jtot, chk, iproc;
    int check1, check2;
    if (GA[handle].block_sl_flag == 0) {
      /* Simple block-cyclic distribution */
      if (hdl->iproc >= GAnproc) return 0;
      /*if (hdl->iproc == GAnproc-1 && hdl->iblock >= blk_tot) return 0;*/
      if (hdl->iblock == hdl->iproc) hdl->offset = 0;
      chk = 0;
      /* loop over blocks until a block with data is found */
      while (!chk) {
        /* get the block corresponding to the current value of iblock */
        idx = hdl->iblock;
        ga_ownsM(handle,idx,blo,bhi);
        /* check to see if this block overlaps with requested block
         * defined by lo and hi */
        chk = 1;
        for (j=0; j<ndim; j++) {
          /* check to see if at least one end point of the interval
           * represented by blo and bhi falls in the interval
           * represented by lo and hi */
          check1 = ((blo[j] >= hdl->lo[j] && blo[j] <= hdl->hi[j]) ||
              (bhi[j] >= hdl->lo[j] && bhi[j] <= hdl->hi[j]));
          /* check to see if interval represented by lo and hi
           * falls entirely within interval represented by blo and bhi */
          check2 = ((hdl->lo[j] >= blo[j] && hdl->lo[j] <= bhi[j]) &&
              (hdl->hi[j] >= blo[j] && hdl->hi[j] <= bhi[j]));
          /* If there is some data, move to the next section of code,
           * otherwise, check next block */
          if (!check1 && !check2) {
            chk = 0;
          }
        }
        
        if (!chk) {
          /* evaluate new offset for block idx */
          jtot = 1;
          for (j=0; j<ndim; j++) {
            jtot *= bhi[j]-blo[j]+1;
          }
          hdl->offset += jtot;
          /* increment to next block */
          hdl->iblock += pnga_nnodes();
          if (hdl->iblock >= blk_tot) {
            hdl->offset = 0;
            hdl->iproc++;
            hdl->iblock = hdl->iproc;
            if (hdl->iproc >= GAnproc) return 0;
          }
        }
      }

      /* The block overlaps some data in lo,hi */
      if (chk) {
        Integer *clo, *chi;
        *plo = hdl->lobuf;
        *phi = hdl->hibuf;
        clo = *plo;
        chi = *phi;
        /* get the patch of block that overlaps requested region */
        gam_GetBlockPatch(blo,bhi,hdl->lo,hdl->hi,clo,chi,ndim);

        /* evaluate offset within block */
        last = ndim - 1;
        jtot = 1;
        if (last == 0) ldrem[0] = bhi[0] - blo[0] + 1;
        l_offset = 0;
        for (j=0; j<last; j++) {
          l_offset += (clo[j]-blo[j])*jtot;
          ldrem[j] = bhi[j]-blo[j]+1;
          jtot *= ldrem[j];
        }
        l_offset += (clo[last]-blo[last])*jtot;
        l_offset += hdl->offset;

        /* get pointer to data on remote block */
        pinv = idx%GAnproc;
        if (p_handle > 0) {
          pinv = PGRP_LIST[p_handle].inv_map_proc_list[pinv];
        }
        *prem =  GA[handle].ptr[pinv]+l_offset*GA[handle].elemsize;
        *proc = pinv;

        /* evaluate new offset for block idx */
        jtot = 1;
        for (j=0; j<ndim; j++) {
          jtot *= bhi[j]-blo[j]+1;
        }
        hdl->offset += jtot;

        hdl->iblock += pnga_nnodes();
        if (hdl->iblock >= blk_tot) {
          hdl->iproc++;
          hdl->iblock = hdl->iproc;
          hdl->offset = 0;
        }
      }
      return 1;
    } else {
      /* Scalapack-type data distribution */
      Integer proc_index[MAXDIM], index[MAXDIM];
      Integer itmp;
      Integer blk_jinc;
      /* Return false at the end of the iteration */
      if (hdl->iproc >= GAnproc) return 0;
      chk = 0;
      /* loop over blocks until a block with data is found */
      while (!chk) {
        /* get bounds for current block */
        for (j = 0; j < ndim; j++) {
          blo[j] = hdl->blk_size[j]*(hdl->index[j])+1;
          bhi[j] = hdl->blk_size[j]*(hdl->index[j]+1);
          if (bhi[j] > GA[handle].dims[j]) bhi[j] = GA[handle].dims[j];
        }
        /* check to see if this block overlaps with requested block
         * defined by lo and hi */
        chk = 1;
        for (j=0; j<ndim; j++) {
          /* check to see if at least one end point of the interval
           * represented by blo and bhi falls in the interval
           * represented by lo and hi */
          check1 = ((blo[j] >= hdl->lo[j] && blo[j] <= hdl->hi[j]) ||
              (bhi[j] >= hdl->lo[j] && bhi[j] <= hdl->hi[j]));
          /* check to see if interval represented by lo and hi
           * falls entirely within interval represented by blo and bhi */
          check2 = ((hdl->lo[j] >= blo[j] && hdl->lo[j] <= bhi[j]) &&
              (hdl->hi[j] >= blo[j] && hdl->hi[j] <= bhi[j]));
          /* If there is some data, move to the next section of code,
           * otherwise, check next block */
          if (!check1 && !check2) {
            chk = 0;
          }
        }
        
        if (!chk) {
          /* evaluate new offset for block */
          itmp = 1;
          for (j=0; j<ndim; j++) {
            itmp *= bhi[j]-blo[j]+1;
          }
          hdl->offset += itmp;
          /* increment to next block */
          hdl->index[0] += GA[handle].nblock[0];
          for (j=0; j<ndim; j++) {
            if (hdl->index[j] >= GA[handle].num_blocks[j] && j < ndim-1) {
              hdl->index[j] = hdl->proc_index[j];
              hdl->index[j+1] += GA[handle].nblock[j+1];
            }
          }
          if (hdl->index[ndim-1] >= GA[handle].num_blocks[ndim-1]) {
            hdl->iproc++;
            if (hdl->iproc >= GAnproc) return 0;
            hdl->offset = 0;
            gam_find_proc_indices(handle, hdl->iproc, hdl->proc_index);
            gam_find_proc_indices(handle, hdl->iproc, hdl->index);
          }
        }
      }
      if (chk) {
        Integer *clo, *chi;
        *plo = hdl->lobuf;
        *phi = hdl->hibuf;
        clo = *plo;
        chi = *phi;
        /* get the patch of block that overlaps requested region */
        gam_GetBlockPatch(blo,bhi,hdl->lo,hdl->hi,clo,chi,ndim);

        /* evaluate offset within block */
        last = ndim - 1;
#if COMPACT_SCALAPACK
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
#else
        l_offset = 0;
        jtot = 1;
        for (j=0; j<last; j++)  {
          ldrem[j] = hdl->blk_ld[j];
          blk_jinc = GA[handle].dims[j]%hdl->blk_size[j];
          if (hdl->blk_inc[j] > 0) {
            if (hdl->proc_index[j]<hdl->hlf_blk[j]) {
              blk_jinc = hdl->blk_size[j];
            } else if (hdl->proc_index[j] == hdl->hlf_blk[j]) {
              blk_jinc = hdl->blk_inc[j]%hdl->blk_size[j];
            } else {
              blk_jinc = 0;
            }
          }
          ldrem[j] += blk_jinc;
          l_offset += (clo[j]-blo[j]
              + ((blo[j]-1)/hdl->blk_dim[j])*hdl->blk_size[j])*jtot;
          jtot *= ldrem[j];
        }
        l_offset += (clo[last]-blo[last]
            + ((blo[last]-1)/hdl->blk_dim[j])*hdl->blk_size[last])*jtot;
#endif
        /* get pointer to data on remote block */
        pinv = (hdl->iproc)%GAnproc;
        if (p_handle > 0) {
          pinv = PGRP_LIST[p_handle].inv_map_proc_list[pinv];
        }
        *prem =  GA[handle].ptr[pinv]+l_offset*GA[handle].elemsize;
        *proc = pinv;

        /* evaluate new offset for block */
        itmp = 1;
        for (j=0; j<ndim; j++) {
          itmp *= bhi[j]-blo[j]+1;
        }
        hdl->offset += itmp;
        /* increment to next block */
        hdl->index[0] += GA[handle].nblock[0];
        for (j=0; j<ndim; j++) {
          if (hdl->index[j] >= GA[handle].num_blocks[j] && j < ndim-1) {
            hdl->index[j] = hdl->proc_index[j];
            hdl->index[j+1] += GA[handle].nblock[j+1];
          }
        }
        if (hdl->index[ndim-1] >= GA[handle].num_blocks[ndim-1]) {
          hdl->iproc++;
          hdl->offset = 0;
          gam_find_proc_indices(handle, hdl->iproc, hdl->proc_index);
          gam_find_proc_indices(handle, hdl->iproc, hdl->index);
        }
      }
    }
  }
  return 1;
}

/**
 * Clean up iterator
 */
void gai_iterator_destroy(_iterator_hdl *hdl)
{
}
