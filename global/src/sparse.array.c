/* $Id: sparse.array.c,v 1.80.2.18 2007/12/18 22:22:27 d3g293 Exp $ */
/* 
 * module: sparse.array.c
 * author: Bruce Palmer
 * description: implements a sparse data layout for 2D arrays (matrices)
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

#define MAX_INT_VALUE 2147483648

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
#include "ga_sparse.array.h"
#include "armci.h"
#include "macdecls.h"
#include "ga-papi.h"
#include "ga-wapi.h"
#include "thread-safe.h"

_sparse_array *SPA;
/**
 * Initial number of values that can be stored on a processor before needing to
 * increase size of local buffers
 */
#define INIT_BUF_SIZE 1024

/**
 * Internal function to initialize sparse array data structures. This needs to
 * be called somewhere inside ga_initialize
 */
void sai_init_sparse_arrays()
{
  Integer i;
  SPA = (_sparse_array*)malloc(MAX_ARRAYS*sizeof(_sparse_array));
  for (i=0; i<MAX_ARRAYS; i++) {
    SPA[i].active = 0;
    SPA[i].ready = 0;
    SPA[i].nblocks = 0;
    SPA[i].blkidx = NULL;
    SPA[i].blksize = NULL;
    SPA[i].offset = NULL;
    SPA[i].idx = NULL;
    SPA[i].jdx = NULL;
    SPA[i].val = NULL;
    SPA[i].g_data = GA_OFFSET-1;
    SPA[i].g_i = GA_OFFSET-1;
    SPA[i].g_j = GA_OFFSET-1;
  }
}

/**
 * Internal function to finalize sparse array data structures. This needs to
 * be called somewhere inside ga_terminate
 */
void sai_terminate_sparse_arrays()
{
  Integer i;
  for (i=0; i<MAX_ARRAYS; i++) {
    Integer ga = SPA[i].g_data + GA_OFFSET;
    if (GA[ga].actv==1) pnga_destroy(SPA[i].g_data);
    ga = SPA[i].g_i + GA_OFFSET;
    if (GA[ga].actv==1) pnga_destroy(SPA[i].g_i);
    ga = SPA[i].g_j + GA_OFFSET;
    if (GA[ga].actv==1) pnga_destroy(SPA[i].g_j);
    if (SPA[i].blkidx) free(SPA[i].blkidx);
    if (SPA[i].blksize) free(SPA[i].blksize);
    if (SPA[i].offset) free(SPA[i].offset);
    if (SPA[i].idx) free(SPA[i].idx);
    if (SPA[i].jdx) free(SPA[i].jdx);
    if (SPA[i].val) free(SPA[i].val);
  }
  free(SPA);
}

/**
 * Create a new sparse array
 * @param idim,jdim I (row) and J (column) dimensions of sparse array
 * @param type type of data stored in sparse array
 * @param size size of indices stored in sparse array
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_create =  pnga_sprs_array_create
#endif
Integer pnga_sprs_array_create(Integer idim, Integer jdim, Integer type, Integer size)
{
  Integer i, hdl, s_a;
  GAvalidtypeM(pnga_type_f2c((int)type));
  if (idim <= 0 || jdim <= 0)
    pnga_error("(ga_sprs_array_create) Invalid array dimenensions",0);
  for (i=0; i<MAX_ARRAYS; i++) {
    if (!SPA[i].active) {
      SPA[i].active = 1;
      SPA[i].idx = (Integer*)malloc(INIT_BUF_SIZE*sizeof(Integer));
      SPA[i].jdx = (Integer*)malloc(INIT_BUF_SIZE*sizeof(Integer));
      SPA[i].type = pnga_type_f2c((int)(type));
      SPA[i].idx_size = size;
      SPA[i].size = GAsizeofM(SPA[i].type);
      SPA[i].val = malloc(INIT_BUF_SIZE*SPA[i].size);
      SPA[i].nval = 0;
      SPA[i].maxval = INIT_BUF_SIZE;
      SPA[i].idim = idim;
      SPA[i].jdim = jdim;
      SPA[i].grp = pnga_pgroup_get_default();
      SPA[i].nprocs = pnga_pgroup_nnodes(SPA[i].grp);
      hdl = i;
      s_a = hdl - GA_OFFSET;
      break;
    }
  }
  return s_a;
}

/**
 * Add an element to a sparse array
 * @param s_a sparse array handle
 * @param idx, jdx I and J indices of sparse array element
 * @param val sparse array element value
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_add_element =  pnga_sprs_array_add_element
#endif
void pnga_sprs_array_add_element(Integer s_a, Integer idx, Integer jdx, void *val)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer nval = SPA[hdl].nval;
  Integer size = SPA[hdl].size;
  /* Check to see if array is active and not ready */
  if (!SPA[hdl].active) 
    pnga_error("(ga_sprs_array_add_element) Array not active",hdl);
  if (SPA[hdl].ready) 
    pnga_error("(ga_sprs_array_add_element) Array is already distributed",hdl);
  /* Check to see if existing buffers can hold another value. If not, then
   * create new buffers that are twice as large and copy old data to new
   * buffers*/
  if (nval >= SPA[hdl].maxval) {
    Integer i;
    Integer *tidx;
    Integer *tjdx;
    char *tval;
    char *oval = (char*)SPA[hdl].val;
    Integer me = pnga_pgroup_nodeid(SPA[hdl].grp);

    tidx = (Integer*)malloc(2*SPA[hdl].maxval*sizeof(Integer));
    tjdx = (Integer*)malloc(2*SPA[hdl].maxval*sizeof(Integer));
    tval = (char*)malloc(2*SPA[hdl].maxval*SPA[hdl].size);
    /* copy data in old arrays to new, larger array */
    for (i=0; i<nval; i++) {
      tidx[i] = SPA[hdl].idx[i];
      tjdx[i] = SPA[hdl].jdx[i];
      memcpy((tval+i*size),(oval+i*size),(size_t)size);
    }
    /* get rid of old arrays */
    free(SPA[hdl].idx);
    free(SPA[hdl].jdx);
    free(SPA[hdl].val);
    /* re-assign local buffers */
    SPA[hdl].idx = tidx;
    SPA[hdl].jdx = tjdx;
    SPA[hdl].val = tval;
    /* add new values */
    SPA[hdl].maxval = 2*SPA[hdl].maxval;
  }
  /* add new value to buffers */
  SPA[hdl].idx[nval] = idx;
  SPA[hdl].jdx[nval] = jdx;
  memcpy((char*)SPA[hdl].val+size*nval,(char*)val,(size_t)size);
  SPA[hdl].nval++;
}

/**
 * Prepare sparse array for use by distributing values into row blocks based on
 * processor and subdivide each row into column blocks, also based on processor
 * @param s_a sparse array handle
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_assemble =  pnga_sprs_array_assemble
#endif
logical pnga_sprs_array_assemble(Integer s_a)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer lo, hi, ld;
  Integer i,j,ilo,ihi;
  int64_t *offset;
  Integer *count;
  Integer *top;
  Integer *list;
  Integer *idx = SPA[hdl].idx;
  Integer *jdx = SPA[hdl].jdx;
  Integer nvals = SPA[hdl].nval;
  Integer nproc = pnga_pgroup_nnodes(SPA[hdl].grp);
  Integer me = pnga_pgroup_nodeid(SPA[hdl].grp);
  Integer elemsize = SPA[hdl].size;
  Integer iproc;
  Integer g_offset;
  Integer ret = 1;
  Integer *size;
  Integer *map;
  Integer totalvals;
  Integer one = 1;
  Integer nrows, irow, idim, jdim;
  char *vals;
  Integer ncnt, icnt, jcnt;
  Integer longidx;
  int *isdx;
  int *jsdx;
  int64_t *ildx;
  int64_t *jldx;

  /* set variable that distinguishes between long and ints for indices */
  if (SPA[hdl].idx_size == sizeof(int64_t)) {
    longidx = 1;
  } else {
    longidx = 0;
  }
  idim = SPA[hdl].idim;
  jdim = SPA[hdl].jdim;
  /* Create a linked list for values on this process and bin values by which
   * processor owns them */
  count = (Integer*)malloc(nproc*sizeof(Integer));
  top = (Integer*)malloc(nproc*sizeof(Integer));
  list = (Integer*)malloc(nvals*sizeof(Integer));
  for (i=0; i<nproc; i++) {
    count[i] = 0;
    top[i] = -1;
  }
  for (i=0; i<nvals; i++) {
    iproc = (idx[i]*nproc)/idim;
    if (iproc >= nproc) iproc = nproc-1;
    count[iproc]++;
    list[i] = top[iproc];
    top[iproc] = i;
  }

  /* determine how many values of matrix are stored on this process and what the
   * offset on remote processes is for the data. Create a global array to
   * perform this calculation */
  g_offset = pnga_create_handle();
  pnga_set_data(g_offset,one,&nproc,C_LONG);
  if (!pnga_allocate(g_offset)) ret = 0;
  pnga_zero(g_offset);
  offset = (int64_t*)malloc(nproc*sizeof(int64_t));
  for (i=0; i<nproc; i++) offset[i] = 0;
  for (i=0; i<nproc; i++) {
    /* internal indices are unit based, so iproc needs
     * to be incremented by 1 */
    iproc = (i+me)%nproc+1;
    /* C indices are still zero based so need to subtract 1 from iproc */
    if (count[iproc-1] > 0) {
      /* offset locations are not deterministic, but do guarantee that a space is
       * available to hold data going to each processor */
      offset[iproc-1] = (int64_t)pnga_read_inc(g_offset,&iproc,count[iproc-1]);
    }
  }
  pnga_pgroup_sync(SPA[hdl].grp);
  size = (Integer*)malloc(nproc*sizeof(Integer));
  /* internal indices are unit based */
  ilo = 1;
  ihi = nproc;
  pnga_get(g_offset,&ilo,&ihi,size,&nproc);

  /* we now know how much data is on all processors (size) and have an offset on
   * remote processors that we can use to store data from this processor. Start by
   * constructing global arrays to hold data */
  map = (Integer*)malloc(nproc*sizeof(Integer));
  /* internal indices are unit based so start map at 1 */
  map[0] = 1;
  totalvals = size[0];
  for (i=1; i<nproc; i++) {
    map[i] = map[i-1] + size[i-1];
    totalvals += size[i];
  }
  free(size);
  SPA[hdl].g_data = pnga_create_handle();
  pnga_set_data(SPA[hdl].g_data,one,&totalvals,SPA[hdl].type);
  pnga_set_irreg_distr(SPA[hdl].g_data,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_data)) ret = 0;
  SPA[hdl].g_j = pnga_create_handle();
  if (longidx) {
    pnga_set_data(SPA[hdl].g_j,one,&totalvals,C_LONG);
  } else {
    pnga_set_data(SPA[hdl].g_j,one,&totalvals,C_INT);
  }
  pnga_set_irreg_distr(SPA[hdl].g_j,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_j)) ret = 0;
  /* create temporary array using g_i to hold *all* i indices. We will fix it up
   * later to only hold location of first j value
   */
  SPA[hdl].g_i = pnga_create_handle();
  if (longidx) {
    pnga_set_data(SPA[hdl].g_i,one,&totalvals,C_LONG);
  } else {
    pnga_set_data(SPA[hdl].g_i,one,&totalvals,C_INT);
  }
  pnga_set_irreg_distr(SPA[hdl].g_i,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_i)) ret = 0;

  /* fill up global arrays with data */
  for (i=0; i<nproc; i++) {
    iproc = (i+me)%nproc;
    if (count[iproc] > 0) {
      Integer j;
      ncnt = 0;
      char *vbuf = (char*)malloc(count[iproc]*elemsize);
      int64_t *ilbuf;
      int64_t *jlbuf;
      int *ibuf;
      int *jbuf;
      if (longidx) {
        ilbuf = (int64_t*)malloc(count[iproc]*sizeof(int64_t));
        jlbuf = (int64_t*)malloc(count[iproc]*sizeof(int64_t));
      } else {
        ibuf = (int*)malloc(count[iproc]*sizeof(int));
        jbuf = (int*)malloc(count[iproc]*sizeof(int));
      } 
      vals = SPA[hdl].val;
      /* fill up buffers with data going to process iproc */
      j = top[iproc]; 
      if (longidx) {
        while (j >= 0) {
          memcpy(((char*)vbuf+ncnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
          ilbuf[ncnt] = (int64_t)SPA[hdl].idx[j];
          jlbuf[ncnt] = (int64_t)SPA[hdl].jdx[j];
          ncnt++;
          j = list[j];
        }
      } else {
        while (j >= 0) {
          memcpy(((char*)vbuf+ncnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
          ibuf[ncnt] = (int)SPA[hdl].idx[j];
          jbuf[ncnt] = (int)SPA[hdl].jdx[j];
          ncnt++;
          j = list[j];
        }
      }
      /* send data to global arrays */
      lo = map[iproc];
      lo += (Integer)offset[iproc];
      hi = lo+count[iproc]-1;
      if (hi>=lo) {
        pnga_put(SPA[hdl].g_data,&lo,&hi,vbuf,&count[iproc]);
        if (longidx) {
          pnga_put(SPA[hdl].g_i,&lo,&hi,ilbuf,&count[iproc]);
          pnga_put(SPA[hdl].g_j,&lo,&hi,jlbuf,&count[iproc]);
        } else {
          pnga_put(SPA[hdl].g_i,&lo,&hi,ibuf,&count[iproc]);
          pnga_put(SPA[hdl].g_j,&lo,&hi,jbuf,&count[iproc]);
        }
      }
      free(vbuf);
      if (longidx) {
        free(ilbuf);
        free(jlbuf);
      } else {
        free(ibuf);
        free(jbuf);
      }
    }
  }
  pnga_pgroup_sync(SPA[hdl].grp);
  /* Local buffers are no longer needed */
  free(SPA[hdl].idx);
  free(SPA[hdl].jdx);
  free(SPA[hdl].val);
  
  /* All data has been moved so that each process has a row block of the sparse
   * matrix. Now need to organize data within each process into column blocks.
   * Start by binning data by column index */
  pnga_distribution(SPA[hdl].g_data,me,&lo,&hi);
  free(list);
  nvals = hi - lo + 1;
  list = (Integer*)malloc(nvals*sizeof(Integer));
  /* get pointer to list of j indices */
  if (longidx) {
    pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jldx,&ld);
  } else {
    pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jsdx,&ld);
  }
  for (i=0; i<nproc; i++) {
    count[i] = 0;
    top[i] = -1;
    offset[i] = -1;
  }
  if (longidx) {
    for (i=0; i<nvals; i++) {
      iproc = (((Integer)jldx[i])*nproc)/jdim;
      if (iproc >= nproc) iproc = nproc-1;
      count[iproc]++;
      list[i] = top[iproc];
      top[iproc] = i;
    }
  } else {
    for (i=0; i<nvals; i++) {
      iproc = (((Integer)jsdx[i])*nproc)/jdim;
      if (iproc >= nproc) iproc = nproc-1;
      count[iproc]++;
      list[i] = top[iproc];
      top[iproc] = i;
    }
  }
  pnga_release(SPA[hdl].g_j,&lo,&hi);
  /* find out how many column blocks have data */
  ncnt = 0;
  for (i=0; i<nproc; i++) {
    if (count[i] > 0) ncnt++;
  }
  SPA[hdl].nblocks = ncnt;
  if (ncnt > 0) SPA[hdl].blkidx = (Integer*)malloc(ncnt*sizeof(Integer));
  if (ncnt > 0) SPA[hdl].offset = (Integer*)malloc(ncnt*sizeof(Integer));
  if (ncnt > 0) SPA[hdl].blksize = (Integer*)malloc(ncnt*sizeof(Integer));
  /* allocate local buffers to sort everything into column blocks */
  SPA[hdl].val = malloc(nvals*SPA[hdl].size);
  SPA[hdl].idx = malloc(nvals*sizeof(Integer));
  SPA[hdl].jdx = malloc(nvals*sizeof(Integer));
  ncnt = 0;
  icnt = 0;
  pnga_access_ptr(SPA[hdl].g_data,&lo,&hi,&vals,&ld);
  if (longidx) {
    pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&ildx,&ld);
    pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jldx,&ld);
  } else {
    pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&isdx,&ld);
    pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jsdx,&ld);
  }
  for (i=0; i<nproc; i++) {
    Integer j = top[i];
    if (j >= 0) {
      char* vbuf = SPA[hdl].val;
      Integer* ibuf = SPA[hdl].idx;
      Integer* jbuf = SPA[hdl].jdx;
      SPA[hdl].blkidx[ncnt] = i;
      /* copy values from global array to local buffers */
      SPA[hdl].blksize[ncnt] = 0;
      if (longidx) {
        while(j >= 0) {
          memcpy(((char*)vbuf+icnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
          ibuf[icnt] = (Integer)ildx[j];
          jbuf[icnt] = (Integer)jldx[j];
          j = list[j];
          SPA[hdl].blksize[ncnt]++;
          icnt++;
        }
      } else {
        while(j >= 0) {
          memcpy(((char*)vbuf+icnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
          ibuf[icnt] = (Integer)isdx[j];
          jbuf[icnt] = (Integer)jsdx[j];
          j = list[j];
          SPA[hdl].blksize[ncnt]++;
          icnt++;
        }
      }
      ncnt++;
    }
  }
  free(count);
  free(top);
  /* Values have all been sorted into column blocks within the row block. Now
   * need to sort them by row. Start by evaluating lower and upper row indices */
  SPA[hdl].ilo = (SPA[hdl].idim*me)/nproc;
  while ((SPA[hdl].ilo*nproc)/idim < me) {
    SPA[hdl].ilo++;
  }
  while ((SPA[hdl].ilo*nproc)/idim > me) {
    SPA[hdl].ilo--;
  }
  if ((SPA[hdl].ilo*nproc)/idim != me) {
    printf("p[%d] ilo: %d (ilo*nproc)/idim: %d\n",
      (int)me,(int)SPA[hdl].ilo,(int)((SPA[hdl].ilo*nproc)/idim));
    SPA[hdl].ilo++;
  }
  if (me < nproc-1) {
    SPA[hdl].ihi = (SPA[hdl].idim*(me+1))/nproc;
    while ((SPA[hdl].ihi*nproc)/idim < me+1) {
      SPA[hdl].ihi++;
    }
    while ((SPA[hdl].ihi*nproc)/idim > me+1) {
      SPA[hdl].ihi--;
    }
    SPA[hdl].ihi--;
  } else {
    SPA[hdl].ihi = SPA[hdl].idim-1;
  }
  if ((SPA[hdl].ihi*nproc)/idim != me)
    printf("p[%d] ihi: %d (ihi*nproc)/idim: %d\n",
      (int)me,(int)SPA[hdl].ihi,(int)((SPA[hdl].ihi*nproc)/idim));
  nrows = SPA[hdl].ihi - SPA[hdl].ilo + 1;
  /* Resize the i-index array to account for row blocks */
  pnga_destroy(SPA[hdl].g_i);
  /* Calculate number of row values that will need to be stored. Add an extra
   * row to account for the total size of the column block */
  for (i=0; i<nproc; i++) {
    offset[i] = 0;
    map[i] = 0;
  }
  offset[me] = (nrows+1)*SPA[hdl].nblocks;
  pnga_pgroup_gop(SPA[hdl].grp,C_LONG,offset,nproc,"+");
  /* Construct new version of g_i */
  map[0] = 1;
  nvals = offset[0];
  for (i=1; i<nproc; i++) {
    map[i] = map[i-1] + offset[i-1];
    nvals += offset[i];
  }
  SPA[hdl].g_i = pnga_create_handle();
  if (longidx) {
    pnga_set_data(SPA[hdl].g_i,one,&nvals,C_LONG);
  } else {
    pnga_set_data(SPA[hdl].g_i,one,&nvals,C_INT);
  }
  pnga_set_irreg_distr(SPA[hdl].g_i,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_i)) ret = 0;
  pnga_distribution(SPA[hdl].g_i,me,&lo,&hi);
  if (longidx) {
    pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&ildx,&ld);
  } else {
    pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&isdx,&ld);
  }

  /* Bin up elements by row index for each column block */
  count = (Integer*)malloc(nrows*sizeof(Integer));
  top = (Integer*)malloc(nrows*sizeof(Integer));
  ncnt = 0;
  icnt = 0;
  jcnt = 0;
  for (i=0; i<SPA[hdl].nblocks; i++) {
    char *vptr = vals+ncnt*SPA[hdl].size;
    Integer *ibuf = SPA[hdl].idx + ncnt;
    Integer *jbuf = SPA[hdl].jdx + ncnt;
    for (j=0; j<nrows; j++) {
      top[j] = -1;
      count[j] = 0;
    }
    for (j=0; j<SPA[hdl].blksize[i]; j++) {
      irow = ibuf[j]-SPA[hdl].ilo;
      list[j] = top[irow];
      top[irow] = j;
      count[irow]++;
    }
    /* copy elements back into  global arrays after sorting by row index */
    icnt = 0;
    char *vbuf = (char*)SPA[hdl].val+ncnt*SPA[hdl].size;
    if (longidx) {
      for (j=0; j<nrows; j++) {
        irow = top[j];
        (ildx+i*(nrows+1))[j] = (int64_t)icnt;
        while (irow >= 0) {
          jldx[jcnt] = (int64_t)jbuf[irow];
          memcpy((vptr+icnt*elemsize),(vbuf+irow*elemsize),(size_t)elemsize);
          irow = list[irow];
          icnt++;
          jcnt++;
        }
        if (top[j] >= 0) (ildx+i*(nrows+1))[nrows] = (int64_t)icnt; 
      }
    } else {
      for (j=0; j<nrows; j++) {
        irow = top[j];
        (isdx+i*(nrows+1))[j] = (int)icnt;
        while (irow >= 0) {
          jsdx[jcnt] = (int)jbuf[irow];
          memcpy((vptr+icnt*elemsize),(vbuf+irow*elemsize),(size_t)elemsize);
          irow = list[irow];
          icnt++;
          jcnt++;
        }
        if (top[j] >= 0) (isdx+i*(nrows+1))[nrows] = (int)icnt; 
      }
    }
    /* TODO: (maybe) sort each row so that individual elements are arrange in
     * order of increasing j */
    SPA[hdl].offset[i] = ncnt;
    ncnt += SPA[hdl].blksize[i];
  }
  free(SPA[hdl].val);
  free(SPA[hdl].idx);
  free(SPA[hdl].jdx);
  SPA[hdl].val = NULL;
  SPA[hdl].idx = NULL;
  SPA[hdl].jdx = NULL;

  /*
  if (GAme == 0) printf("G_DATA Array\n");
  pnga_print(SPA[hdl].g_data);
  if (GAme == 0) printf("G_I Array\n");
  pnga_print(SPA[hdl].g_i);
  if (GAme == 0) printf("G_J Array\n");
  pnga_print(SPA[hdl].g_j);
  */
  pnga_release(SPA[hdl].g_data,&lo,&hi);
  pnga_release(SPA[hdl].g_i,&lo,&hi);
  pnga_release(SPA[hdl].g_j,&lo,&hi);

  free(count);
  free(top);
  free(list);
  free(offset);
  free(map);
  return ret;
}

/**
 * Return the range of rows held by processor iproc. Note that this will return
 * valid index ranges for the processor even if none of the rows contain
 * non-zero values. This function is zero-based
 * @param s_a sparse array handle
 * @param iproc process for which index ranges are requested
 * @param lo,hi low and high values of the row indices held by this processor */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_row_distribution =  pnga_sprs_array_row_distribution
#endif
void pnga_sprs_array_row_distribution(Integer s_a, Integer iproc, Integer *lo,
    Integer *hi)
{
  Integer hdl = GA_OFFSET + s_a;
  *lo = SPA[hdl].ilo;
  if (iproc == pnga_pgroup_nodeid(SPA[hdl].grp)) {
    *lo = SPA[hdl].ilo;
    *hi = SPA[hdl].ihi;
  } else {
    pnga_sprs_array_column_distribution(s_a, iproc, lo, hi);
  }
}

/**
 * Return the range of columns in column block iproc. Note that this will return
 * valid index ranges for the processor even the column block contains no
 * non-zero values. This function is zero-based
 * @param s_a sparse array handle
 * @param iproc process for which index ranges are requested
 * @param lo,hi low and high values of the row indices held by this processor */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_column_distribution =  pnga_sprs_array_column_distribution
#endif
void pnga_sprs_array_column_distribution(Integer s_a, Integer iproc, Integer *lo,
    Integer *hi)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer nproc = SPA[hdl].nprocs;
  Integer jdim = SPA[hdl].jdim;
  *lo = (SPA[hdl].jdim*iproc)/nproc;
  while (((*lo)*nproc)/jdim < iproc) {
    (*lo)++;
  }
  while (((*lo)*nproc)/jdim > iproc) {
    (*lo)--;
  }
  if (iproc < nproc-1) {
    (*hi) = (jdim*(iproc+1))/nproc;
    while (((*hi)*nproc)/jdim < iproc+1) {
      (*hi)++;
    }
    while (((*hi)*nproc)/jdim > iproc+1) {
      (*hi)--;
    }
    (*hi)--;
  } else {
    *hi = jdim-1;
  }
}

/**
 * Return list of column blocks containing data on this processor
 * @param void *idx indices of colum blocks containing data
 * @param Integer n number of column blocks with data
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_col_block_list =  pnga_sprs_array_col_block_list
#endif
void pnga_sprs_array_col_block_list(Integer s_a, Integer **idx, Integer *n)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer i, index;
  *n = SPA[hdl].nblocks;
  *idx = (Integer*)malloc(SPA[hdl].nblocks*sizeof(Integer));
  for (i=0; i<SPA[hdl].nblocks; i++) {
    (*idx)[i] = SPA[hdl].blkidx[i];
  }
}


/**
 * Return pointers to the compressed sparse row formatted data corresponding to
 * the column block icol. If the column block has no non-zero values, the
 * pointers are returned as null.
 * @param s_a sparse array handle
 * @param icol index indicating column block (corresponds to a processor
 *             location)
 * @param idx location of first index in jdx correspoding to local row index i
 *            idx[i+1]-idx[i]+1 corresponds to the number of non-zero values
 *            in local row i
 * @param jdx column indices of non-zero zero matrix values
 * @param val array of non-zero matrix values
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_access_col_block =  pnga_sprs_array_access_col_block
#endif
void pnga_sprs_array_access_col_block(Integer s_a, Integer icol,
    void *idx, void *jdx, void *val)
{
  Integer hdl = GA_OFFSET + s_a;
  char *lptr;
  Integer i,index;
  Integer longidx;
  if (SPA[hdl].idx_size == sizeof(int64_t)) {
    longidx = 1;
  } else {
    longidx = 0;
  }
  index = -1;
  for (i=0; i<SPA[hdl].nblocks; i++) {
    if (SPA[hdl].blkidx[i] == icol) {
      index = i;
      break;
    }
  }
  if (index == -1) {
    if (longidx) {
      *(int**)idx = NULL;
      *(int**)jdx = NULL;
    } else {
      *(int64_t**)idx = NULL;
      *(int64_t**)jdx = NULL;
    }
    *(char**)val = NULL;
  }  else {
    int *tidx;
    int *tjdx;
    int64_t *tlidx;
    int64_t *tljdx;
    char *lptr;
    Integer lo, hi, ld;
    Integer me = pnga_pgroup_nodeid(SPA[hdl].grp);
    Integer offset = SPA[hdl].offset[index];

    /* access local portions of GAs containing data */
    pnga_distribution(SPA[hdl].g_data,me,&lo,&hi);
    pnga_access_ptr(SPA[hdl].g_data,&lo,&hi,&lptr,&ld);
    pnga_distribution(SPA[hdl].g_i,me,&lo,&hi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&tlidx,&ld);
    } else {
      pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&tidx,&ld);
    }
    pnga_distribution(SPA[hdl].g_j,me,&lo,&hi);
    if (longidx) {
      pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&tljdx,&ld);
    } else {
      pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&tjdx,&ld);
    }
    pnga_release(SPA[hdl].g_data,&lo,&hi);
    pnga_release(SPA[hdl].g_i,&lo,&hi);
    pnga_release(SPA[hdl].g_j,&lo,&hi);

    /* shift pointers to correct location */
    ld = SPA[hdl].ihi - SPA[hdl].ilo + 2;
    lptr = lptr + offset*SPA[hdl].size;
    if (longidx) {
      tljdx = tljdx + (int64_t)offset;
      tlidx = tlidx + (int64_t)(ld*index);
      *(int64_t**)idx = tlidx;
      *(int64_t**)jdx = tljdx;
    } else {
      tjdx = tjdx + (int)offset;
      tidx = tidx + (int)(ld*index);
      *(int**)idx = tidx;
      *(int**)jdx = tjdx;
    }
    *(char**)val = lptr;
  }
}

/**
 * Multiply a sparse matrix by a sparse vector
 * @param s_a handle for sparse matrix
 * @param g_a handle for vector
 * @param g_v handle for product vector
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_matvec_multiply =  pnga_sprs_array_matvec_multiply
#endif
void pnga_sprs_array_matvec_multiply(Integer s_a, Integer g_a, Integer g_v)
{
  Integer s_hdl = GA_OFFSET + s_a;
  int local_sync_begin,local_sync_end;
  Integer s_grp = SPA[s_hdl].grp;
  Integer me = pnga_pgroup_nodeid(s_grp);
  Integer nproc = pnga_pgroup_nnodes(s_grp);

  Integer ilo, ihi, jlo, jhi, klo, khi;
  double *vsum, *vbuf, *vptr;
  int64_t *iptr = NULL, *jptr = NULL;
  Integer i, j, iproc, ncols;
  double one_r = 1.0;
  Integer one = 1;
  Integer adim, vdim, arank, vrank, dims[GA_MAX_DIM];
  Integer atype, vtype;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  /* Check that g_hdl and v_hdl are both vectors and that sizes
   * match */
  if (local_sync_begin) pnga_sync();
  pnga_inquire(g_a, &atype, &arank, dims);
  adim = dims[0];
  pnga_inquire(g_v, &vtype, &vrank, dims);
  vdim = dims[0];
  if (arank != 1) pnga_error("rank of A must be 1 (vector)",arank);
  if (vrank != 1) pnga_error("rank of V must be 1 (vector)",vrank);
  if (adim != SPA[s_hdl].jdim) {
    pnga_error("length of A must equal second dimension of sparse matrix",adim);
  }
  if (vdim != SPA[s_hdl].jdim) {
    pnga_error("length of V must equal second dimension of sparse matrix",vdim);
  }
  if (atype != SPA[s_hdl].type || vtype != SPA[s_hdl].type) {
    pnga_error("Data type of sparse matrix and A and V vectors must match",
        SPA[s_hdl].type);
  }
  
  /* Make sure product vector is zero */
  pnga_zero(g_v);
  /* multiply sparse matrix by sparse vector */
  pnga_sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  vsum = (double*)malloc((ihi-ilo+1)*sizeof(double));
  for (i=ilo; i<=ihi; i++) {
    vsum[i-ilo] = 0.0;
  }
  for (iproc=0; iproc<nproc; iproc++) {
    pnga_sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    pnga_sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
    if (vptr != NULL) {
      vbuf = (double*)malloc((jhi-jlo+1)*sizeof(double));
      klo = jlo+1;
      khi = jhi+1;
      pnga_get(g_a,&klo,&khi,vbuf,&one);
      for (i=ilo; i<=ihi; i++) {
        ncols = iptr[i+1-ilo]-iptr[i-ilo];
        for (j=0; j<ncols; j++) {
          vsum[i-ilo] += vptr[iptr[i-ilo]+j]*vbuf[jptr[iptr[i-ilo]+j]-jlo];
        }
      }
      free(vbuf);
    }
  }
  if (ihi>=ilo) {
    klo = ilo + 1;
    khi = ihi + 1;
    pnga_acc(g_v,&klo,&khi,vsum,&one,&one_r);
  }
  if (local_sync_end)  pnga_sync();
  free(vsum);
}

/**
 * Delete a sparse array and free any resources it may be using
 * @param s_a sparse array handle
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_destroy =  pnga_sprs_array_destroy
#endif
logical pnga_sprs_array_destroy(Integer s_a)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer ret = 1;
  if (SPA[hdl].ready) {
    if (!pnga_destroy(SPA[hdl].g_data)) ret = 0;
    if (!pnga_destroy(SPA[hdl].g_i)) ret = 0;
    if (!pnga_destroy(SPA[hdl].g_j)) ret = 0;
    if (SPA[hdl].blkidx != NULL) free(SPA[hdl].blkidx);
    if (SPA[hdl].blksize != NULL) free(SPA[hdl].blksize);
    if (SPA[hdl].offset != NULL) free(SPA[hdl].offset);
  } else if (SPA[hdl].active) {
    free(SPA[hdl].val);
    SPA[hdl].val = NULL;
    free(SPA[hdl].idx);
    SPA[hdl].idx = NULL;
    free(SPA[hdl].jdx);
    SPA[hdl].jdx = NULL;
  }
  SPA[hdl].active = 0;
  SPA[hdl].ready = 0;
  return ret;
}

/**
 * Print out values of sparse matrix using i,j,val format. Only non-zero
 * value of matrix are printed. This routine gathers all data on one process so
 * it may cause a memory overflow for a large matrix
 * @param s_a sparse array handle
 * @param file name of file to store sparse array
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_export =  pnga_sprs_array_export
#endif
void pnga_sprs_array_export(Integer s_a, const char* file)
{
  Integer hdl = GA_OFFSET + s_a;
  int size  = SPA[hdl].size;
  int type = SPA[hdl].type;
  char frmt[32];
  char *cptr;
  int i;
  int offset;
  Integer nlen_data;
  Integer nlen_j;
  Integer nlen_i;
  Integer dim;
  Integer type_t;
  Integer tcnt;
  int idx_size = SPA[hdl].idx_size;
  void *vptr;
  void *iptr;
  void *jptr;
  Integer lo, hi, ld;
  char op[2];
  FILE *SPRS;
  int *ilo, *ihi;
  Integer me = pnga_pgroup_nodeid(SPA[hdl].grp);
  Integer nprocs = pnga_pgroup_nnodes(SPA[hdl].grp);
  Integer iproc, iblock;
  Integer nblocks;
  int *istart;
  int icnt;
  Integer g_blks;
  Integer one = 1;

  /* find low and hi row indices on each processor */
  ilo = (int*)malloc(nprocs*sizeof(int));
  ihi = (int*)malloc(nprocs*sizeof(int));
  for (iproc=0; iproc<nprocs; iproc++) {
    ilo[iproc] = 0;
    ihi[iproc] = 0;
  }
  ilo[me] = (int)SPA[hdl].ilo;
  ihi[me] = (int)SPA[hdl].ihi;
  op[0] = '+';
  op[1] = '\0';
  pnga_pgroup_gop(SPA[hdl].grp,C_INT,ilo,nprocs,op);
  pnga_pgroup_gop(SPA[hdl].grp,C_INT,ihi,nprocs,op);

  /* find proc indices of non-zero column blocks on each processor */
  istart = (int*)malloc((nprocs+1)*sizeof(int));
  for (iproc=0; iproc<nprocs+1; iproc++) {
    istart[iproc] = 0;
  }
  istart[me] = (int)SPA[hdl].nblocks;
  pnga_pgroup_gop(SPA[hdl].grp,C_INT,istart,nprocs,op);
  {
    int ilast = 0;
    icnt = 0;
    for (iproc=0; iproc<nprocs; iproc++) {
      icnt += ilast;
      ilast = istart[iproc];
      istart[iproc] = (int)icnt;
    }
    icnt += ilast;
    istart[nprocs] = icnt;
  }
  for (i=0; i<nprocs; i++) {
  }
    
  g_blks = pnga_create_handle();
  lo = (Integer)icnt;
  pnga_set_data(g_blks,one,&lo,MT_F_INT);
  pnga_set_pgroup(g_blks,SPA[hdl].grp);
  pnga_allocate(g_blks);
  /* assign non-zero blocks to g_blks */
  lo = (Integer)(istart[me]+1);
  hi = (Integer)istart[me+1];
  pnga_put(g_blks,&lo,&hi,SPA[hdl].blkidx,&one);
  pnga_pgroup_sync(SPA[hdl].grp);

  /* format print statement */
  if (idx_size == 4) {
    strncpy(frmt,"\%d \%d",5);
    offset = 5;
  } else {
    strncpy(frmt,"\%ld \%ld",7);
    offset = 7;
  }
  cptr = frmt+offset;
  if (type == C_FLOAT || type == C_DBL) {
    strncpy(cptr," \%14.6e",7);
    offset = 7;
  } else if (type == C_INT) {
    strncpy(cptr," \%d",3);
    offset = 3;
  } else if (type == C_LONG) {
    strncpy(cptr," \%ld",4);
    offset = 4;
  } else if (type == C_SCPL || type == C_DCPL) {
    strncpy(cptr," \%14.6e \%14.6e",14);
    offset = 14;
  } 
  cptr += offset;
  cptr[0] = '\n';
  cptr++;
  cptr[0] = '\0';
  tcnt = 0;

  if (me == 0) {
    /* open file */
    SPRS = fopen(file,"w");
    /* Loop over all processors */
    for (iproc = 0; iproc<nprocs; iproc++) {
      /* find out how much data is on each process, allocate
       * local buffers to hold it and copy data to local buffer */
      /* Copy data from global arrays to local buffers */
      Integer iilo, iihi;
      int ibl;
      Integer *blkptr;
      Integer count;
      ld = 1;
      pnga_distribution(SPA[hdl].g_data, iproc, &lo, &hi);
      nlen_data = hi-lo+1;
      vptr = malloc(nlen_data*size);
      pnga_get(SPA[hdl].g_data,&lo,&hi,vptr,&ld);

      pnga_distribution(SPA[hdl].g_i, iproc, &lo, &hi);
      nlen_i = hi-lo+1;
      iptr = malloc(nlen_i*idx_size);
      pnga_get(SPA[hdl].g_i,&lo,&hi,iptr,&ld);

      pnga_distribution(SPA[hdl].g_j, iproc, &lo, &hi);
      nlen_j = hi-lo+1;
      jptr = malloc(nlen_j*idx_size);
      pnga_get(SPA[hdl].g_j,&lo,&hi,jptr,&ld);

      nblocks = SPA[hdl].nblocks;
      /* loop over column blocks in row block */
      blkptr = (Integer*)malloc(nblocks*sizeof(Integer));
      lo = (Integer)(istart[iproc]+1);
      hi = (Integer)istart[iproc+1];
      pnga_get(g_blks,&lo,&hi,blkptr,&one);
      offset = 0;
      count = 0;
      for (ibl = 0; ibl<nblocks; ibl++) {
        Integer joffset = count;
        iblock = blkptr[ibl];
        /* write out data to file */
        /* loop over rows on processor iproc */
        iilo = ilo[iproc];
        iihi = ihi[iproc];
        nlen_i = iihi-iilo+1;
        /* iilo = ilo[iblock]; */
        if (idx_size == 4) {
          int i, j;
          Integer *idx = (Integer*)iptr;
          Integer *jdx = (Integer*)jptr;
          for (i=0; i<nlen_i; i++) {
            int tlo = idx[i+offset];
            int thi = idx[i+offset+1];
            count += (Integer)(thi-tlo);
            for (j=tlo; j<thi; j++) {
              if (type == C_FLOAT) {
                float val = ((float*)vptr)[j+joffset];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],val);
              } else if (type == C_DBL) {
                double val = ((double*)vptr)[j+joffset];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],val);
              } else if (type == C_INT) {
                int val = ((int*)vptr)[j];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],val);
              } else if (type == C_LONG) {
                int64_t val = ((int64_t*)vptr)[j+joffset];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],val);
              } else if (type == C_SCPL) {
                float rval = ((float*)vptr)[2*(j+joffset)];
                float ival = ((float*)vptr)[2*(j+joffset)+1];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],rval,ival);
              } else if (type == C_DCPL) {
                double rval = ((double*)vptr)[2*(j+joffset)];
                double ival = ((double*)vptr)[2*(j+joffset)+1];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],rval,ival);
              }
            }
          }
        } else {
          int64_t i, j;
          int64_t *idx = (int64_t*)iptr;
          int64_t *jdx = (int64_t*)jptr;
          for (i=0; i<nlen_i; i++) {
            int64_t tlo = idx[i+offset];
            int64_t thi = idx[i+offset+1];
            count += (Integer)(thi-tlo);
            for (j=tlo; j<thi; j++) {
              if (type == C_FLOAT) {
                float val = ((float*)vptr)[j+joffset];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],val);
              } else if (type == C_DBL) {
                double val = ((double*)vptr)[j+joffset];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],val);
              } else if (type == C_INT) {
                int val = ((int*)vptr)[j];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],val);
              } else if (type == C_LONG) {
                int64_t val = ((int64_t*)vptr)[j+joffset];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],val);
              } else if (type == C_SCPL) {
                float rval = ((float*)vptr)[2*(j+joffset)];
                float ival = ((float*)vptr)[2*(j+joffset)+1];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],rval,ival);
              } else if (type == C_DCPL) {
                double rval = ((double*)vptr)[2*(j+joffset)];
                double ival = ((double*)vptr)[2*(j+joffset)+1];
                fprintf(SPRS,frmt,i+iilo,jdx[j+joffset],rval,ival);
              }
            }
          }
        }
        offset += (nlen_i+1);
      }
      free(blkptr);

      /* free local buffers */
      free(vptr);
      free(jptr);
      free(iptr);
    }
    fclose(SPRS);
  }
  free(ilo);
  free(ihi);
  pnga_pgroup_sync(SPA[hdl].grp);
}

/**
 * Extract diagonal portion of matrix and store it in a distributed vector.
 * Vector has same partition as row partition of matrix
 * @param s_a sparse array handle
 * @param g_d handle of 1D array containint diagonal
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_get_diag =  pnga_sprs_array_get_diag
#endif
void pnga_sprs_array_get_diag(Integer s_a, Integer *g_d)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer *map;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nproc = pnga_pgroup_nnodes(grp);
  Integer iproc;
  Integer one = 1;
  Integer lo, hi;
  Integer ilo, ihi, jlo, jhi, ncols;
  Integer *iptr, *jptr;
  Integer i, j;
  void *diag, *vptr;
  char op[2];
  /* Get row distribution of matrix */
  map = (Integer*)malloc((nproc+1)*sizeof(Integer));
  for (iproc=0; iproc<nproc+1; iproc++) map[iproc] = 0;
  map[me] = SPA[hdl].ilo+1;
  if (me == nproc-1) map[me+1] = SPA[hdl].ihi+1;
  op[0] = '+';
  op[1] = '\0';
  if (sizeof(Integer) == 4) {
    pnga_pgroup_gop(grp, C_INT, map, nproc+1, op);
  } else {
    pnga_pgroup_gop(grp, C_LONG, map, nproc+1, op);
  }
  /* Create array to hold diagonal */
  *g_d = pnga_create_handle();
  pnga_set_data(*g_d,one,&SPA[hdl].idim,SPA[hdl].type);
  pnga_set_irreg_distr(*g_d,map,&nproc);
  pnga_allocate(*g_d);
  pnga_distribution(*g_d,me,&lo,&hi);
  pnga_access_ptr(*g_d, &lo, &hi, &diag, &one);
  /* extract diagonal elements */
  pnga_sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  pnga_sprs_array_column_distribution(s_a,me,&jlo,&jhi);
  pnga_sprs_array_access_col_block(s_a,me,&iptr,&jptr,&vptr);
  for (i=ilo; i<=ihi; i++) {
    ncols = iptr[i+1-ilo]-iptr[i-ilo];
    for (j=0; j<ncols; j++) {
      if (i == jptr[iptr[i-ilo]+j]) {
        if (SPA[hdl].type == C_INT) {
          ((int*)diag)[i-ilo] = ((int*)vptr)[iptr[i-ilo]+j];
        } else if (SPA[hdl].type == C_LONG) {
          ((long*)diag)[i-ilo] = ((long*)vptr)[iptr[i-ilo]+j];
        } else if (SPA[hdl].type == C_LONGLONG) {
          ((long long*)diag)[i-ilo] = ((long long*)vptr)[iptr[i-ilo]+j];
        } else if (SPA[hdl].type == C_FLOAT) {
          ((float*)diag)[i-ilo] = ((float*)vptr)[iptr[i-ilo]+j];
        } else if (SPA[hdl].type == C_DBL) {
          ((double*)diag)[i-ilo] = ((double*)vptr)[iptr[i-ilo]+j];
        } else if (SPA[hdl].type == C_SCPL) {
          ((SingleComplex*)diag)[i-ilo] = ((SingleComplex*)vptr)[iptr[i-ilo]+j];
        } else if (SPA[hdl].type == C_DCPL) {
          ((DoubleComplex*)diag)[i-ilo] = ((DoubleComplex*)vptr)[iptr[i-ilo]+j];
        }
      }
    }
  }
  pnga_release_update(*g_d, &lo, &hi);
  pnga_pgroup_sync(grp);
}

/**
 * Left multiply sparse matrix by vector representing a diagonal matrix
 * @param s_a handle of sparse matrix
 * @param g_d handle of vector representing diagonal matrix
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_diag_left_multiply =  pnga_sprs_array_diag_left_multiply
#endif
void pnga_sprs_array_diag_left_multiply(Integer s_a, Integer g_d)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer d_hdl = GA_OFFSET + g_d;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nproc = pnga_pgroup_nnodes(grp);
  Integer ilo, ihi, jlo, jhi, klo, khi;
  Integer i, j, iproc, ncols;
  Integer type = SPA[hdl].type;
  Integer *iptr, *jptr;
  Integer one = 1;
  void *vbuf;
  void *vptr;

  /* check for basic compatibility */
  if (SPA[hdl].idim != GA[d_hdl].dims[0]) {
    pnga_error("(pnga_sprs_array_diag_left_multiply) dimensions don't match",0);
  }
  if (type != GA[d_hdl].type) {
    pnga_error("(pnga_sprs_array_diag_left_multiply) data types don't match",0);
  }
  if (GA[d_hdl].ndim != 1) {
    pnga_error("(pnga_sprs_array_diag_left_multiply) vector not of dimension 1",0);
  }

  /* get block from diagonal array corresponding to this row block (there is
   * only one) */
  pnga_sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  vbuf = malloc((ihi-ilo+1)*SPA[hdl].size);
  klo = ilo+1;
  khi = ihi+1;
  pnga_get(g_d,&klo,&khi,vbuf,&one);
  /* loop over blocks in sparse array */
  for (iproc=0; iproc<nproc; iproc++) {
    pnga_sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    pnga_sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
    if (vptr != NULL) {
      if (type == C_INT) {
        int *ibuf = (int*)vbuf;
        int *inptr = (int*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            inptr[iptr[i-ilo]+j] = ibuf[i-ilo]*inptr[iptr[i-ilo]+j];
          }
        }
      } else if (type == C_LONG) {
        long *lbuf = (long*)vbuf;
        long *lptr = (long*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            lptr[iptr[i-ilo]+j] = lbuf[i-ilo]*lptr[iptr[i-ilo]+j];
          }
        }
      } else if (type == C_LONGLONG) {
        long long *llbuf = (long long*)vbuf;
        long long *llptr = (long long*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            llptr[iptr[i-ilo]+j] = llbuf[i-ilo]*llptr[iptr[i-ilo]+j];
          }
        }
      } else if (type == C_FLOAT) {
        float *fbuf = (float*)vbuf;
        float *fptr = (float*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            fptr[iptr[i-ilo]+j] = fbuf[i-ilo]*fptr[iptr[i-ilo]+j];
          }
        }
      } else if (type == C_DBL) {
        double *dbuf = (double*)vbuf;
        double *dptr = (double*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            dptr[iptr[i-ilo]+j] = dbuf[i-ilo]*dptr[iptr[i-ilo]+j];
          }
        }
      } else if (type == C_SCPL) {
        float *sbuf = (float*)vbuf;
        float *sptr = (float*)vptr;
        float rbuf, ibuf, rval, ival;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            rbuf = sbuf[2*(i-ilo)];
            ibuf = sbuf[2*(i-ilo)+1];
            rval = sptr[2*(iptr[i-ilo]+j)];
            ival = sptr[2*(iptr[i-ilo]+j)+1];
            sptr[2*(iptr[i-ilo]+j)] = rbuf*rval-ibuf*ival;
            sptr[2*(iptr[i-ilo]+j)+1] = rbuf*ival+ibuf*rval;
          }
        }
      } else if (type == C_DCPL) {
        double *zbuf = (double*)vbuf;
        double *zptr = (double*)vptr;
        double rbuf, ibuf, rval, ival;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            rbuf = zbuf[2*(i-ilo)];
            ibuf = zbuf[2*(i-ilo)+1];
            rval = zptr[2*(iptr[i-ilo]+j)];
            ival = zptr[2*(iptr[i-ilo]+j)+1];
            zptr[2*(iptr[i-ilo]+j)] = rbuf*rval-ibuf*ival;
            zptr[2*(iptr[i-ilo]+j)+1] = rbuf*ival+ibuf*rval;
          }
        }
      }
    }
  }
  free(vbuf);
  pnga_pgroup_sync(grp);
}

/**
 * Shift diagonal values of sparse matrix by adding a constant to all diagonal
 * values
 * @param s_a handle of sparse matrix
 * @param shift pointer to shift value
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_shift_diag =  pnga_sprs_array_shift_diag
#endif
void pnga_sprs_array_shift_diag(Integer s_a, void *shift)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nproc = pnga_pgroup_nnodes(grp);
  Integer ilo, ihi, jlo, jhi, klo, khi;
  Integer i, j, iproc, ncols;
  Integer type = SPA[hdl].type;
  Integer *iptr, *jptr;
  void *vptr;

  /* get block from diagonal array corresponding to this row block (there is
   * only one) */
  pnga_sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  klo = ilo+1;
  khi = ihi+1;
  /* loop over blocks in sparse array */
  for (iproc=0; iproc<nproc; iproc++) {
    pnga_sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    pnga_sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
    if (vptr != NULL) {
      if (type == C_INT) {
        int ishift = *((int*)shift);
        int *inptr = (int*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            if (i == jptr[iptr[i-ilo]+j]) {
              inptr[iptr[i-ilo]+j] += ishift;
            }
          }
        }
      } else if (type == C_LONG) {
        long lshift = *((long*)shift);
        long *lptr = (long*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            if (i == jptr[iptr[i-ilo]+j]) {
              lptr[iptr[i-ilo]+j] += lshift;
            }
          }
        }
      } else if (type == C_LONGLONG) {
        long long llshift = *((long long*)shift);
        long long *llptr = (long long*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            if (i == jptr[iptr[i-ilo]+j]) {
              llptr[iptr[i-ilo]+j] += llshift;
            }
          }
        }
      } else if (type == C_FLOAT) {
        float fshift = *((float*)shift);
        float *fptr = (float*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            if (i == jptr[iptr[i-ilo]+j]) {
              fptr[iptr[i-ilo]+j] += fshift;
            }
          }
        }
      } else if (type == C_DBL) {
        double dshift = *((double*)shift);
        double *dptr = (double*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            if (i == jptr[iptr[i-ilo]+j]) {
              dptr[iptr[i-ilo]+j] += dshift;
            }
          }
        }
      } else if (type == C_SCPL) {
        float ishift = ((float*)shift)[0];
        float rshift = ((float*)shift)[1];
        float *sptr = (float*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            if (i == jptr[iptr[i-ilo]+j]) {
              sptr[2*(iptr[i-ilo]+j)] += rshift;
              sptr[2*(iptr[i-ilo]+j)+1] += ishift;
            }
          }
        }
      } else if (type == C_DCPL) {
        double ishift = ((double*)shift)[0];
        double rshift = ((double*)shift)[1];
        double *zptr = (double*)vptr;
        for (i=ilo; i<=ihi; i++) {
          ncols = iptr[i+1-ilo]-iptr[i-ilo];
          for (j=0; j<ncols; j++) {
            if (i == jptr[iptr[i-ilo]+j]) {
              zptr[2*(iptr[i-ilo]+j)] += rshift;
              zptr[2*(iptr[i-ilo]+j)+1] += ishift;
            }
          }
        }
      }
    }
  }
  pnga_pgroup_sync(grp);
}

/**
 * @param s_a sparse array that is to be duplicated
 * @return handle of duplicate array
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_duplicate =  pnga_sprs_array_duplicate
#endif
Integer pnga_sprs_array_duplicate(Integer s_a)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer new_hdl;
  Integer grp = SPA[hdl].grp;
  Integer me = pnga_pgroup_nodeid(grp);
  Integer nproc = pnga_pgroup_nnodes(grp);
  Integer s_dup;
  Integer i;

  /* create new array with same properties as old array */
  s_dup = pnga_sprs_array_create(SPA[hdl].idim,SPA[hdl].jdim,SPA[hdl].type,
      SPA[hdl].idx_size);
  /* find handle for new array and duplicate internal GAs */
  new_hdl = GA_OFFSET + s_dup;
  if (!pnga_duplicate(SPA[hdl].g_data,&SPA[new_hdl].g_data,"sparse_data_copy")) {
    pnga_error("(pnga_sprs_array_duplicate) Could not duplicate g_data",0);
  }
  if (!pnga_duplicate(SPA[hdl].g_i,&SPA[new_hdl].g_i,"sparse_i_index_copy")) {
    pnga_error("(pnga_sprs_array_duplicate) Could not duplicate g_i",0);
  }
  if (!pnga_duplicate(SPA[hdl].g_j,&SPA[new_hdl].g_j,"sparse_i_index_copy")) {
    pnga_error("(pnga_sprs_array_duplicate) Could not duplicate g_i",0);
  }
  /* copy remaining data structures */
  SPA[new_hdl].ilo = SPA[hdl].ilo;
  SPA[new_hdl].ihi = SPA[hdl].ihi;
  SPA[new_hdl].nblocks = SPA[hdl].nblocks;
  SPA[new_hdl].nval = SPA[hdl].nval;
  SPA[new_hdl].maxval = SPA[hdl].maxval;
  SPA[new_hdl].ready = SPA[hdl].ready;
  if (SPA[new_hdl].nblocks > 0) {
    SPA[new_hdl].blkidx
      = (Integer*)malloc(SPA[new_hdl].nblocks*sizeof(Integer));
    SPA[new_hdl].blksize
      = (Integer*)malloc(SPA[new_hdl].nblocks*sizeof(Integer));
    SPA[new_hdl].offset
      = (Integer*)malloc(SPA[new_hdl].nblocks*sizeof(Integer));
  }
  for (i=0; i<SPA[new_hdl].nblocks; i++) {
    SPA[new_hdl].blkidx[i] = SPA[hdl].blkidx[i];
    SPA[new_hdl].blksize[i] = SPA[hdl].blksize[i];
    SPA[new_hdl].offset[i] = SPA[hdl].offset[i];
  }
  return s_dup;
}
