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
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_create =  pnga_sprs_array_create
#endif
logical pnga_sprs_array_create(Integer idim, Integer jdim, Integer type)
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
 * @param idx,jdx I and J indices of sparse array element
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
  Integer *offset;
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

  for (i=0; i<nvals; i++) {
    printf("(p[%d] (assemble) i: %d j: %d v: %f\n",me,
        SPA[hdl].idx[i],SPA[hdl].jdx[i],((double*)SPA[hdl].val)[i]);
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
  printf("p[%d] (assemble) nvals: %d\n",me,nvals);
  for (i=0; i<nvals; i++) {
    iproc = ((idx[i]-1)*nproc)/idim;
    if (iproc >= nproc) iproc = nproc-1;
    printf("p[%d] (assemble) i: %d idx: %d iproc: %d\n",me,i,idx[i],iproc);
    count[iproc]++;
    list[i] = top[iproc];
    top[iproc] = i;
  }
  printf("p[%d] (assemble) Got to 1\n",me);

  /* determine how many values of matrix are stored on this process and what the
   * offset on remote processes is for the data. Create a global array to
   * perform this calculation */
  g_offset = pnga_create_handle();
  pnga_set_data(g_offset,one,&nproc,MT_F_INT);
  if (!pnga_allocate(g_offset)) ret = 0;
  pnga_zero(g_offset);
  offset = (Integer*)malloc(nproc*sizeof(Integer));
  for (i=0; i<nproc; i++) {
    iproc = (i+me)%nproc+1;
    if (count[iproc-1] > 0) {
      offset[iproc-1] = pnga_read_inc(g_offset,&iproc,count[iproc-1]);
    printf("p[%d] (assemble) Got to 1a count[%d]: %d offset[%d]: %d\n",
        me,iproc-1,count[iproc-1],iproc-1,offset[iproc-1]);
    }
  }
  pnga_pgroup_sync(SPA[hdl].grp);
  size = (Integer*)malloc(nproc*sizeof(Integer));
  ilo = 1;
  ihi = nproc;
  pnga_get(g_offset,&ilo,&ihi,size,&nproc);
  printf("p[%d] (assemble) Got to 2\n",me);

  /* we now know how much data is on all processors (size) and have an offset on
   * remote processors that we can use to store data from this processor. Start by
   * constructing global arrays to hold data */
  map = (Integer*)malloc(nproc*sizeof(Integer));
  map[0] = 1;
  totalvals = size[0];
    printf("p[%d] (assemble) Got to 2a map[%d]: %d\n",me,0,map[0]);
  for (i=1; i<nproc; i++) {
    map[i] = map[i-1] + size[i-1];
    totalvals += size[i];
    printf("p[%d] (assemble) Got to 2a map[%d]: %d\n",me,i,map[i]);
  }
  free(size);
  SPA[hdl].g_data = pnga_create_handle();
  pnga_set_data(SPA[hdl].g_data,one,&totalvals,SPA[hdl].type);
  pnga_set_irreg_distr(SPA[hdl].g_data,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_data)) ret = 0;
  SPA[hdl].g_j = pnga_create_handle();
  pnga_set_data(SPA[hdl].g_j,one,&totalvals,MT_F_INT);
  pnga_set_irreg_distr(SPA[hdl].g_j,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_j)) ret = 0;
  /* create temporary array using g_i to hold *all* i indices. We will fix it up
   * later to only hold location of first j value
   */
  SPA[hdl].g_i = pnga_create_handle();
  pnga_set_data(SPA[hdl].g_i,one,&totalvals,MT_F_INT);
  pnga_set_irreg_distr(SPA[hdl].g_i,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_i)) ret = 0;
  printf("(assemble) Got to 3\n");

  /* fill up global arrays with data */
  for (i=0; i<nproc; i++) {
    iproc = (i+me)%nproc;
    printf("p[%d] Got to 3a count[%d]: %d\n",me,i,count[i]);
    if (count[iproc] > 0) {
      Integer j;
      ncnt = 0;
      char *vbuf = (char*)malloc(count[iproc]*elemsize);
      Integer *ibuf = (Integer*)malloc(count[iproc]*sizeof(Integer));
      Integer *jbuf = (Integer*)malloc(count[iproc]*sizeof(Integer));
      vals = SPA[hdl].val;
      printf("p[%d] Got to 3b vals: %p\n",me,vals);
      /* fill up buffers with data going to process iproc */
      j = top[iproc]; 
      printf("p[%d] Got to 3c top: %d vbuf: %p\n",me,j,vbuf);
      while (j >= 0) {
        memcpy(((char*)vbuf+ncnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
        ibuf[ncnt] = SPA[hdl].idx[j];
        jbuf[ncnt] = SPA[hdl].jdx[j];
  printf("p[%d] (assemble) idx: %d jdx: %d val: %f\n",me,ibuf[ncnt],jbuf[ncnt],
      *((double*)((char*)vbuf+ncnt*elemsize)));
        ncnt++;
        j = list[j];
      }
      printf("p[%d] Got to 3d ncnt: %d\n",me,ncnt);
      /* send data to global arrays */
      lo = map[iproc];
      lo += (Integer)offset[iproc];
      hi = lo+count[iproc]-1;
      printf("p[%d] Got to 3e lo: %d hi: %d\n",me,lo,hi);
      if (hi>=lo) {
        pnga_put(SPA[hdl].g_data,&lo,&hi,vbuf,&count[iproc]);
        pnga_put(SPA[hdl].g_i,&lo,&hi,ibuf,&count[iproc]);
        pnga_put(SPA[hdl].g_j,&lo,&hi,jbuf,&count[iproc]);
      }
      free(vbuf);
      free(ibuf);
      free(jbuf);
    }
  }
  pnga_pgroup_sync(SPA[hdl].grp);
  /* Local buffers are no longer needed */
  free(SPA[hdl].idx);
  free(SPA[hdl].jdx);
  free(SPA[hdl].val);
  printf("p[%d] (assemble) Got to 4\n",me);
  
  /* All data has been moved so that each process has a row block of the sparse
   * matrix. Now need to organize data within each process into column blocks.
   * Start by binning data by column index */
  pnga_distribution(SPA[hdl].g_data,me,&lo,&hi);
  free(list);
  nvals = hi - lo + 1;
  list = (Integer*)malloc(nvals*sizeof(Integer));
  pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jdx,&ld);
  for (i=0; i<nproc; i++) {
    count[i] = 0;
    top[i] = -1;
    offset[i] = -1;
  }
  for (i=0; i<nvals; i++) {
    iproc = ((jdx[i]-1)*nproc)/jdim;
    if (iproc >= nproc) iproc = nproc-1;
    count[iproc]++;
    list[i] = top[iproc];
    top[iproc] = i;
  }
  pnga_release(SPA[hdl].g_j,&lo,&hi);
  printf("p[%d] (assemble) Got to 5\n",me);
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
  printf("p[%d] (assemble) Got to 6\n",me);
  pnga_access_ptr(SPA[hdl].g_data,&lo,&hi,&vals,&ld);
  pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&idx,&ld);
  pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&jdx,&ld);
  for (i=0; i<nproc; i++) {
    Integer j = top[i];
    if (j >= 0) {
      char* vbuf = SPA[hdl].val;
      Integer* ibuf = SPA[hdl].idx;
      Integer* jbuf = SPA[hdl].jdx;
      SPA[hdl].blkidx[ncnt] = i;
      /* copy values from global array to local buffers */
      SPA[hdl].blksize[ncnt] = 0;
      while(j >= 0) {
        memcpy(((char*)vbuf+icnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
        ibuf[icnt] = idx[j];
        jbuf[icnt] = jdx[j];
  printf("p[%d] (assemble) idx: %d jdx: %d val: %f\n",me,ibuf[icnt],jbuf[icnt],
      *((double*)((char*)vbuf+icnt*elemsize)));
        j = list[j];
        SPA[hdl].blksize[ncnt]++;
        icnt++;
      }
      ncnt++;
    }
  }
  free(count);
  free(top);
  printf("p[%d] (assemble) Got to 7\n",me);
  /* Values have all been sorted into column blocks within the row block. Now
   * need to sort them by row. Start by evaluating lower and upper row indices */
  SPA[hdl].ilo = (SPA[hdl].idim*me)/nproc;
  while ((SPA[hdl].ilo*nproc)/idim < me) {
    SPA[hdl].ilo++;
  }
  while ((SPA[hdl].ilo*nproc)/idim > me) {
    SPA[hdl].ilo--;
  }
  if ((SPA[hdl].ilo*nproc)/idim != me) { printf("p[%d] ilo: %d (ilo*nproc)/idim: %d\n",
      me,SPA[hdl].ilo,(SPA[hdl].ilo*nproc)/idim);
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
  if ((SPA[hdl].ihi*nproc)/idim != me) printf("p[%d] ihi: %d (ihi*nproc)/idim: %d\n",
      me,SPA[hdl].ihi,(SPA[hdl].ihi*nproc)/idim);
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
  printf("p[%d] (assemble) Got to 7a ilo: %d ihi: %d nblocks: %d\n",
      me,SPA[hdl].ilo,SPA[hdl].ihi,SPA[hdl].nblocks);
  pnga_pgroup_gop(SPA[hdl].grp,pnga_type_f2c(MT_F_INT),offset,nproc,"+");
  printf("p[%d] (assemble) Got to 8\n",me);
  /* Construct new version of g_i */
  map[0] = 1;
  nvals = offset[0];
  printf("p[%d] (assemble) map[0]: %d offset: %d\n",me,map[0],offset[0]);
  for (i=1; i<nproc; i++) {
    map[i] = map[i-1] + offset[i-1];
  printf("p[%d] (assemble) map[%d]: %d offset[%d]\n",me,i,map[i],i,offset[i]);
    nvals += offset[i];
  }
  SPA[hdl].g_i = pnga_create_handle();
  pnga_set_data(SPA[hdl].g_i,one,&nvals,MT_F_INT);
  pnga_set_irreg_distr(SPA[hdl].g_i,map,&nproc);
  if (!pnga_allocate(SPA[hdl].g_i)) ret = 0;
  pnga_distribution(SPA[hdl].g_i,me,&lo,&hi);
  pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&idx,&ld);
  printf("p[%d] (assemble) Got to 9 lo: %d hi: %d nvals: %d\n",me,lo,hi,nvals);

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
  printf("p[%d] (assemble) Got to 10 blksize[%d]: %d ibuf: %p\n",me,i,SPA[hdl].blksize[i],ibuf);
    for (j=0; j<SPA[hdl].blksize[i]; j++) {
      irow = ibuf[j]-1-SPA[hdl].ilo;
      if (irow >= nrows || irow < 0) {
        printf("p[%d] irow out of bounds irow: %d nrows: %d\n", irow,nrows);
      }
      printf("p[%d] Storing irow: %d j: %d\n",me,irow,j);
      list[j] = top[irow];
      top[irow] = j;
      count[irow]++;
    }
  printf("p[%d] (assemble) Got to 11\n",me);
    /* copy elements back into  global arrays after sorting by row index */
    icnt = 0;
    char *vbuf = (char*)SPA[hdl].val+ncnt*SPA[hdl].size;
  printf("p[%d] (assemble) Got to 12 nrows: %d\n",me,nrows);
    for (j=0; j<nrows; j++) {
      irow = top[j];
      (idx+i*(nrows+1))[j] = icnt;
      printf("p[%d] (assemble) Got to 12a idx[%d*%d][%d]: %d irow: %d\n",me,i,nrows,j,
          (idx+i*(nrows+1))[j],irow);
      while (irow >= 0) {
        jdx[jcnt] = jbuf[irow];
        memcpy((vptr+icnt*elemsize),(vbuf+irow*elemsize),(size_t)elemsize);
  printf("p[%d] (assemble) Got to 12b jcnt: %d idx: %d jdx: %d val: %f\n",me,jcnt,j,
      jdx[jcnt],*((double*)(vptr+icnt*elemsize)));
        irow = list[irow];
        icnt++;
        jcnt++;
      }
    }
    if (top[j] >= 0) (idx+i*(nrows+1))[nrows] = icnt; 
  printf("p[%d] (assemble) Got to 13 idx[%d*%d][%d]: %d\n",
      me,i,nrows,nrows,(idx+i*nrows)[nrows]);
    /* TODO: (maybe) sort each row so that individual elements are arrange in
     * order of increasing j */
    SPA[hdl].offset[i] = ncnt;
    ncnt += SPA[hdl].blksize[i];
  printf("p[%d] (assemble) Got to 14 offset[%d]: %d\n",me,i,SPA[hdl].offset[i]);
  }
  printf("p[%d] (assemble) Got to 15\n",me);
  free(SPA[hdl].val);
  free(SPA[hdl].idx);
  free(SPA[hdl].jdx);
  SPA[hdl].val = NULL;
  SPA[hdl].idx = NULL;
  SPA[hdl].jdx = NULL;
  printf("p[%d] (assemble) Got to 16\n",me);

  pnga_release(SPA[hdl].g_data,&lo,&hi);
  pnga_release(SPA[hdl].g_i,&lo,&hi);
  pnga_release(SPA[hdl].g_j,&lo,&hi);
  printf("p[%d] (assemble) Got to 17\n",me);

  free(count);
  printf("p[%d] (assemble) Got to 17a\n",me);
  free(top);
  printf("p[%d] (assemble) Got to 17b\n",me);
  free(list);
  printf("p[%d] (assemble) Got to 17c\n",me);
  free(offset);
  printf("p[%d] (assemble) Got to 17d\n",me);
  free(map);
  printf("p[%d] (assemble) Got to 18\n",me);
  return ret;
}

/**
 * Return the range of rows held by processor iproc. Note that this will return
 * valid index ranges for the processor even if none of the rows contain
 * non-zero values
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
  *lo = SPA[hdl].ilo + 1;
  *hi = SPA[hdl].ihi + 1;
}

/**
 * Return the range of columns in column block iproc. Note that this will return
 * valid index ranges for the processor even the column block contains no
 * non-zero values
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
  (*lo)++;
  (*hi)++;
}

/**
 * Return pointers to the compressed sparse row formatted data corresponding to
 * the column block icol. If the column block has no non-zero values, the
 * pointers are returned as null.
 * @param s_a sparse array handle
 * @param 
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_access_col_block =  pnga_sprs_array_access_col_block
#endif
void pnga_sprs_array_access_col_block(Integer s_a, Integer icol,
    Integer **idx, Integer **jdx, void *val)
{
  Integer hdl = GA_OFFSET + s_a;
  char *lptr;
  Integer i,index;
  index = -1;
  for (i=0; i<SPA[hdl].nblocks; i++) {
    printf("icol: %d blkidx[%d]: %d\n",icol,i,SPA[hdl].blkidx[i]);
    if (SPA[hdl].blkidx[i] == icol) {
      index = i;
      break;
    }
  }
  if (index == -1) {
    *idx = NULL;
    *jdx = NULL;
    *(char**)val = NULL;
  }  else {
    Integer *tidx;
    Integer *tjdx;
    char *lptr;
    Integer lo, hi, ld;
    Integer me = pnga_pgroup_nodeid(SPA[hdl].grp);
    Integer offset = SPA[hdl].offset[index];
    printf("p[%d] (access) offset: %d index: %d\n",me,offset,index);

    /* access local portions of GAs containing data */
    pnga_distribution(SPA[hdl].g_data,me,&lo,&hi);
    pnga_access_ptr(SPA[hdl].g_data,&lo,&hi,&lptr,&ld);
    printf("p[%d] (access) lo: %d hi: %d lptr: %p\n",me,lo,hi,lptr);
    pnga_distribution(SPA[hdl].g_i,me,&lo,&hi);
    pnga_access_ptr(SPA[hdl].g_i,&lo,&hi,&tidx,&ld);
    pnga_distribution(SPA[hdl].g_j,me,&lo,&hi);
    pnga_access_ptr(SPA[hdl].g_j,&lo,&hi,&tjdx,&ld);
    pnga_release(SPA[hdl].g_data,&lo,&hi);
    pnga_release(SPA[hdl].g_i,&lo,&hi);
    pnga_release(SPA[hdl].g_j,&lo,&hi);

    /* shift pointers to correct location */
    ld = SPA[hdl].ihi - SPA[hdl].ilo + 2;
    lptr = lptr + offset*SPA[hdl].size;
    tjdx = tjdx + offset;
    tidx = tidx + ld*index;
    
    *idx = tidx;
    *jdx = tjdx;
    printf("tidx: %p *idx: %p *jdx: %p sizeof(Integer): %d lptr: %p\n",
        tidx,*idx,*jdx,sizeof(Integer),lptr);
    *(char**)val = lptr;
    /*
    printf("val: %p idx: %p jdx: %p\n",lptr,idx,jdx);
    */
  }
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
