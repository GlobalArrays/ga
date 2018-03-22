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
#define INIT_BUF_SIZE 1024;

/**
 * Internal function to initialize sparse array data structures
 */
void sai_init_sparse_arrays()
{
  Integer i;
  SPA = (_sparse_array*)malloc(MAX_ARRAY*sizeof(_sparse_array));
  for (i=0; i<MAX_ARRAY; i++) {
    SPA[i].active = 0;
    SPA[i].ready = 0;
    SPA[i].nblocks = 0;
    SPA[i].blkidx = NULL;
    SPA[i].offset = NULL;
    SPA[i].idx = NULL;
    SPA[i].jdx = NULL;
    SPA[i].val = NULL;
  }
}

/**
 * Create a new sparse array
 */
int pnga_sprs_array_create(Integer idim, Integer jdim, Integer type)
{
  Integer i, hdl, s_a;
  gam_checktype(pnga_type_f2c((int)type));
  if (idim <= 0 || jdim <= 0)
    pnga_error("(ga_sprs_array_create) Invalid array dimenensions",0);
  if (SPA[hdl].ready) 
    pnga_error("(ga_sprs_array_create) Array is already distributed",hdl);
  for (i=0; i<MAX_ARRAY; i++) {
    if (!SPA[i].active) {
      SPA[i].active = 1;
      SPA[i].idx = (Intger*)malloc(INIT_BUF_SIZE*sizeof(Integer));
      SPA[i].jdx = (Intger*)malloc(INIT_BUF_SIZE*sizeof(Integer));
      SPA[i].type = pnga_type_f2c((int)(type));
      SPA[i].size = GAsizeofM(SPA[i].type);
      SPA[i].val = malloc(INIT_BUF_SIZE*SPA[i].size);
      SPA[i].nval = 0;
      SPA[i].maxval = INIT_BUF_SIZE;
      SPA[i].idim = idim;
      SPA[i].jdim = jdim;
      SPA[i].grp = pnga_pgroup_get_default();
      hdl = i;
      s_a = hdl - GA_OFFSET;
      break;
    }
  }
  return s_a;
}

/**
 * Add element to a sparse array
 * @param s_a sparse array handle
 * @param idx,jdx I and J indices of sparse array element
 * @param val sparse array element value
 */
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
  memcopy((char)*SPA[hdl].val+size*nval,(char*)val,(size_t)size);
  SPA[hdl].nval++;
}

/**
 * Prepare sparse array for use by distributing values into row blocks based on
 * processor and subdivide each row into column blocks, also based on processor
 */
Integer pnga_sprs_array_assemble(Integer s_a)
{
  Integer hdl = GA_OFFSET + s_a;
  Integer lo, hi, ld;
  Integer i,ilo,ihi;
  int *offset;
  Integer *count;
  Integer *top;
  Integer *list;
  Integer *idx = SPA[hdl].idx;
  Integer *jdx = SPA[hdl].jdx;
  Integer nvals = SPA[hdl].nval;
  Integer nproc = pgna_pgroup_nnodes(SPA[hdl].grp);
  Integer me = pgna_pgroup_nodeid(SPA[hdl].grp);
  Integer elemsize = SPA[hdl].size;
  Integer iproc;
  Integer g_offset;
  Integer ret = 1;
  Integer *size;
  Integer *map;
  Integer totalvals;
  Integer one = 1;
  Integer nrows, irow;
  char *vals;
  Integer ncnt, icnt;

  /* determine lower and upper indices of row block held by this process */
  lo = (SPA[hdl].idim*me)/nproc;
  for (i=0; i<nproc; i++) {
    offset[i] = 0;
  }
  offset[me] = (int)lo;
  pnga_pgroup_gop(SPA[hdl].grp,MT_F_INT,offset,nproc,"+");
  if (me<nproc-1) {
    hi = offset[me+1];
  } else {
    hi = SPA[hdl].idim-1;
  }

  /* Create a linked list for values on this process and bin values by which
   * processor owns them */
  count = (Integer*)malloc(nproc*sizeof(Integer));
  top = (Integer*)malloc(nproc*sizeof(Integer));
  list = (Integer*)malloc(nvals*sizeof(Integer));
  for (i=0; i<nprocs; i++) {
    count[i] = 0;
    top[i] = -1;
    offset[i] = -1;
  }
  for (i=0; i<nvals; i++) {
    iproc = idx[i]/nproc;
    if (iproc >= nproc) iproc = nproc-1;
    count[iproc]++;
    list[i] = top[iproc];
    top[iproc] = i;
  }

  /* determine how many values of matrix are stored on this process and what the
   * offset on remote processes is for the data. Create a global array to
   * perform this calculation */
  g_offset = pnga_create_handle();
  pnga_set_data(g_offset,one,&nproc,MT_F_INT);
  if (!pnga_allocate(g_offset)) ret = 0;
  pnga_zero(g_offset);
  offset = (Integer*)malloc(nproc*sizeof(int));
  for (i=0; i<nproc; i++) {
    iproc = (i+me)%nproc;
    if (count[i] > 0) {
      offset[i] = (Integer)pnga_read_inc(g_offset,&iproc,count[i]);
    }
  }
  pnga_pgroup_sync(SPA[hdl].grp);
  size = (Integer*)malloc(nproc*sizeof(Integer));
  ilo = 1;
  ihi = nproc;
  pnga_get(g_offset,&ilo,&ihi,size,&nproc);

  /* we now know how much data is on all processors (map) and have an offset on
   * remote processors that we can use to store data from this processor. Start by
   * constructing global arrays to hold data */
  map = (Integer*)malloc(nproc*sizeof(Integer));
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

  /* fill up global arrays with data */
  for (i=0; i<nprocs; i++) {
    iproc = (i+me)%nproc;
    if (count[iproc] > 0) {
      Integer j;
      ncnt = 0;
      char *vbuf = (char*)malloc(count[iproc]*elemsize);
      Integer *ibuf = (Integer*)malloc(count[iproc]*sizeof(Integer));
      Integer *jbuf = (Integer*)malloc(count[iproc]*sizeof(Integer));
      vals = SPA[hdl].val;
      /* fill up buffers with data going to process iproc */
      j = top[iproc]; 
      while (j >= 0) {
        memcpy(((char*)vbuf+ncnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
        ibuf[ncnt] = SPA[hdl].g_i[j];
        jbuf[ncnt] = SPA[hdl].g_j[j];
        ncnt++;
        j = list[j];
      }
      /* send data to global arrays */
      lo = (SPA[hdl].idim*iproc)/iproc;
      lo += (Integer)offset[iproc]+1;
      hi = lo+count[i];
      pnga_put(SPA[hdl].g_data,&lo,&hi,vbuf,&count[iproc]);
      pnga_put(SPA[hdl].g_i,&lo,&hi,ibuf,&count[iproc]);
      pnga_put(SPA[hdl].g_j,&lo,&hi,jbuf,&count[iproc]);
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
  
  /* All data has been moved so that each process has a row block of the sparse
   * matrix. Now need to organize data within each process into column blocks.
   * Start by binning data by column index */
  pnga_distribution(SPA[hdl].g_data,me,&lo,&hi);
  free(list);
  nvals = hi - lo + 1;
  list = (Integer*)malloc(nvals*sizeof(Integer));
  pnga_access(SPA[hdl].g_j,lo,hi,&jdx,&ld);
  for (i=0; i<nprocs; i++) {
    count[i] = 0;
    top[i] = -1;
    offset[i] = -1;
  }
  for (i=0; i<nvals; i++) {
    iproc = jdx[i]/nproc;
    if (iproc >= nproc) iproc = nproc-1;
    count[iproc]++;
    list[i] = top[iproc];
    top[iproc] = i;
  }
  pnga_release(SPA[hdl].g_j,&lo,&hi);
  /* find out how many column blocks have data */
  ncnt = 0;
  for (i=0; i<nprocs; i++) {
    if (count[i] > 0) ncnt++;
  }
  SPA[hdl].nblocks = ncnt;
  SPA[hdl].blkidx = (Integer*)malloc(ncnt*sizeof(Integer));
  SPA[hdl].offset = (Integer*)malloc(ncnt*sizeof(Integer));
  SPA[hdl].blksize = (Integer*)malloc(ncnt*sizeof(Integer));
  /* allocate local buffers to sort everything into column blocks */
  SPA[hdl].val = malloc(nvals*SPA[hdl].size);
  SPA[hdl].idx = malloc(nvals*sizeof(Integer));
  SPA[hdl].jdx = malloc(nvals*sizeof(Integer));
  ncnt = 0;
  icnt = 0;
  pnga_access(SPA[hdl].g_data,&lo,&hi,&vals,&ld);
  pnga_access(SPA[hdl].g_i,&lo,&hi,&idx,&ld);
  pnga_access(SPA[hdl].g_j,&lo,&hi,&jdx,&ld);
  for (i=0; i<nprocs; i++) {
    Integer j = top[i];
    if (j >= 0) {
      char* vbuf = SPA[hdl].val;
      Integer* ibuf = SPA[hdl].idx;
      Integer* jbuf = SPA[hdl].jdx;
      SPA[hdl].blkidx[ncnt] = i;
      SPA[hdl].offset[ncnt] = icnt;
      /* copy values from global array to local buffers */
      SPA[hdl].blksize[ncnt] = 0;
      while(j >= 0) {
        memcpy(((char*)vbuf+icnt*elemsize),(vals+j*elemsize),(size_t)elemsize);
        ibuf[icnt] = idx[j];
        jbuf[icnt] = jdx[j];
        j = list[j];
        SPA[hdl].blksize[ncnt]++;
        icnt++;
      }
      ncnt++;
    }
  }
  free(count);
  free(top);
  /* Values have all been sorted into column blocks within the row block. Now
   * need to sort them by row. Start by evaluating lower and upper row indices */
  SPA[hdl].ilo = (SPA[hdl].idim*me)/nproc;
  if (me < nproc-1) {
    SPA[hdl].ihi = (SPA[hdl].idim*(me+1))/nproc-1;
  } else {
    SPA[hdl].ihi = SPA[hdl].idim-1;
  }
  nrows = SPA[hdl].ihi - SPA[hdl].ilo + 1;
  /* Resize the i-index array to account for row blocks */
  pnga_destroy(SPA[hdl].g_i);
  /* Calculate number of row values that will need to be stored */
  for (i=0; i<nproc; i++) {
    offset[i] = 0;
    map[i] = 0;
  }
  for (i=0; i<SPA[hdl].nblocks; i++) {
    SPA[hdl].offset[i] = i*nrows;
  }
  offset[me] = nrows*SPA[hdl].nblocks;
  pnga_pgroup_gop(SPA[hdl].grp,MT_F_INT,offset,nproc,"+");
  /* Construct new version of g_i */
  map[0] = 1;
  nvals = offset[0];
  for (i=1; i<nproc; i++) {
    map[i] = map[i-1] + offset[i-1];
    nvals += offset[i];
  }
  SPA[hdl].g_i = pnga_create_handle();
  pnga_set_data(SPA[hdl].g_i,&one,&nvals,MT_F_INT);
  pnga_set_irreg_distr(SPA[hdl].g_i,map,nproc);
  if (!pnga_allocate(SPA[hdl].g_i)) ret = 0;
  pnga_distribution(SPA[hdl].g_i,&lo,&hi,me,&ld);
  pnga_access(SPA[hdl].g_i,lo,hi,&idx,&ld);

  /* Bin up elements by row index for each column block */
  count = (Integer*)malloc(nrows*sizeof(Integer));
  top = (Integer*)malloc(nrows*sizeof(Integer));
  ncnt = 0;
  jcnt = 0;
  for (i=0; i<SPA[hdl].nblocks; i++) {
    if (SPA[hdl].blksize[i] == 0) continue;
    char *vptr = vals+ncnt*SPA[hdl].size;
    Integer *ibuf = idx + ncnt;
    Integer *jbuf = jdx + ncnt;
    for (j=0; j<nrows; j++) {
      top[j] = -1;
      count[j] = 0;
    }
    for (j=0; j<SPA[hdl].blksize; j++) {
      irow = ibuf[j]-SPH[hdl].ilo;
      list[j] = top[irow];
      top[irow] = j;
      count[irow]++;
    }
    /* copy elements back into  global arrays after sorting by row index */
    jcnt = 0;
    char *vbuf = (char*)SPA[hdl].val+ncnt*SPA[hdl].size;
    for (j=0; j<nrows; j++) {
      irow = top[j];
      (idx+i*nrows)[j] = jcnt;
      while (irow >= 0) {
        jdx[jcnt] = jbuf[irow];
        memcpy((vptr+jcnt*elemsize),((vbuf+irow*elemsize),(size_t)elemsize);
        irow = list[irow];
        jcnt++;
      }
    }
    /* TODO: (maybe) sort each row so that individual elements are arrange in
     * order of increasing j */
    ncnt += SPA[hdl].blksize[i];
  }
  free(SPA[hdl].val);
  free(SPA[hdl].idx);
  free(SPA[hdl].jdx);
  pnga_release(SPA[hdl].g_data,&lo,&hi);
  pnga_release(SPA[hdl].g_i,&lo,&hi);
  pnga_release(SPA[hdl].g_j,&lo,&hi);

  free(count);
  free(top);
  free(list);
  free(offset);
  free(map)
  return ret;
}
