/* $Id: sketch.c,v 1.80.2.18 2007/12/18 22:22:27 d3g293 Exp $ */
/* 
 * module: sketch.c
 * author: Bruce Palmer
 * description: implements a version fo the count-sketch algorithm
 *              for a sparse matrix
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
#include "ga-papi.h"
#include "ga-wapi.h"
#include "thread-safe.h"

#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC (1.0/MBIG)


/**
 * Create a new sparse array
 * @param s_a handle of sparse matrix
 * @param size_k number of rows in sparse matrix
 * @param g_k global array containing map to k values
 * @param g_w global array containing weights
 * @param trans flag that indicates whether or not to calculate
 *        transpose of sketch matrix. Needed for C interface
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_count_sketch =  pnga_sprs_array_count_sketch
#endif
Integer pnga_sprs_array_count_sketch(Integer s_a, Integer size_k,
    Integer *g_k, Integer *g_w, Integer trans)
{
  Integer i, ii, j, jj, n, hdl, g_a;
  Integer nrows;
  hdl = s_a + GA_OFFSET;
  Integer *klabels;
  void *kweights;
  Integer idim, jdim;
  Integer dims[2];
  Integer one = 1;
  Integer two = 2;
  Integer nblocks[2];
  Integer *top;
  Integer *list;
  void *kbuf;
  Integer nprocs, me;
  Integer k, kk, kcnt;
  Integer idx_size;
  Integer type;
  Integer lo[2], hi[2], ld;
  Integer *map, *size;
  if (SPA[hdl].idim < size_k) {
    pnga_error("Row dimension of sketch matrix must be less than row"
        "dimension of original matrix",size_k);
  }
  /* accumulate operation does not support type C_LONGLONG so fail if this
   *    * data type encountered */
  if (SPA[hdl].type == C_LONGLONG) {
    pnga_error("Data type of sparse matrix"
        " cannot be of type long long",SPA[hdl].type);
  }
  /* how many rows do I own */
  nrows = SPA[hdl].ihi-SPA[hdl].ilo+1;
  idim = SPA[hdl].idim;
  jdim = SPA[hdl].jdim;
  idx_size = SPA[hdl].idx_size;
  type = SPA[hdl].type;
  nprocs = pnga_pgroup_nnodes(SPA[hdl].grp);
  me = pnga_pgroup_nodeid(SPA[hdl].grp);
  /* create map array */
  size = (Integer*)malloc(nprocs*sizeof(Integer));
  map = (Integer*)malloc((nprocs+1)*sizeof(Integer));
  for (i=0; i<nprocs; i++) size[i] = 0;
  size[me] = nrows;
  if (idx_size == 4) {
    pnga_pgroup_gop(SPA[hdl].grp,C_INT,size,nprocs,"+");
  } else {
    pnga_pgroup_gop(SPA[hdl].grp,C_LONG,size,nprocs,"+");
  }
  map[0] = 1;
  for (i=1; i<nprocs; i++) {
    map[i] = map[i-1] + size[i-1];
  }
  map[nprocs] = 1;
  nblocks[0] = nprocs;
  nblocks[1] = 1;
  /* Assign each row to a label in the range  [0,size_k-1] */
  klabels = (Integer*)malloc(nrows*sizeof(Integer));
  for (i=0; i<nrows; i++) {
    klabels[i] = (Integer)(((double)size_k)*pnga_rand(0));
  }
  /* Assign each row a weight +/-1 */
  kweights = malloc(nrows*SPA[hdl].size);
  for (i=0; i<nrows; i++) {
    int val = (int)(2.0*pnga_rand(0));
    if (val == 0) {
      val = -1;
    } else {
      val = 1;
    }
    if (type == C_INT) {
      ((int*)kweights)[i] = val;
    } else if (type == C_LONG) {
      ((long*)kweights)[i] = (long)val;
    } else if (type == C_LONGLONG) {
      ((long long*)kweights)[i] = (long long)val;
    } else if (type == C_FLOAT) {
      ((float*)kweights)[i] = (float)val;
    } else if (type == C_DBL) {
      ((double*)kweights)[i] = (double)val;
    } else if (type == C_SCPL) {
      ((float*)kweights)[2*i] = (float)val;
      ((float*)kweights)[2*i+1] = 0.0;
    } else if (type == C_DCPL) {
      ((double*)kweights)[2*i] = (double)val;
      ((double*)kweights)[2*i+1] = 0.0;
    }
  }
  /* create distributed vectors to hold values in kweights and klabels */
  *g_w = pnga_create_handle();
  dims[0] = idim;
  pnga_set_data(*g_w,one,dims,SPA[hdl].type);
  pnga_set_irreg_distr(*g_w,map,nblocks);
  pnga_set_pgroup(*g_w,SPA[hdl].grp);
  pnga_allocate(*g_w);

  *g_k = pnga_create_handle();
  pnga_set_data(*g_k,one,dims,C_INT);
  pnga_set_irreg_distr(*g_k,map,nblocks);
  pnga_set_pgroup(*g_k,SPA[hdl].grp);
  pnga_allocate(*g_k);

  /* fill k-label and weight arrays */
  lo[0] = SPA[hdl].ilo+1; 
  hi[0] = SPA[hdl].ihi+1; 
  ld = nrows;
  kbuf = malloc(nrows*sizeof(int));
  for (i=0; i<nrows; i++) {
    ((int*)kbuf)[i] = (int)klabels[i];
  }
  pnga_put(*g_k,lo,hi,kbuf,&ld);
  free(kbuf);
  pnga_put(*g_w,lo,hi,kweights,&ld);
  pnga_sync(SPA[hdl].grp);
  /* Create an output matrix of size_k by jdim */
  g_a = pnga_create_handle();
  if (trans) {
    dims[0] = jdim;
    dims[1] = size_k;
  } else {
    dims[0] = size_k;
    dims[1] = jdim;
  }
  pnga_set_data(g_a,two,dims,SPA[hdl].type);
  pnga_set_pgroup(g_a,SPA[hdl].grp);
  pnga_allocate(g_a);
  pnga_zero(g_a);

  /* create a linked list of rows that map to each k-index */
  top = (Integer*)malloc(size_k*sizeof(Integer));
  list = (Integer*)malloc(nrows*sizeof(Integer));
  for (i=0; i<size_k; i++) top[i] = -1;
  for (i=0; i<nrows; i++) {
    list[i] = top[klabels[i]];
    top[klabels[i]] = i;
  }
  
  /* loop over all values of k */
  kbuf = malloc(jdim*SPA[hdl].size);
  if (trans) {
    lo[0] = 1;
    hi[0] = jdim;
    ld = jdim;
  } else {
    lo[1] = 1;
    hi[1] = jdim;
    ld = 1;
  }
  for (kk = 0; kk<size_k; kk++) {
    /* accumulate all non-zero values that contribute to this value of k */
    if (top[kk] > -1) {
      Integer irow = top[kk];
      memset(kbuf,0,jdim*SPA[hdl].size);
      while (irow > -1) {
        /* loop over column blocks */
        if (trans) {
          lo[1] = kk+1;
          hi[1] = kk+1;
        } else {
          lo[0] = kk+1;
          hi[0] = kk+1;
        }
        for (n=0; n<SPA[hdl].nblocks; n++) {
          Integer nblk = SPA[hdl].blkidx[n];
          void *vptr;
          Integer jlo, jhi;
          pnga_sprs_array_column_distribution(s_a,nblk,&jlo,&jhi);
          if (idx_size == 4) {
            int *iptr;
            int *jptr;
            pnga_sprs_array_access_col_block(s_a,n,&iptr,&jptr,&vptr);
            Integer idx = irow;
            Integer kcols = iptr[idx+1]-iptr[idx];
            for (k = 0; k<kcols; k++) {
              Integer jdx = jptr[iptr[idx]+k];
              if (type == C_INT) {
                ((int*)kbuf)[jdx] += ((int*)kweights)[idx]*
                  ((int*)vptr)[iptr[idx]+k];
              } else if (type == C_LONG) {
                ((long*)kbuf)[jdx] += ((long*)kweights)[idx]*
                  ((long*)vptr)[iptr[idx]+k];
              } else if (type == C_LONGLONG) {
                ((long long*)kbuf)[jdx] += ((long long*)kweights)[idx]*
                  ((long long*)vptr)[iptr[idx]+k];
              } else if (type == C_FLOAT) {
                ((float*)kbuf)[jdx] += ((float*)kweights)[idx]*
                  ((float*)vptr)[iptr[idx]+k];
              } else if (type == C_DBL) {
                ((double*)kbuf)[jdx] += ((double*)kweights)[idx]*
                  ((double*)vptr)[iptr[idx]+k];
              } else if (type == C_SCPL) {
                float rw, iw, rv, iv;
                rw = ((float*)kweights)[2*idx];
                iw = ((float*)kweights)[2*idx+1];
                rv = ((float*)vptr)[2*(iptr[idx]+k)];
                iv = ((float*)vptr)[2*(iptr[idx]+k)+1];
                ((float*)kbuf)[2*jdx] += rw*rv-iw*iv;
                ((float*)kbuf)[2*jdx+1] += rw*iv+iw*rv;
              } else if (type == C_DCPL) {
                double rw, iw, rv, iv;
                rw = ((double*)kweights)[2*idx];
                iw = ((double*)kweights)[2*idx+1];
                rv = ((double*)vptr)[2*(iptr[idx]+k)];
                iv = ((double*)vptr)[2*(iptr[idx]+k)+1];
                ((double*)kbuf)[2*jdx] += rw*rv-iw*iv;
                ((double*)kbuf)[2*jdx+1] += rw*iv+iw*rv;
              }
            }
          } else {
            int64_t *iptr;
            int64_t *jptr;
            pnga_sprs_array_access_col_block(s_a,n,&iptr,&jptr,&vptr);
            Integer idx = irow;
            Integer kcols = iptr[idx+1]-iptr[idx];
            for (k = 0; k<kcols; k++) {
              Integer jdx = jptr[iptr[idx]+k];
              if (type == C_INT) {
                ((int*)kbuf)[jdx] += ((int*)kweights)[idx]*
                  ((int*)vptr)[iptr[idx]+k];
              } else if (type == C_LONG) {
                ((long*)kbuf)[jdx] += ((long*)kweights)[idx]*
                  ((long*)vptr)[iptr[idx]+k];
              } else if (type == C_LONGLONG) {
                ((long long*)kbuf)[jdx] += ((long long*)kweights)[idx]*
                  ((long long*)vptr)[iptr[idx]+k];
              } else if (type == C_FLOAT) {
                ((float*)kbuf)[jdx] += ((float*)kweights)[idx]*
                  ((float*)vptr)[iptr[idx]+k];
              } else if (type == C_DBL) {
                ((double*)kbuf)[jdx] += ((double*)kweights)[idx]*
                  ((double*)vptr)[iptr[idx]+k];
              } else if (type == C_SCPL) {
                float rw, iw, rv, iv;
                rw = ((float*)kweights)[2*idx];
                iw = ((float*)kweights)[2*idx+1];
                rv = ((float*)vptr)[2*(iptr[idx]+k)];
                iv = ((float*)vptr)[2*(iptr[idx]+k)+1];
                ((float*)kbuf)[2*jdx] += rw*rv-iw*iv;
                ((float*)kbuf)[2*jdx+1] += rw*iv+iw*rv;
              } else if (type == C_DCPL) {
                double rw, iw, rv, iv;
                rw = ((double*)kweights)[2*idx];
                iw = ((double*)kweights)[2*idx+1];
                rv = ((double*)vptr)[2*(iptr[idx]+k)];
                iv = ((double*)vptr)[2*(iptr[idx]+k)+1];
                ((double*)kbuf)[2*jdx] += rw*rv-iw*iv;
                ((double*)kbuf)[2*jdx+1] += rw*iv+iw*rv;
              }
            }
          }
        }
        irow = list[irow];
      }
      /* accumulate row into g_a */
      if (type == C_INT) {
        int ione = 1;
        pnga_acc(g_a, lo, hi, kbuf, &ld, &ione);
      } else if (type == C_LONG) {
        long lone = 1;
        pnga_acc(g_a, lo, hi, kbuf, &ld, &lone);
      } else if (type == C_LONGLONG) {
        long long llone = 1;
        pnga_acc(g_a, lo, hi, kbuf, &ld, &llone);
      } else if (type == C_FLOAT) {
        float fone = 1.0;
        pnga_acc(g_a, lo, hi, kbuf, &ld, &fone);
      } else if (type == C_DBL) {
        double done = 1.0;
        pnga_acc(g_a, lo, hi, kbuf, &ld, &done);
      } else if (type == C_SCPL) {
        float cone[2];
        cone[0] = 1.0;
        cone[1] = 0.0;
        pnga_acc(g_a, lo, hi, kbuf, &ld, &cone);
      } else if (type == C_DCPL) {
        double zone[2];
        zone[0] = 1.0;
        zone[1] = 0.0;
        pnga_acc(g_a, lo, hi, kbuf, &ld, &zone);
      }
    }
  }
  pnga_pgroup_sync(SPA[hdl].grp);

  /* free up arrays */
  free(kbuf);
  free(klabels);
  free(kweights);
  free(top);
  free(list);
  return g_a;
}

/**
 * Determine the number of non-zero elements in a dense array
 * @param g_a handle to dense global array
 * @return number of nonzero values in array;
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_sprs_array_num_nonzeros =  pnga_sprs_array_num_nonzeros
#endif
int pnga_sprs_array_num_nonzeros(Integer g_a)
{
  Integer cnt = 0;
  void *ptr;
  Integer lo[2], hi[2], ld;
  Integer handle = g_a + GA_OFFSET;
  Integer type = GA[handle].type;
  Integer me = pnga_pgroup_nodeid(GA[handle].p_handle);
  Integer nprocs = pnga_pgroup_nnodes(GA[handle].p_handle);
  Integer nelem;
  Integer i;

  if (GA[handle].ndim != 2)
    pnga_error("Array must be 2 dimensional",GA[handle].ndim);
  pnga_distribution(g_a,me,lo,hi);
  nelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  pnga_access_ptr(g_a,lo,hi,&ptr,&ld);
  if (type == C_INT) {
    int *iptr = (int*)ptr;
    for (i=0; i<nelem; i++) {
      if (iptr[i] != 0) cnt++;
    }
  } else if (type == C_LONG) {
    long *lptr = (long*)ptr;
    for (i=0; i<nelem; i++) {
      if (lptr[i] != 0) cnt++;
    }
  } else if (type == C_LONGLONG) {
    long long *llptr = (long long*)ptr;
    for (i=0; i<nelem; i++) {
      if (llptr[i] != 0) cnt++;
    }
  } else if (type == C_FLOAT) {
    float *fptr = (float*)ptr;
    for (i=0; i<nelem; i++) {
      if (fptr[i] != 0.0) cnt++;
    }
  } else if (type == C_DBL) {
    double *dptr = (double*)ptr;
    for (i=0; i<nelem; i++) {
      if (dptr[i] != 0.0) cnt++;
    }
  } else if (type == C_SCPL) {
    float *fptr = (float*)ptr;
    for (i=0; i<nelem; i++) {
      if (fptr[2*i] != 0.0 || fptr[2*i+1] != 0.0) cnt++;
    }
  } else if (type == C_DCPL) {
    double *dptr = (double*)ptr;
    for (i=0; i<nelem; i++) {
      if (dptr[2*i] != 0.0 || dptr[2*i+1] != 0.0) cnt++;
    }
  }
  pnga_pgroup_gop(GA[handle].p_handle,type,&cnt,nprocs,"+");
}
