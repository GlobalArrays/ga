#if HAVE_CONFIG_H
#   include "config.h"
#endif
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

#define WRITE_VTK
#define CG_SOLVE 1
#define NDIM_SPRS 1048576
#define NDIM_DNS 16384

/**
 *  Solve Laplace's equation on a cubic domain using the sparse matrix
 *  functionality in GA.
 */

#define MAX_FACTOR 1024
void grid_factor(int p, int xdim, int ydim, int zdim,
    int *idx, int *idy, int *idz) {
  int i, j, k; 
  int ip, ifac, pmax, prime[MAX_FACTOR];
  int fac[MAX_FACTOR];
  int ix, iy, iz, ichk;

  i = 1;
/**
 *   factor p completely
 *   first, find all prime numbers, besides 1, less than or equal to 
 *   the square root of p
 */
  ip = p;
  pmax = 0;
  for (i=2; i<=ip; i++) {
    ichk = 1;
    for (j=0; j<pmax; j++) {
      if (i%prime[j] == 0) {
        ichk = 0;
        break;
      }
    }
    if (ichk) {
      pmax = pmax + 1;
      if (pmax > MAX_FACTOR) printf("Overflow in grid_factor\n");
      prime[pmax-1] = i;
    }
  }
/**
 *   find all prime factors of p
 */
  ip = p;
  ifac = 0;
  for (i=0; i<pmax; i++) {
    while(ip%prime[i] == 0) {
      ifac = ifac + 1;
      fac[ifac-1] = prime[i];
      ip = ip/prime[i];
    }
  }
/**
 *  p is prime
 */
  if (ifac==0) {
    ifac++;
    fac[0] = p;
  }
/**
 *    find three factors of p of approximately the same size
 */
  *idx = 1;
  *idy = 1;
  *idz = 1;
  for (i = ifac-1; i >= 0; i--) {
    ix = xdim/(*idx);
    iy = ydim/(*idy);
    iz = zdim/(*idz);
    if (ix >= iy && ix >= iz && ix > 1) {
      *idx = fac[i]*(*idx);
    } else if (iy >= ix && iy >= iz && iy > 1) {
      *idy = fac[i]*(*idy);
    } else if (iz >= ix && iz >= iy && iz > 1) {
      *idz = fac[i]*(*idz);
    } else {
      printf("Too many processors in grid factoring routine\n");
    }
  }
}

/**
 * subroutine to set up a sparse matrix for testing purposes
 * @param s_a sparse matrix handle
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 * @param flag if 1, decrease number of off-diagonal elements by 1
 */
void setup_matrix(int *s_a, int64_t dim, int type, int flag)
{
  int64_t jlo, jhi, idx; 
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j;
  int64_t skip_len, onum;
  void *d_val, *o_val;
  int size;
  int nskip = 5;

  if (me == 0) {
    printf("\n  Create sparse matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create sparse matrix */
  *s_a = NGA_Sprs_array_create64(dim, dim, type);

  /* Determine column block set by me */
  jlo = dim*me/nprocs;
  jhi = dim*(me+1)/nprocs-1;
  if (me == nprocs-1) jhi = dim-1;

  /* set up data values. Diagonal values are 2, off-diagonal values are -1 */
  if (type == C_INT) {
    size = sizeof(int);
  } else if (type == C_LONG) {
    size = sizeof(long);
  } else if (type == C_LONGLONG) {
    size = sizeof(long long);
  } else if (type == C_FLOAT) {
    size = sizeof(float);
  } else if (type == C_DBL) {
    size = sizeof(double);
  } else if (type == C_SCPL) {
    size = 2*sizeof(float);
  } else if (type == C_DCPL) {
    size = 2*sizeof(double);
  }

  d_val = malloc(size);
  o_val = malloc(size);

  if (type == C_INT) {
    *((int*)(d_val)) = 2;
    *((int*)(o_val)) = -1;
  } else if (type == C_LONG) {
    *((long*)(d_val)) = 2;
    *((long*)(o_val)) = -1;
  } else if (type == C_LONGLONG) {
    *((long long*)(d_val)) = 2;
    *((long long*)(o_val)) = -1;
  } else if (type == C_FLOAT) {
    *((float*)(d_val)) = 2.0;
    *((float*)(o_val)) = -1.0;
  } else if (type == C_DBL) {
    *((double*)(d_val)) = 2.0;
    *((double*)(o_val)) = -1.0;
  } else if (type == C_SCPL) {
    ((float*)d_val)[0]= 2.0;
    ((float*)d_val)[1]= 0.0;
    ((float*)o_val)[0]= -1.0;
    ((float*)o_val)[1]= 0.0;
  } else if (type == C_DCPL) {
    ((double*)d_val)[0]= 2.0;
    ((double*)d_val)[1]= 0.0;
    ((double*)o_val)[0]= -1.0;
    ((double*)o_val)[1]= 0.0;
  }

  /* loop over all columns in column block and add elements for each column.
   * Currently assume that each column has 5 elements, one on the diagonal 
   * and 4 others off the diagonal. Final matrix is partitioned into row blocks
   * so this guarantees that sorting routines for elements are tested */
  skip_len = dim/nskip;
  if (skip_len < 2)  {
    nskip = dim/2;
    skip_len = dim/nskip;
  }
  if (flag) {
    onum = nskip;
  } else {
    onum = nskip-1;
  }
  for (j=jlo; j<=jhi; j++) {
    NGA_Sprs_array_add_element64(*s_a,j,j,d_val);
    for (i=0; i<onum-1; i++) {
      int idx = (j+(i+1)*skip_len)%dim;
      NGA_Sprs_array_add_element64(*s_a,idx,j,o_val);
    }
  }
  if (NGA_Sprs_array_assemble(*s_a) && me == 0) {
    printf("\n  Sparse array assembly completed\n\n");
  }
  free(d_val);
  free(o_val);
}

/**
 * subroutine to set up a dense matrix for testing purposes
 * @param g_a dense matrix handle
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_dense_matrix(int *g_a, int64_t dim, int type)
{
  int64_t jlo, jhi, idx; 
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j;
  int64_t skip_len;
  void *d_val, *o_val;
  int size;
  int nskip = 5;
  int two = 2;
  int dims[2],lo[2],hi[2],ld[2];
  void *a;

  if (me == 0) {
    printf("\n  Create dense matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create dense matrix */
  *g_a = NGA_Create_handle();
  dims[0] = dim;
  dims[1] = dim;
  NGA_Set_data(*g_a,two,dims,type);
  NGA_Allocate(*g_a);
  GA_Zero(*g_a);

  /* set up data values. Diagonal values are 2, off-diagonal values are -1 */
  if (type == C_INT) {
    size = sizeof(int);
  } else if (type == C_LONG) {
    size = sizeof(long);
  } else if (type == C_LONGLONG) {
    size = sizeof(long long);
  } else if (type == C_FLOAT) {
    size = sizeof(float);
  } else if (type == C_DBL) {
    size = sizeof(double);
  } else if (type == C_SCPL) {
    size = 2*sizeof(float);
  } else if (type == C_DCPL) {
    size = 2*sizeof(double);
  }

  d_val = malloc(size);
  o_val = malloc(size);

  if (type == C_INT) {
    *((int*)(d_val)) = 2;
    *((int*)(o_val)) = -1;
  } else if (type == C_LONG) {
    *((long*)(d_val)) = 2;
    *((long*)(o_val)) = -1;
  } else if (type == C_LONGLONG) {
    *((long long*)(d_val)) = 2;
    *((long long*)(o_val)) = -1;
  } else if (type == C_FLOAT) {
    *((float*)(d_val)) = 2.0;
    *((float*)(o_val)) = -1.0;
  } else if (type == C_DBL) {
    *((double*)(d_val)) = 2.0;
    *((double*)(o_val)) = -1.0;
  } else if (type == C_SCPL) {
    ((float*)d_val)[0]= 2.0;
    ((float*)d_val)[1]= 0.0;
    ((float*)o_val)[0]= -1.0;
    ((float*)o_val)[1]= 0.0;
  } else if (type == C_DCPL) {
    ((double*)d_val)[0]= 2.0;
    ((double*)d_val)[1]= 0.0;
    ((double*)o_val)[0]= -1.0;
    ((double*)o_val)[1]= 0.0;
  }

  skip_len = dim/nskip;
  if (skip_len < 2)  {
    nskip = dim/2;
    skip_len = dim/nskip;
  }
  NGA_Distribution(*g_a,me,lo,hi);
  NGA_Access(*g_a,lo,hi,&a,ld);
  /* set all values in local local chunk of global arrays */
  for (j=lo[1]; j<=hi[1]; j++) {
    if (j>=lo[0] && j<=hi[0]) {
      memcpy(a+size*(j-lo[1]+(j-lo[0])*ld[0]),d_val,size);
    }
    for (i=0; i<nskip-1; i++) {
      int idx = (j+(i+1)*skip_len)%dim;
      if (idx>=lo[0] && idx<=hi[0]) {
      memcpy(a+size*(j-lo[1]+(idx-lo[0])*ld[0]),o_val,size);
      }
    }
  }
  NGA_Release(*g_a,lo,hi);

  GA_Sync();

  free(d_val);
  free(o_val);
}


/**
 * subroutine to set up a diagonal matrix for testing purposes
 * @param g_d handle to 1D array representing diagonal matrix
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_diag_matrix(int *g_d, int64_t dim, int type)
{
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t ilo, ihi, ld;
  int64_t i, j;
  int size;
  void *ptr;

  if (me == 0) {
    printf("  Create diagonal matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create a 1D global array */
  *g_d = NGA_Create_handle();
  NGA_Set_data64(*g_d,1,&dim,type);
  GA_Allocate(*g_d);

  /* Determine row block set by me */
  ilo = dim*me/nprocs;
  ihi = dim*(me+1)/nprocs-1;
  if (me == nprocs-1) ihi = dim-1;

  /* set up data values. Diagonal values are 2, off-diagonal values are -1 */
  if (type == C_INT) {
    size = sizeof(int);
  } else if (type == C_LONG) {
    size = sizeof(long);
  } else if (type == C_LONGLONG) {
    size = sizeof(long long);
  } else if (type == C_FLOAT) {
    size = sizeof(float);
  } else if (type == C_DBL) {
    size = sizeof(double);
  } else if (type == C_SCPL) {
    size = 2*sizeof(float);
  } else if (type == C_DCPL) {
    size = 2*sizeof(double);
  }


  /* get pointers to local data */
  NGA_Distribution64(*g_d,me,&ilo,&ihi);
  NGA_Access64(*g_d,&ilo,&ihi,&ptr,&ld);
  /* set diagonal values */
  for (i=ilo; i<=ihi; i++) {
    if (type == C_INT) {
      ((int*)ptr)[i-ilo] = (int)i;
    } else if (type == C_LONG) {
      ((long*)ptr)[i-ilo] = (long)i;
    } else if (type == C_LONGLONG) {
      ((long long*)ptr)[i-ilo] = (long long)i;
    } else if (type == C_FLOAT) {
      ((float*)ptr)[i-ilo] = (float)i;
    } else if (type == C_DBL) {
      ((double*)ptr)[i-ilo] = (double)i;
    } else if (type == C_SCPL) {
      ((float*)ptr)[2*(i-ilo)] = (float)i;
      ((float*)ptr)[2*(i-ilo)+1] = 0;
    } else if (type == C_DCPL) {
      ((double*)ptr)[2*(i-ilo)] = (double)i;
      ((double*)ptr)[2*(i-ilo)+1] = 0;
    }
  }
  NGA_Release64(*g_d,&ilo,&ihi);
  NGA_Sync();

  if (me == 0) {
    printf("\n  Diagonal array completed\n\n");
  }
}

/**
 * subroutine to set up a dense matrix with unique values at all entries
 * for testing purposes
 * @param g_a dense matrix handle
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_cdense_matrix(int *g_a, int64_t dim, int type)
{
  int64_t jlo, jhi, idx; 
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j;
  void *val;
  int size;
  int two = 2;
  int64_t dims[2],lo[2],hi[2],ld[2];
  void *a;

  if (me == 0) {
    printf("\n  Create complete dense matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create dense matrix */
  *g_a = NGA_Create_handle();
  dims[0] = dim;
  dims[1] = dim;
  NGA_Set_data64(*g_a,two,dims,type);
  NGA_Allocate(*g_a);
  NGA_Zero(*g_a);

  /* set up data values.  */
  if (type == C_INT) {
    size = sizeof(int);
  } else if (type == C_LONG) {
    size = sizeof(long);
  } else if (type == C_LONGLONG) {
    size = sizeof(long long);
  } else if (type == C_FLOAT) {
    size = sizeof(float);
  } else if (type == C_DBL) {
    size = sizeof(double);
  } else if (type == C_SCPL) {
    size = 2*sizeof(float);
  } else if (type == C_DCPL) {
    size = 2*sizeof(double);
  }

  val = malloc(size);

  /* set all values in local buffer */
  NGA_Distribution64(*g_a,me,lo,hi);
  NGA_Access64(*g_a,lo,hi,&a,ld);
  for (i=lo[0]; i<=hi[0]; i++) {
    for (j=lo[1]; j<=hi[1]; j++) {
      if (type == C_INT) {
        *((int*)val) = (int)(j+i*dim);
      } else if (type == C_LONG) {
        *((long*)val) = (long)(j+i*dim);
      } else if (type == C_LONGLONG) {
        *((long long*)val) = (long long)(j+i*dim);
      } else if (type == C_FLOAT) {
        *((float*)val) = (float)(j+i*dim);
      } else if (type == C_DBL) {
        *((double*)val) = (double)(j+i*dim);
      } else if (type == C_SCPL) {
        ((float*)val)[0] = (float)(j+i*dim);
        ((float*)val)[1] = 0.0;
      } else if (type == C_DCPL) {
        ((double*)val)[0] = (double)(j+i*dim);
        ((double*)val)[1] = 0.0;
      }
      memcpy(a+size*(j-lo[1]+(i-lo[0])*ld[0]),val,size);
    }
  }
  NGA_Release64(*g_a,lo,hi);

  GA_Sync();

  free(val);
}

/**
 * subroutine to set up matrix for quantization test
 * @param s_a sparse matrix handle
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_quant_matrix(int *s_a, int64_t dim, int type)
{
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j, idx, n;
  void *a;
  int64_t lo[2], hi[2];
#define REPEAT_LEN 16
#define SKIP_LEN 7

  if (type == C_SCPL || type == C_DCPL) {
    GA_Error("(setup_quant_matrix) Illegal data type requested",type);
  }

  if (me == 0) {
    printf("\n  Create quant matrix of size %ld x %ld\n",dim,dim);
  }

  /* Create sparse matrix */
  *s_a = NGA_Sprs_array_create64(dim, dim, type);

  lo[0] = (me*dim)/nprocs;
  hi[0] = ((me+1)*dim)/nprocs-1;
  if (me == nprocs-1) hi[0] = dim-1;
  lo[1] = 0;
  hi[1] = dim-1;
  if (type == C_INT) {
   a = malloc(sizeof(int));
  } else if (type == C_LONG) {
   a = malloc(sizeof(long));
  } else if (type == C_LONGLONG) {
   a = malloc(sizeof(long long));
  } else if (type == C_FLOAT) {
   a = malloc(sizeof(float));
  } else if (type == C_DBL) {
   a = malloc(sizeof(double));
  }
  for (i=lo[0]; i<=hi[0]; i++) {
    for (j=lo[1]; j<=hi[1]; j++) {
      idx = i*dim+j;
      if (idx%SKIP_LEN == 0 || i == j) {
        n = idx%REPEAT_LEN;
        if (type == C_INT) {
          ((int*)(a))[0] = (int)n;
          NGA_Sprs_array_add_element(*s_a,i,j,a);
        } else if (type == C_LONG) {
          ((long*)(a))[0] = (long)n;
          NGA_Sprs_array_add_element(*s_a,i,j,a);
        } else if (type == C_LONGLONG) {
          ((long long*)(a))[0] = (long long)n;
          NGA_Sprs_array_add_element(*s_a,i,j,a);
        } else if (type == C_FLOAT) {
          ((float*)(a))[0] = (float)n;
          NGA_Sprs_array_add_element(*s_a,i,j,a);
        } else if (type == C_DBL) {
          ((double*)(a))[0] = (double)n;
          NGA_Sprs_array_add_element(*s_a,i,j,a);
        }
      }
    }
  }

  free(a);

  if (NGA_Sprs_array_assemble(*s_a) && me == 0) {
    printf("\n  Sparse array assembly completed\n\n");
  }
}

void matrix_test(int type)
{
  int s_a, s_b, s_c, g_a, g_b, g_c, g_d;
  int64_t dim = NDIM_SPRS;
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int one = 1;
  int64_t ilo, ihi, jlo, jhi;
  int64_t i, j, k, l, iproc;
  int64_t ld;
  void *ptr;
  int64_t *idx, *jdx;
  int *nz_map;
  int ok;
  char op[2],plus[2];
  void *shift_val;
  void *a, *b, *c, *cp, *d;
  int *ab;
  double tbeg, time;
  double diff;
  int64_t lo[2],hi[2],tld[2];
  int all_ok = 1;
  
  /* create sparse matrix */
  setup_matrix(&s_a, dim, type, 0);

  /* extract diagonal of s_a to g_d */
  tbeg = GA_Wtime();
  NGA_Sprs_array_get_diag(s_a, &g_d);
  time = GA_Wtime()-tbeg;

  plus[0] = '+';
  plus[1] = '\0';
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    printf("    Time for matrix get diagonal operation: %16.8f\n",time);
  }
  NGA_Destroy(g_d);

  /**
   * Test shift diagonal operation
   */
  if (type == C_INT) {
    shift_val = malloc(sizeof(int));
    *((int*)shift_val) = 1;
  } else if (type == C_LONG) {
    shift_val = malloc(sizeof(long));
    *((long*)shift_val) = 1;
  } else if (type == C_LONGLONG) {
    shift_val = malloc(sizeof(long long));
    *((long long*)shift_val) = 1;
  } else if (type == C_FLOAT) {
    shift_val = malloc(sizeof(float));
    *((float*)shift_val) = 1.0;
  } else if (type == C_DBL) {
    shift_val = malloc(sizeof(double));
    *((double*)shift_val) = 1.0;
  } else if (type == C_SCPL) {
    shift_val = malloc(sizeof(SingleComplex));
    ((float*)shift_val)[0] = 1.0;
    ((float*)shift_val)[1] = 0.0;
  } else if (type == C_DCPL) {
    shift_val = malloc(sizeof(DoubleComplex));
    ((double*)shift_val)[0] = 1.0;
    ((double*)shift_val)[1] = 0.0;
  }
  tbeg = GA_Wtime();
  NGA_Sprs_array_shift_diag(s_a, shift_val);
  time = GA_Wtime()-tbeg;

  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    printf("\n    Time for matrix shift diagonal operation: %16.8f\n",time);
  }

  NGA_Sprs_array_destroy(s_a);
  free(shift_val);

  /* Create a fresh copy of sparse matrix */
  setup_matrix(&s_a, dim, type, 0);

  /* Create diagonal matrix */
  setup_diag_matrix(&g_d, dim, type);

  /* Do a right hand multiply */
  tbeg = GA_Wtime();
  NGA_Sprs_array_diag_right_multiply(s_a, g_d);
  time = GA_Wtime()-tbeg;
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    printf("    Time for matrix right diagonal multiply operation: %16.8f\n",time);
  }

  NGA_Sprs_array_destroy(s_a);
  NGA_Destroy(g_d);

  /* Create a fresh copy of sparse matrix */
  setup_matrix(&s_a, dim, type, 0);

  /* Create diagonal matrix */
  setup_diag_matrix(&g_d, dim, type);

  /* Do a left hand multiply */
  tbeg = GA_Wtime();
  NGA_Sprs_array_diag_left_multiply(s_a, g_d);
  time = GA_Wtime()-tbeg;
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    printf("    Time for matrix left diagonal multiply operation: %16.8f\n",time);
  }

  NGA_Sprs_array_destroy(s_a);
  NGA_Destroy(g_d);

  /* create sparse matrix A */
  setup_matrix(&s_a, dim, type, 0);

  time = 0.0;
  for (k=0; k<nprocs; k++) {
    for (l=0; l<nprocs; l++) {
      tbeg = GA_Wtime();
      if (!NGA_Sprs_array_get_block64(s_a, k, l, &idx, &jdx, &ptr,
          &ilo, &ihi, &jlo, &jhi)) {
        continue;
      }
      time += GA_Wtime()-tbeg;
      if (idx != NULL) free(idx);
      if (jdx != NULL) free(jdx);
      if (ptr != NULL) free(ptr);
    }
  }

  GA_Dgop(&time,1,plus);
  time /= (double)(nprocs*nprocs*nprocs);
  if (me == 0) {
    printf("    Time for matrix get block operation: %16.8f\n",time);
  }

  NGA_Sprs_array_destroy(s_a);

  /* create sparse matrix A */
  setup_matrix(&s_a, dim, type, 0);
  /* create sparse matrix B */
  setup_matrix(&s_b, dim, type, 0);

  /* multiply sparse matrix A times sparse matrix B */
  tbeg = GA_Wtime();
  s_c = NGA_Sprs_array_matmat_multiply(s_a, s_b);
  time = GA_Wtime()-tbeg;


  NGA_Sprs_array_destroy(s_b);
  tbeg = GA_Wtime();
  s_b = NGA_Sprs_array_matmat_multiply(s_c, s_a);
  time += GA_Wtime()-tbeg;

  GA_Dgop(&time,1,plus);
  time /= (double)(2*nprocs);
  if (me == 0) {
    printf("    Time for matrix-matrix multiply operation: %16.8f\n",time);
  }

  NGA_Sprs_array_destroy(s_a);
  NGA_Sprs_array_destroy(s_b);
  NGA_Sprs_array_destroy(s_c);

  dim = NDIM_DNS;
  /* create sparse matrix A */
  setup_matrix(&s_a, dim, type, 0);
  /* create dense matrix B */
  setup_dense_matrix(&g_b, dim, type);

  /* multiply sparse matrix A times dense matrix B */
  tbeg = GA_Wtime();
  g_c = NGA_Sprs_array_sprsdns_multiply(s_a, g_b);
  time = GA_Wtime()-tbeg;


  NGA_Sprs_array_destroy(s_a);
  NGA_Destroy(g_b);
  NGA_Destroy(g_c);

  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    printf("    Time for sparse-dense matrix-matrix"
           " multiply operation: %16.8f\n",time);
  }

  /* create dense matrix A */
  setup_dense_matrix(&g_a, dim, type);
  /* create sparse matrix B */
  setup_matrix(&s_b, dim, type, 0);

  /* multiply dense matrix A times sparse matrix B */
  tbeg = GA_Wtime();
  g_c = NGA_Sprs_array_dnssprs_multiply(g_a, s_b);
  time = GA_Wtime()-tbeg;

  NGA_Destroy(g_a);
  NGA_Sprs_array_destroy(s_b);
  NGA_Destroy(g_c);

  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    printf("    Time for dense-sparse matrix-matrix"
           " multiply operation: %16.8f\n",time);
  }

  /* create an ordinary global array with sparse non-zeros */
  setup_dense_matrix(&g_a, dim, type);
  /* copy dense matrix g_a to sparse matrix s_a */
  tbeg = GA_Wtime();
  s_a = NGA_Sprs_array_create_from_dense64(g_a);
  time = GA_Wtime()-tbeg;
  GA_Destroy(g_a);
  NGA_Sprs_array_destroy(s_a);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    printf("    Time for create from dense array operation: %16.8f\n",time);
  }

  /* create an ordinary global array with sparse non-zeros */
  setup_dense_matrix(&g_a, dim, type);
  /* copy dense matrix g_a to sparse matrix s_a */
  s_a = NGA_Sprs_array_create_from_dense64(g_a);
  /* now copy sparse matrix back to dense matrix g_b */
  tbeg = GA_Wtime();
  g_b = NGA_Sprs_array_create_from_sparse(s_a);
  time = GA_Wtime()-tbeg;
  GA_Destroy(g_a);
  GA_Destroy(g_b);
  NGA_Sprs_array_destroy(s_a);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    printf("    Time for create from sparse array operation: %16.8f\n",time);
  }

}

int main(int argc, char **argv) {
  int me,nproc;
  int ok = 1;
  int eight = 8;

  /**
   * Initialize MPI
   */
  MP_INIT(argc,argv);

  /* Initialize GA */
  NGA_Initialize();

  me = GA_Nodeid();
  nproc = GA_Nnodes();
  if (eight == SIZEOF_F77_INTEGER) {

    if (me == 0) {
      printf("\nTesting sparse matrices of size %d x %d\n"
          " and dense matrices of size %d x %d on %d processors\n\n",
          NDIM_SPRS,NDIM_SPRS,NDIM_DNS,NDIM_DNS,nproc);
    }

    /**
     * Test different data types
     */
#if 1
    if (me == 0) {
      printf("\nTesting matrices of type int\n");
    }
    matrix_test(C_INT);
#endif

#if 1
    if (me == 0) {
      printf("\nTesting matrices of type long\n");
    }
    matrix_test(C_LONG);

    if (me == 0) {
      printf("\nTesting matrices of type long long\n");
    }
    matrix_test(C_LONGLONG);

    if (me == 0) {
      printf("\nTesting matrices of type float\n");
    }
    matrix_test(C_FLOAT);

    if (me == 0) {
      printf("\nTesting matrices of type double\n");
    }
    matrix_test(C_DBL);

    if (me == 0) {
      printf("\nTesting matrices of type single complex\n");
    }
    matrix_test(C_SCPL);

#endif
    if (me == 0) {
      printf("\nTesting matrices of type double complex\n");
    }
    matrix_test(C_DCPL);

    if (me == 0) {
      printf("\nSparse matrix tests complete\n\n");
    }
  } else {
    if (me == 0) {
      printf("Test only runs if built with 8-byte integers\n");
    }
  }

  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
