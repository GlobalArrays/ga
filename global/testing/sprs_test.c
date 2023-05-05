#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

#define WRITE_VTK
#define CG_SOLVE 1
#define NDIM 1024

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
 * @param a pointer to a regular matrix that is equivalent to s_a
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_matrix(int *s_a, void **a, int64_t dim, int type)
{
  int64_t jlo, jhi, idx; 
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t i, j;
  int64_t skip_len;
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
  *a = malloc(dim*dim*size);
  memset(*a,0,dim*dim*size);

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
   * and 4 others off the diagonl. Final matrix is partitioned into row blocks
   * so this guarantees that sorting routines for elements are tested */
  skip_len = dim/nskip;
  if (skip_len < 2)  {
    nskip = dim/2;
    skip_len = dim/nskip;
  }
  for (j=jlo; j<=jhi; j++) {
    NGA_Sprs_array_add_element64(*s_a,j,j,d_val);
    for (i=0; i<nskip-1; i++) {
      int idx = (j+(i+1)*skip_len)%dim;
      NGA_Sprs_array_add_element64(*s_a,idx,j,o_val);
    }
  }
  /* create local array with same values */
  for (j=0; j<dim; j++) {
      memcpy(*a+size*(j+j*dim),d_val,size);
    for (i=0; i<nskip-1; i++) {
      int idx = (j+(i+1)*skip_len)%dim;
      memcpy(*a+size*(j+idx*dim),o_val,size);
    }
  }

  if (NGA_Sprs_array_assemble(*s_a) && me == 0) {
    printf("\n  Sparse array assembly completed\n\n");
  }
  free(d_val);
  free(o_val);
}

/**
 * subroutine to set up a diagonal matrix for testing purposes
 * @param g_d handle to 1D array representing diagonal matrix
 * @param a pointer to a local array that is equivalent to g_d
 * @param dim dimension of sparse matrix
 * @param type data type used by sparse matrix
 */
void setup_diag_matrix(int *g_d, void **d, int64_t dim, int type)
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

  /* make copy of g_d in local array */
  *d = malloc(size*dim);
  ilo = 0;
  ihi = dim-1;
  NGA_Get64(*g_d,&ilo,&ihi,*d,&ld);

  if (me == 0) {
    printf("\n  Diagonal array completed\n\n");
  }
}

void matrix_test(int type)
{
  int s_a, s_b, s_c, g_d;
  int64_t dim = NDIM;
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
  void *a, *b, *c, *d;
  double tbeg, time;
  
  /* create sparse matrix */
  setup_matrix(&s_a, &a, dim, type);

  /* extract diagonal of s_a to g_d */
  tbeg = GA_Wtime();
  NGA_Sprs_array_get_diag(s_a, &g_d);
  time = GA_Wtime()-tbeg;

  /* check values of g_d to see if they are all 2 */
  NGA_Distribution64(g_d,me,&ilo,&ihi);
  ld = ihi-ilo;
  NGA_Access64(g_d,&ilo,&ihi,&ptr,&ld);
  ok = 1;
  for (i=ilo; i<=ihi; i++) {
    if (type == C_INT) {
      if (((int*)ptr)[i-ilo] != 2) {
        printf("p[%d] diag[%ld]: %d ilo: %ld ihi: %ld\n",me,i,((int*)ptr)[i-ilo],ilo,ihi);
        ok = 0;
      }
    } else if (type == C_LONG) {
      if (((long*)ptr)[i-ilo] != 2) {
        ok = 0;
      }
    } else if (type == C_LONGLONG) {
      if (((long long*)ptr)[i-ilo] != 2) {
        ok = 0;
      }
    } else if (type == C_FLOAT) {
      if (((float*)ptr)[i-ilo] != 2.0) {
        ok = 0;
      }
    } else if (type == C_DBL) {
      if (((double*)ptr)[i-ilo] != 2.0) {
        ok = 0;
      }
    } else if (type == C_SCPL) {
      if (((float*)ptr)[2*(i-ilo)] != 2.0 ||
          ((float*)ptr)[2*(i-ilo)+1] != 0.0) {
        ok = 0;
      }
    } else if (type == C_DCPL) {
      if (((double*)ptr)[2*(i-ilo)] != 2.0 ||
          ((double*)ptr)[2*(i-ilo)+1] != 0.0) {
        ok = 0;
      }
    }
  }
  NGA_Release64(g_d,&ilo,&ihi);
  op[0] = '*';
  op[1] = '\0';
  plus[0] = '+';
  plus[1] = '\0';
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix get diagonal operation PASSES**\n");
      printf("    Time for matrix get diagonal operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix get diagonal operation FAILS**\n");
    }
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

  /* extract diagonal of s_a to g_d */
  NGA_Sprs_array_get_diag(s_a, &g_d);

  /* check values of g_d to see if they are all 3 */
  NGA_Distribution64(g_d,me,&ilo,&ihi);
  ld = ihi-ilo;
  NGA_Access64(g_d,&ilo,&ihi,&ptr,&ld);
  ok = 1;
  for (i=ilo; i<=ihi; i++) {
    if (type == C_INT) {
      if (((int*)ptr)[i-ilo] != 3) {
        ok = 0;
      }
    } else if (type == C_LONG) {
      if (((long*)ptr)[i-ilo] != 3) {
        ok = 0;
      }
    } else if (type == C_LONGLONG) {
      if (((long long*)ptr)[i-ilo] != 3) {
        ok = 0;
      }
    } else if (type == C_FLOAT) {
      if (((float*)ptr)[i-ilo] != 3.0) {
        ok = 0;
      }
    } else if (type == C_DBL) {
      if (((double*)ptr)[i-ilo] != 3.0) {
        ok = 0;
      }
    } else if (type == C_SCPL) {
      if (((float*)ptr)[2*(i-ilo)] != 3.0 ||
          ((float*)ptr)[2*(i-ilo)+1] != 0.0) {
        ok = 0;
      }
    } else if (type == C_DCPL) {
      if (((double*)ptr)[2*(i-ilo)] != 3.0 ||
          ((double*)ptr)[2*(i-ilo)+1] != 0.0) {
        ok = 0;
      }
    }
  }
  NGA_Release64(g_d,&ilo,&ihi);
  op[0] = '*';
  op[1] = '\0';
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("\n    **Sparse matrix shift diagonal operation PASSES**\n");
      printf("    Time for matrix get diagonal operation: %16.8f\n",time);
    } else {
      printf("\n    **Sparse matrix shift diagonal operation FAILS**\n");
    }
  }
  NGA_Destroy(g_d);

  NGA_Sprs_array_destroy(s_a);
  free(shift_val);
  free(a);

  /* Create a fresh copy of sparse matrix */
  setup_matrix(&s_a, &a, dim, type);

  /* Create diagonal matrix */
  setup_diag_matrix(&g_d, &d, dim, type);

  /* Do a right hand multiply */
  tbeg = GA_Wtime();
  NGA_Sprs_array_diag_right_multiply(s_a, g_d);
  time = GA_Wtime()-tbeg;
  /* Do a right hand multiply in regular arrays */
#define MULTIPLY_REAL_AIJ_DJ_M(_a, _d, _i, _j, _type)  \
  {                                                    \
    _type _aij = ((_type*)_a)[_j+dim*_i];              \
    _type _dj  = ((_type*)_d)[_j];                     \
    ((_type*)_a)[_j+dim*_i] = _aij*_dj;                \
  }

#define MULTIPLY_COMPLEX_AIJ_DJ_M(_a, _d, _i, _j, _type)      \
  {                                                           \
    _type _aij_r = ((_type*)_a)[2*(_j+dim*_i)];               \
    _type _aij_i = ((_type*)_a)[2*(_j+dim*_i)+1];             \
    _type _dj_r  = ((_type*)_d)[2*_j];                        \
    _type _dj_i  = ((_type*)_d)[2*_j+1];                      \
    ((_type*)_a)[2*(_j+dim*_i)] = _aij_r*_dj_r-_aij_i*_dj_i;  \
    ((_type*)_a)[2*(_j+dim*_i)+1] = _aij_r*_dj_i+_aij_i*_dj_r;\
  }

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      if (type == C_INT) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, int);
      } else if (type == C_LONG) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, long);
      } else if (type == C_LONGLONG) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, long long);
      } else if (type == C_FLOAT) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, float);
      } else if (type == C_DBL) {
        MULTIPLY_REAL_AIJ_DJ_M(a, d, i, j, double);
      } else if (type == C_SCPL) {
        MULTIPLY_COMPLEX_AIJ_DJ_M(a, d, i, j, float);
      } else if (type == C_DCPL) {
        MULTIPLY_COMPLEX_AIJ_DJ_M(a, d, i, j, double);
      }
    }
  }

#undef MULTIPLY_REAL_AIJ_DJ_M
#undef MULTIPLY_COMPLEX_AIJ_DJ_M
  /* Compare matrix from sparse array operations with
   * local left multiply. Start by getting row block owned by this
   * process */
  ok = 1;
  NGA_Sprs_array_row_distribution64(s_a, me, &ilo, &ihi);
  /* loop over column blocks */
  for (iproc = 0; iproc<nprocs; iproc++) {
    NGA_Sprs_array_column_distribution64(s_a, iproc, &jlo, &jhi);
    if (jhi >= jlo) {
      int64_t nrows = ihi-ilo+1;
      /* column block corresponding to iproc has data. Get pointers
       * to index and data arrays */
      NGA_Sprs_array_access_col_block64(s_a, iproc, &idx, &jdx, &ptr);
      if (idx != NULL) {
        for (i=0; i<nrows; i++) {
          int64_t nvals = idx[i+1]-idx[i];
          for (j=0; j<nvals; j++) {
            if (type == C_INT) {
              if (((int*)ptr)[idx[i]+j]
                  != ((int*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_LONG) {
              if (((long*)ptr)[idx[i]+j]
                  != ((long*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_LONGLONG) {
              if (((long long*)ptr)[idx[i]+j]
                  != ((long long*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_FLOAT) {
              if (((float*)ptr)[idx[i]+j]
                  != ((float*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_DBL) {
              if (((double*)ptr)[idx[i]+j]
                  != ((double*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_SCPL) {
              float rval = ((float*)ptr)[2*(idx[i]+j)];
              float ival = ((float*)ptr)[2*(idx[i]+j)+1];
              float ra = ((float*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])];
              float ia = ((float*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
              if (rval != ra || ival != ia) {
                ok = 0;
              }
            } else if (type == C_DCPL) {
              double rval = ((double*)ptr)[2*(idx[i]+j)];
              double ival = ((double*)ptr)[2*(idx[i]+j)+1];
              double ra = ((double*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])];
              double ia = ((double*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
              if (rval != ra || ival != ia) {
                ok = 0;
              }
            }
          }
        }
      }
    }
  }
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix right diagonal multiply operation PASSES**\n");
      printf("    Time for matrix right diagonal multiply operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix right diagonal multiply operation FAILS**\n");
    }
  }

  NGA_Sprs_array_destroy(s_a);
  free(a);
  free(d);

  /* Create a fresh copy of sparse matrix */
  setup_matrix(&s_a, &a, dim, type);

  /* Create diagonal matrix */
  setup_diag_matrix(&g_d, &d, dim, type);

  /* Do a left hand multiply */
  tbeg = GA_Wtime();
  NGA_Sprs_array_diag_left_multiply(s_a, g_d);
  time = GA_Wtime()-tbeg;
  /* Do a right hand multiply in regular arrays */
#define MULTIPLY_REAL_DI_AIJ_M(_a, _d, _i, _j, _type)  \
  {                                                    \
    _type _aij = ((_type*)_a)[_j+dim*_i];              \
    _type _di  = ((_type*)_d)[_i];                     \
    ((_type*)_a)[_j+dim*_i] = _di*_aij;                \
  }

#define MULTIPLY_COMPLEX_DI_AIJ_M(_a, _d, _i, _j, _type)      \
  {                                                           \
    _type _aij_r = ((_type*)_a)[2*(_j+dim*_i)];               \
    _type _aij_i = ((_type*)_a)[2*(_j+dim*_i)+1];             \
    _type _di_r  = ((_type*)_d)[2*_i];                        \
    _type _di_i  = ((_type*)_d)[2*_i+1];                      \
    ((_type*)_a)[2*(_j+dim*_i)] = _di_r*_aij_r-_di_i*_aij_i;  \
    ((_type*)_a)[2*(_j+dim*_i)+1] = _di_i*_aij_r+_di_r*_aij_i;\
  }

  for (i=0; i<dim; i++) {
    for (j=0; j<dim; j++) {
      if (type == C_INT) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, int);
      } else if (type == C_LONG) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, long);
      } else if (type == C_LONGLONG) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, long long);
      } else if (type == C_FLOAT) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, float);
      } else if (type == C_DBL) {
        MULTIPLY_REAL_DI_AIJ_M(a, d, i, j, double);
      } else if (type == C_SCPL) {
        MULTIPLY_COMPLEX_DI_AIJ_M(a, d, i, j, float);
      } else if (type == C_DCPL) {
        MULTIPLY_COMPLEX_DI_AIJ_M(a, d, i, j, double);
      }
    }
  }

#undef MULTIPLY_REAL_DI_AIJ_M
#undef MULTIPLY_COMPLEX_DI_AIJ_M
  /* Compare matrix from sparse array operations with
   * local left multiply. Start by getting row block owned by this
   * process */
  ok = 1;
  NGA_Sprs_array_row_distribution64(s_a, me, &ilo, &ihi);
  /* loop over column blocks */
  for (iproc = 0; iproc<nprocs; iproc++) {
    NGA_Sprs_array_column_distribution64(s_a, iproc, &jlo, &jhi);
    if (jhi >= jlo) {
      int64_t nrows = ihi-ilo+1;
      /* column block corresponding to iproc has data. Get pointers
       * to index and data arrays */
      NGA_Sprs_array_access_col_block64(s_a, iproc, &idx, &jdx, &ptr);
      if (idx != NULL) {
        for (i=0; i<nrows; i++) {
          int64_t nvals = idx[i+1]-idx[i];
          for (j=0; j<nvals; j++) {
            if (type == C_INT) {
              if (((int*)ptr)[idx[i]+j] != ((int*)a)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_LONG) {
              if (((long*)ptr)[idx[i]+j] != ((long*)a)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_LONGLONG) {
              if (((long long*)ptr)[idx[i]+j] 
                  != ((long long*)a)[(i+ilo)*dim + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_FLOAT) {
              if (((float*)ptr)[idx[i]+j] != ((float*)a)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_DBL) {
              if (((double*)ptr)[idx[i]+j] != ((double*)a)[(i+ilo)*dim
                  + jdx[idx[i]+j]]) {
                ok = 0;
              }
            } else if (type == C_SCPL) {
              float rval = ((float*)ptr)[2*(idx[i]+j)];
              float ival = ((float*)ptr)[2*(idx[i]+j)+1];
              float ra = ((float*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])];
              float ia = ((float*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
              if (rval != ra || ival != ia) {
                ok = 0;
              }
            } else if (type == C_DCPL) {
              double rval = ((double*)ptr)[2*(idx[i]+j)];
              double ival = ((double*)ptr)[2*(idx[i]+j)+1];
              double ra = ((double*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])];
              double ia = ((double*)a)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
              if (rval != ra || ival != ia) {
                ok = 0;
              }
            }
          }
        }
      }
    }
  }
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix left diagonal multiply operation PASSES**\n");
      printf("    Time for matrix left diagonal multiply operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix left diagonal multiply operation FAILS**\n");
    }
  }

  NGA_Sprs_array_destroy(s_a);
  free(a);
  free(d);

  /* create sparse matrix A */
  setup_matrix(&s_a, &a, dim, type);

  nz_map = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) nz_map[i] = 0;
  time = 0.0;
  for (k=0; k<nprocs; k++) {
    for (l=0; l<nprocs; l++) {
      tbeg = GA_Wtime();
      NGA_Sprs_array_get_block64(s_a, k, l, &idx, &jdx, &ptr,
          &ilo, &ihi, &jlo, &jhi);
      time += GA_Wtime()-tbeg;
      /*
      printf("p[%d] block [%d,%d] ilo: %d ihi: %d jlo: %d jhi: %d\n",
      me,k,l,ilo,ihi,jlo,jhi);
      */
      /* check for correctness */
      ok = 1;
      for (i=ilo; i<=ihi; i++) {
        int64_t jcols = idx[i+1-ilo]-idx[i-ilo];
        /*
        printf("p[%d]     row: %d jcols: %d\n",me,i,jcols);
        */

        for (j=0; j<jcols; j++) {
          nz_map[i*dim+jdx[idx[i-ilo]+j]] = 1;
          /*
        printf("p[%d]         row: %d jcols: %d col: %d\n",me,i,jcols,
            jdx[idx[i-ilo]+j]+jlo);
            */
          if (type == C_INT) {
            if (((int*)ptr)[idx[i-ilo]+j]
                != ((int*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
              if (ok) printf("(int) block [%d,%d] element i: %d j: %d"
                  " expected: %d actual: %d\n",
                  me,k,i,jdx[idx[i-ilo]+j],((int*)a)[i*dim+jdx[idx[i-ilo]+j]],
                  ((int*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          } else if (type == C_LONG) {
            if (((long*)ptr)[idx[i-ilo]+j]
                != ((long*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
              printf("(long) block [%d,%d] element i: %d j: %d"
                  " expected: %ld actual: %ld\n",
                  me,k,i,jdx[idx[i-ilo]+j],((long*)a)[i*dim
                  +jdx[idx[i-ilo]+j]], ((long*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          } else if (type == C_LONGLONG) {
            if (((long long*)ptr)[idx[i-ilo]+j] != ((long long*)a)[i*dim
                +jdx[idx[i-ilo]+j]]) {
              printf("(long long) block [%d,%d] element i: %d j: %d"
                  " expected: %ld actual: %ld\n",
                  me,k,i,jdx[idx[i-ilo]+j],
                  (long)(((long long*)a)[i*dim+jdx[idx[i-ilo]+j]]),
                  (long)(((long long*)ptr)[idx[i-ilo]+j]));
              ok = 0;
            }
          } else if (type == C_FLOAT) {
            if (((float*)ptr)[idx[i-ilo]+j] 
                != ((float*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
              printf("(float) block [%d,%d] element i: %d j: %d"
                  " expected: %f actual: %f\n",
                  me,k,i,jdx[idx[i-ilo]+j],((float*)a)[i*dim
                  +jdx[idx[i-ilo]+j]], ((float*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          } else if (type == C_DBL) {
            if (((double*)ptr)[idx[i-ilo]+j]
                != ((double*)a)[i*dim+jdx[idx[i-ilo]+j]]) {
              printf("(double) block [%d,%d] element i: %d j: %d"
                  " expected: %f actual: %f\n",
                  me,k,i,jdx[idx[i-ilo]+j],((double*)a)[i*dim
                  +jdx[idx[i-ilo]+j]], ((double*)ptr)[idx[i-ilo]+j]);
              ok = 0;
            }
          } else if (type == C_SCPL) {
            if (((float*)ptr)[2*(idx[i-ilo]+j)] 
                != ((float*)a)[2*(i*dim+jdx[idx[i-ilo]+j])] ||
                ((float*)ptr)[2*(idx[i-ilo]+j)+1] 
                != ((float*)a)[2*(i*dim+jdx[idx[i-ilo]+j])+1]) {
              printf("(single complex) block [%d,%d] element i: %d j: %d"
                  " expected: (%f,%f) actual: (%f,%f)\n",
                  me,k,i,jdx[idx[i-ilo]+j],
                  ((float*)a)[2*(i*dim+jdx[idx[i-ilo]+j])],
                  ((float*)a)[2*(i*dim+jdx[idx[i-ilo]+j])+1],
                  ((float*)ptr)[2*(idx[i-ilo]+j)],
                  ((float*)ptr)[2*(idx[i-ilo]+j)+1]);
              ok = 0;
            }
          } else if (type == C_DCPL) {
            if (((double*)ptr)[2*(idx[i-ilo]+j)] 
                != ((double*)a)[2*(i*dim+jdx[idx[i-ilo]+j])] ||
                ((double*)ptr)[2*(idx[i-ilo]+j)+1] 
                != ((double*)a)[2*(i*dim+jdx[idx[i-ilo]+j])+1]) {
              printf("(double complex) block [%d,%d] element i: %d j: %d"
                  " expected: (%f,%f) actual: (%f,%f)\n",
                  me,k,i,jdx[idx[i-ilo]+j],
                  ((double*)a)[2*(i*dim+jdx[idx[i-ilo]+j])],
                  ((double*)a)[2*(i*dim+jdx[idx[i-ilo]+j])+1],
                  ((double*)ptr)[2*(idx[i-ilo]+j)],
                  ((double*)ptr)[2*(idx[i-ilo]+j)+1]);
              ok = 0;
            }
          }
        }
      }
      if (idx != NULL) free(idx);
      if (idx != NULL) free(jdx);
      if (idx != NULL) free(ptr);
    }
  }
  for (i=0; i<dim*dim; i++) {
    if (type == C_INT) {
      if (((int*)a)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_LONG) {
      if (((long*)a)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_LONGLONG) {
      if (((long long*)a)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_FLOAT) {
      if (((float*)a)[i] != 0.0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_DBL) {
      if (((double*)a)[i] != 0.0 && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_SCPL) {
      if ((((float*)a)[2*i] != 0.0 || ((float*)a)[2*i+1] != 0.0)
          && nz_map[i] == 0) {
        ok = 0;
      }
    } else if (type == C_SCPL) {
      if ((((double*)a)[2*i] != 0.0 || ((double*)a)[2*i+1] != 0.0)
          && nz_map[i] == 0) {
        ok = 0;
      }
    }
  }
  free(nz_map);

  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)(nprocs*nprocs*nprocs);
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix get block operation PASSES**\n");
      printf("    Time for matrix get block operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix get block operation FAILS**\n");
    }
  }

  NGA_Sprs_array_destroy(s_a);
  free(a);

  /* create sparse matrix A */
  setup_matrix(&s_a, &a, dim, type);
  /* create sparse matrix B */
  setup_matrix(&s_b, &b, dim, type);
#if 0
  if (me == 0 && type == C_INT) {
    printf("\nMatrix A\n");
    for (i=0; i<dim; i++) {
      for(j=0; j<dim; j++) {
        printf(" %3d",((int*)a)[j+dim*i]);
      }
      printf("\n");
    }
    printf("\nMatrix B\n");
    for (i=0; i<dim; i++) {
      for(j=0; j<dim; j++) {
        printf(" %3d",((int*)b)[j+dim*i]);
      }
      printf("\n");
    }
  }
#endif


  /* multiply sparse matrix A times sparse matrix B */
  tbeg = GA_Wtime();
  s_c = NGA_Sprs_array_matmat_multiply(s_a, s_b);
  time = GA_Wtime()-tbeg;

#if 1
  /* Do regular matrix-matrix multiply of A and B */
  if (type == C_INT) {
    c = malloc(dim*dim*sizeof(int));
  } else if (type == C_LONG) {
    c = malloc(dim*dim*sizeof(long));
  } else if (type == C_LONGLONG) {
    c = malloc(dim*dim*sizeof(long long));
  } else if (type == C_FLOAT) {
    c = malloc(dim*dim*sizeof(float));
  } else if (type == C_DBL) {
    c = malloc(dim*dim*sizeof(double));
  } else if (type == C_SCPL) {
    c = malloc(dim*dim*2*sizeof(float));
  } else if (type == C_DCPL) {
    c = malloc(dim*dim*2*sizeof(double));
  }

#define REAL_MATMAT_MULTIPLY_M(_type, _a, _b, _c, _dim)     \
{                                                           \
  int _i, _j, _k;                                           \
  _type *_aa = (_type*)_a;                                  \
  _type *_bb = (_type*)_b;                                  \
  _type *_cc = (_type*)_c;                                  \
  for (_i=0; _i<_dim; _i++) {                               \
    for (_j=0; _j<_dim; _j++) {                             \
      _cc[_j+_i*_dim] = (_type)0;                           \
      for(_k=0; _k<_dim; _k++) {                            \
        _cc[_j+_i*_dim] += _aa[_k+_i*_dim]*_bb[_j+_k*_dim]; \
      }                                                     \
    }                                                       \
  }                                                         \
}

#define COMPLEX_MATMAT_MULTIPLY_M(_type, _a, _b, _c, _dim)  \
{                                                           \
  int _i, _j, _k;                                           \
  _type *_aa = (_type*)_a;                                  \
  _type *_bb = (_type*)_b;                                  \
  _type *_cc = (_type*)_c;                                  \
  _type _ar, _ai, _br, _bi;                                 \
  for (_i=0; _i<_dim; _i++) {                               \
    for (_j=0; _j<_dim; _j++) {                             \
      _cc[2*(_j+_i*_dim)] = (_type)0;                       \
      _cc[2*(_j+_i*_dim)+1] = (_type)0;                     \
      for(_k=0; _k<_dim; _k++) {                            \
        _ar = _aa[2*(_k+_i*_dim)];                          \
        _ai = _aa[2*(_k+_i*_dim)+1];                        \
        _br = _bb[2*(_j+_k*_dim)];                          \
        _bi = _bb[2*(_j+_k*_dim)+1];                        \
        _cc[2*(_j+_i*_dim)] += _ar*_br-_ai*_bi;             \
        _cc[2*(_j+_i*_dim)+1] += _ar*_bi+_ai*_br;           \
      }                                                     \
    }                                                       \
  }                                                         \
}

  if (type == C_INT) {
    REAL_MATMAT_MULTIPLY_M(int, a, b, c, dim);
  } else if (type == C_LONG) {
    REAL_MATMAT_MULTIPLY_M(long, a, b, c, dim);
  } else if (type == C_LONGLONG) {
    REAL_MATMAT_MULTIPLY_M(long long, a, b, c, dim);
  } else if (type == C_FLOAT) {
    REAL_MATMAT_MULTIPLY_M(float, a, b, c, dim);
  } else if (type == C_DBL) {
    REAL_MATMAT_MULTIPLY_M(double, a, b, c, dim);
  } else if (type == C_SCPL) {
    COMPLEX_MATMAT_MULTIPLY_M(float, a, b, c, dim);
  } else if (type == C_DCPL) {
    COMPLEX_MATMAT_MULTIPLY_M(double, a, b, c, dim);
  }
#if 0
  if (me == 0 && type == C_INT) {
    printf("\nMatrix C\n");
    for (i=0; i<dim; i++) {
      for(j=0; j<dim; j++) {
        printf(" %3d",((int*)c)[j+dim*i]);
      }
      printf("\n");
    }
  }
#endif

#undef REAL_MATMAT_MULTIPLY_M
#undef COMPLEX_MATMAT_MULTIPLY_M

  int *ab = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) ab[i] = 0;
  nz_map = (int*)malloc(dim*dim*sizeof(int));
  for (i=0; i<dim*dim; i++) nz_map[i] = 0;
  /* Compare results from regular matrix-matrix multiply with
   * sparse matrix-matrix multiply */
  ok = 1;
  NGA_Sprs_array_row_distribution64(s_c, me, &ilo, &ihi);
  /* loop over column blocks */
  ld = 0;
  for (iproc = 0; iproc<nprocs; iproc++) {
    int64_t nrows = ihi-ilo+1;
    /* column block corresponding to iproc has data. Get pointers
     * to index and data arrays */
    NGA_Sprs_array_access_col_block64(s_c, iproc, &idx, &jdx, &ptr);
    /*
        printf("p[%d] iproc: %ld idx: %p jdx: %p ptr: %p\n",me,iproc,idx,jdx,ptr);
        */
    if (idx != NULL) {
      for (i=0; i<nrows; i++) {
        int64_t nvals = idx[i+1]-idx[i];
        /*
        printf("p[%d] iproc: %ld i: %ld nrows: %ld nvals: %ld\n",me,
            iproc,i+ilo,nrows,nvals);
            */
        for (j=0; j<nvals; j++) {
          /*
        printf("p[%d]     iproc: %ld i: %ld j: %ld nvals: %ld\n",me,
            iproc,i+ilo,jdx[idx[i]+j],nvals);
            */
          ld++;
          nz_map[(i+ilo)*dim+jdx[idx[i]+j]] = 1;
          if (type == C_INT) {
            /*
            printf("p[%d]        proc: %ld i: %ld j: %ld c: %d\n",me,
                iproc,i+ilo,jdx[idx[i]+j],((int*)ptr)[idx[i]+j]);
                */
            ab[(i+ilo)*dim+jdx[idx[i]+j]] = ((int*)ptr)[idx[i]+j];
            if (((int*)ptr)[idx[i]+j] != ((int*)c)[(i+ilo)*dim+jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %d actual: %d\n",me,
                    i+ilo,jdx[idx[i]+j],((int*)c)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((int*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONG) {
            if (((long*)ptr)[idx[i]+j] != ((long*)c)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],((long*)c)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_LONGLONG) {
            if (((long long*)ptr)[idx[i]+j] 
                != ((long long*)c)[(i+ilo)*dim + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %ld actual: %ld\n",me,
                    i+ilo,jdx[idx[i]+j],(long)((long long*)c)[(i+ilo)*dim
                    +jdx[idx[i]+j]],
                    (long)((long long*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_FLOAT) {
            if (((float*)ptr)[idx[i]+j] != ((float*)c)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((float*)c)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((float*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_DBL) {
            if (((double*)ptr)[idx[i]+j] != ((double*)c)[(i+ilo)*dim
                + jdx[idx[i]+j]]) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: %f actual: %f\n",me,
                    i+ilo,jdx[idx[i]+j],((double*)c)[(i+ilo)*dim+jdx[idx[i]+j]],
                    ((double*)ptr)[idx[i]+j]);
              }
              ok = 0;
            }
          } else if (type == C_SCPL) {
            float rval = ((float*)ptr)[2*(idx[i]+j)];
            float ival = ((float*)ptr)[2*(idx[i]+j)+1];
            float ra = ((float*)c)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            float ia = ((float*)c)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          } else if (type == C_DCPL) {
            double rval = ((double*)ptr)[2*(idx[i]+j)];
            double ival = ((double*)ptr)[2*(idx[i]+j)+1];
            double ra = ((double*)c)[2*((i+ilo)*dim + jdx[idx[i]+j])];
            double ia = ((double*)c)[2*((i+ilo)*dim + jdx[idx[i]+j])+1];
            if (rval != ra || ival != ia) {
              if (ok) {
                printf("p[%d] [%ld,%ld] expected: (%f,%f) actual: (%f,%f)\n",me,
                    i+ilo,jdx[idx[i]+j],ra,ia,rval,ival);
              }
              ok = 0;
            }
          }
        }
      }
    }
  }
#if 0
  if (me == 0 && type == C_INT) {
    printf("\nMatrix NZ_MAP\n");
    for (i=0; i<dim; i++) {
      for(j=0; j<dim; j++) {
        printf(" %3d",((int*)nz_map)[j+dim*i]);
      }
      printf("\n");
    }
  }
#endif
  /* Only do non-zero check for integers */
  if (type == C_INT) {
    for (i=0; i<dim*dim; i++) {
      if (((int*)ab)[i] != 0 && nz_map[i] == 0) {
        ok = 0;
      }
    }
  }
#if 0
  if (type == C_INT) {
    GA_Igop(ab,(int)(dim*dim),"+");
  }
  if (me == 0 && type == C_INT) {
    printf("\nMatrix AB\n");
    for (i=0; i<dim; i++) {
      for(j=0; j<dim; j++) {
        printf(" %3d",ab[j+dim*i]);
      }
      printf("\n");
    }
  }
#endif
  free(ab);
  GA_Igop(&ok,1,op);
  GA_Dgop(&time,1,plus);
  time /= (double)nprocs;
  if (me == 0) {
    if (ok) {
      printf("    **Sparse matrix-matrix multiply operation PASSES**\n");
      printf("    Time for matrix-matrix multiply operation: %16.8f\n",time);
    } else {
      printf("    **Sparse matrix-matrix multiply operation FAILS**\n");
    }
  }
  free(a);
  free(b);
  free(c);
  free(nz_map);
#endif
  NGA_Sprs_array_destroy(s_a);
  NGA_Sprs_array_destroy(s_b);

}

int main(int argc, char **argv) {
  int me,nproc;

  /**
   * Initialize GA
   */
  MP_INIT(argc,argv);

  /* Initialize GA */
  NGA_Initialize();

  me = GA_Nodeid();
  nproc = GA_Nnodes();
  if (me == 0) {
    printf("\nTesting sparse matrices of size %d x %d on %d processors\n\n",
        NDIM,NDIM,nproc);
  }

  /**
   * Test different data types
   */
  if (me == 0) {
    printf("\nTesting matrices of type int\n");
  }
  matrix_test(C_INT);
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

  if (me == 0) {
    printf("\nTesting matrices of type double complex\n");
  }
  matrix_test(C_DCPL);
#endif
  if (me == 0) {
    printf("\nSparse matrix tests complete\n\n");
  }

  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
