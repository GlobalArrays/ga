#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"


#define NDIM 2048

/**
 *  Solve Laplace's equation on a cubic domain using the sparse matrix
 *  functionality in GA.
 */

/* Create 2D process grid for p processors */
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
 *   first, find all prime numbers, besides 1, less than or equal to p
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

/* Copy a column of data from a matrix (2D array) to a vector (1D array)
 * @param g_m matrix handle
 * @parma g_v vector handle
 * @param lo, hi bounding indices of data from matrix
 */
void copy_to_vector(int g_m, int g_v, int *lo, int *hi)
{
  int vlo[2], vhi[2], ld;
  double *buf;
  int me = NGA_Nodeid();
  NGA_Distribution(g_v,me,vlo,vhi);
  if (vlo[0] < lo[0]) vlo[0] = lo[0];
  if (vhi[0] > hi[0]) vhi[0] = hi[0];
  if (vlo[0] > vhi[0]) return;
  NGA_Access(g_v,vlo,vhi,&buf,&ld);
  vlo[1] = lo[1];
  vhi[1] = hi[1];
  ld = 1;
  NGA_Get(g_m,vlo,vhi,buf,&ld);
  NGA_Release(g_v,vlo,vhi);
}

/* Copy a vector (1D array) to a column in a matrix (2D array)
 * @param g_m matrix handle
 * @parma g_v vector handle
 * @param lo, hi bounding indices of data from matrix
 */
void copy_to_matrix(int g_m, int g_v, int *lo, int *hi)
{
  int vlo[2], vhi[2], ld;
  double *buf;
  int me = NGA_Nodeid();
  NGA_Distribution(g_v,me,vlo,vhi);
  if (vlo[0] < lo[0]) vlo[0] = lo[0];
  if (vhi[0] > hi[0]) vhi[0] = hi[0];
  if (vlo[0] > vhi[0]) return;
  NGA_Access(g_v,vlo,vhi,&buf,&ld);
  vlo[1] = lo[1];
  vhi[1] = hi[1];
  ld = 1;
  NGA_Put(g_m,vlo,vhi,buf,&ld);
  NGA_Release(g_v,vlo,vhi);
}

int main(int argc, char **argv) {
  int g_ran, s_sk, g_a, g_s, g_ss, g_st, g_pm, g_q, g_r, g_c1, g_c2;
  int g_tmp, g_am, g_p, g_qm;
  int dims[2], chunk[2], one, two;
  int lo[2], hi[2], ld[2];
  int i, j, idx, idim, jdim, kdim, ihi, ilo, klo, khi;
  int iter;
  int me, nproc;
  int heap=10000000, stack=10000000;
  double x, snorm, rone;
  double *dptr;
  void *ptr;

  /*Initialize MPI*/
  MP_INIT(argc,argv);
  /* Initialize GA */
  NGA_Initialize();
  heap /= nproc;
  stack /= nproc;
  if(! MA_init(MT_F_DBL, stack, heap))
    GA_Error("MA_init failed",stack+heap);  /* initialize memory allocator*/
  /* initialize random number generator */

  me = GA_Nodeid();
  nproc = GA_Nnodes();
  NGA_Rand(332182);

  idim = NDIM;
  jdim = NDIM;

  if (me == 0) {
    printf("Apply Gram-Schmidt orthoganlization to %d x %d matrix\n",idim,jdim);
  }

  /* Create random 2D matrix of size NDIM x NDIM */
  two = 2;
  dims[0] = idim;
  dims[1] = jdim;
  chunk[0] = idim;
  chunk[1] = -1;
  g_ran = NGA_Create_handle();
  NGA_Set_data(g_ran,two,dims,C_DBL);
  NGA_Set_chunk(g_ran,chunk);
  if (!NGA_Allocate(g_ran))
    GA_Error("Could not allocate random GA",0);

  /* Fill GA with random values */
  NGA_Distribution(g_ran,me,lo,hi);
  NGA_Access(g_ran,lo,hi,&ptr,ld);
  dptr = (double*)ptr;
  ld[0] = hi[0]-lo[0]+1;
  ld[1] = hi[1]-lo[1]+1;
  for (i=0; i<ld[0]; i++) {
    for (j=0; j<ld[1]; j++) {
      idx = i*ld[1]+j;
      dptr[idx] = NGA_Rand(0);
    }
  }

  /* Create sparse matrix representing projection of g_ran onto a sketch matrix */
  kdim = NDIM/10;
  s_sk = NGA_Sprs_array_create(kdim,idim,C_DBL);
  /* Add elements to sparse projection */
  ilo = me*idim/nproc;
  ihi = (me+1)*idim/nproc-1;
  if (me == nproc-1) ihi = idim-1;
  for (i=ilo; i<=ihi; i++) {
    int k;
    k = (int)(((double)kdim)*NGA_Rand(0));
    x = 1.0;
    if (NGA_Rand(0) > 0.5) x = -1.0;
    NGA_Sprs_array_add_element(s_sk, k, i, &x);
  }
  NGA_Sprs_array_assemble(s_sk);

  /* Create Q and R matrices that form the set of orthogonal vectors (Q) and
   * upper triangular matrix (R) from the Gram-Schmidt orthogonalization */
  dims[0] = idim;
  dims[1] = jdim;
  chunk[0] = idim;
  chunk[1] = -1;
  g_q = NGA_Create_handle();
  NGA_Set_data(g_q,two,dims,C_DBL);
  NGA_Set_chunk(g_q,chunk);
  if (!NGA_Allocate(g_q))
    GA_Error("Could not allocate Q matrix",0);
  NGA_Zero(g_q);
  dims[0] = jdim;
  dims[1] = jdim;
  chunk[0] = jdim;
  chunk[1] = -1;
  g_r = NGA_Create_handle();
  NGA_Set_data(g_r,two,dims,C_DBL);
  NGA_Set_chunk(g_r,chunk);
  if (!NGA_Allocate(g_r))
    GA_Error("Could not allocate R matrix",0);
  NGA_Zero(g_r);
  dims[0] = kdim;
  dims[1] = jdim;
  chunk[0] = kdim;
  chunk[1] = -1;
  g_ss = NGA_Create_handle();
  NGA_Set_data(g_ss,two,dims,C_DBL);
  NGA_Set_chunk(g_ss,chunk);
  if (!NGA_Allocate(g_ss))
    GA_Error("Could not allocate S matrix",0);
  NGA_Zero(g_ss);
  dims[0] = kdim;
  dims[1] = 1;
  g_pm = NGA_Create_handle();
  NGA_Set_data(g_pm,two,dims,C_DBL);
  if (!NGA_Allocate(g_pm))
    GA_Error("Could not allocate P matrix",0);
  NGA_Zero(g_pm);
  dims[0] = idim;
  dims[1] = 1;
  g_qm = NGA_Create_handle();
  NGA_Set_data(g_qm,two,dims,C_DBL);
  if (!NGA_Allocate(g_qm))
    GA_Error("Could not allocate P matrix",0);
  NGA_Zero(g_qm);

  /* Create distributed vectors for Gram-Schmidt orthogonalization */
  one = 1;
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, one, &idim, C_DBL);
  if (!NGA_Allocate(g_a))
    GA_Error("Could not allocate A vector",0);
  NGA_Zero(g_a);
  g_s = NGA_Create_handle();
  NGA_Set_data(g_s, one, &kdim, C_DBL);
  if (!NGA_Allocate(g_s))
    GA_Error("Could not allocate s vector",0);
  NGA_Zero(g_s);
  g_st = NGA_Create_handle();
  NGA_Set_data(g_st, one, &kdim, C_DBL);
  if (!NGA_Allocate(g_st))
    GA_Error("Could not allocate s~ vector",0);
  NGA_Zero(g_st);
  g_p = NGA_Create_handle();
  NGA_Set_data(g_p, one, &kdim, C_DBL);
  if (!NGA_Allocate(g_p))
    GA_Error("Could not allocate p vector",0);
  NGA_Zero(g_p);
  dims[0] = jdim;
  dims[1] = 1;
  g_c1 = NGA_Create_handle();
  NGA_Set_data(g_c1, two, dims, C_DBL);
  if (!NGA_Allocate(g_c1))
    GA_Error("Could not allocate c1 vector",0);
  NGA_Zero(g_c1);
  g_c2 = NGA_Create_handle();
  NGA_Set_data(g_c2, two, dims, C_DBL);
  if (!NGA_Allocate(g_c2))
    GA_Error("Could not allocate c2 vector",0);
  NGA_Zero(g_c2);
  g_tmp = NGA_Create_handle();
  NGA_Set_data(g_tmp, two, dims, C_DBL);
  if (!NGA_Allocate(g_tmp))
    GA_Error("Could not allocate tmp vector",0);
  NGA_Zero(g_tmp);
  dims[0] = idim;
  dims[1] = 1;
  g_am = NGA_Create_handle();
  NGA_Set_data(g_am, two, dims, C_DBL);
  if (!NGA_Allocate(g_am))
    GA_Error("Could not allocate am vector",0);
  NGA_Zero(g_am);
  /* Initialize system and calculate first vector */
  lo[0] = 0;
  hi[0] = idim-1;
  lo[1] = 0;
  hi[1] = 0;
  copy_to_vector(g_ran,g_a,lo,hi);
  GA_Sync();
  NGA_Sprs_array_matvec_multiply(s_sk,g_a,g_st);
  GA_Copy(g_st,g_s);
  snorm = GA_Ddot(g_s,g_s);
  snorm = sqrt(snorm);
  lo[0] = 0;
  hi[0] = 0;
  lo[1] = 0;
  hi[1] = 0;
  ld[0] = 1;
  if (me == 0) {
    NGA_Put(g_r,lo,hi,&snorm,ld);
  }
  snorm = 1.0/snorm;
  GA_Scale(g_s,&snorm);
  lo[0] = 0;
  lo[1] = 0;
  copy_to_matrix(g_ss,g_s,lo,hi);
  /* Iterate over remaining columns of g_ran */
  for (iter = 1; iter<jdim; iter++) {
    /* Multiply column of original matrix by sketch projector */
    lo[0] = 0;
    hi[0] = idim-1;
    lo[1] = iter;
    hi[1] = iter;
    copy_to_vector(g_ran,g_a,lo,hi);
    NGA_Sprs_array_matvec_multiply(s_sk,g_a,g_p);
    /* Multiply projected vector by matrix S_i-1 */
    lo[0] = 0;
    hi[0] = jdim-2;
    lo[1] = 0;
    hi[1] = 0;
    copy_to_matrix(g_pm,g_p,lo,hi);
    GA_Dgemm('t','n',jdim,1,kdim,1.0,g_ss,g_pm,0.0,g_c1);
    /* Update projected vector */
    GA_Dgemm('n','n',kdim,1,jdim,-1.0,g_ss,g_c1,1.0,g_pm);
    /* Calculate column in R */
    GA_Dgemm('t','n',jdim,1,kdim,1.0,g_ss,g_pm,0.0,g_c2);
    rone = 1.0; 
    GA_Add(&rone,g_c1,&rone,g_c2,g_tmp);
    /* calculate new orthogonal vector q_i */
    lo[0] = 0;
    hi[0] = idim-1;
    lo[1] = 0;
    hi[1] = 0;
    copy_to_matrix(g_am,g_a,lo,hi);
    GA_Dgemm('n','n',idim,1,jdim,-1.0,g_q,g_tmp,1.0,g_am);
    copy_to_vector(g_am,g_a,lo,hi);
    NGA_Sprs_array_matvec_multiply(s_sk,g_a,g_s);
    snorm = GA_Ddot(g_s,g_s);
    snorm = 1.0/sqrt(snorm);
    GA_Scale(g_s,&snorm);
    GA_Scale(g_a,&snorm);
    lo[0] = 0;
    hi[0] = idim-1;
    lo[1] = iter;
    hi[1] = iter;
    copy_to_matrix(g_a,g_q,lo,hi);
  }
  if (me == 0) {
    printf("Completed Gram-Schmidt orthogonalization\n");
  }
  /* Clean up all global arrays */
  NGA_Sprs_array_destroy(s_sk);
  GA_Destroy(g_ran);
  GA_Destroy(g_a);
  GA_Destroy(g_ss);
  GA_Destroy(g_st);
  GA_Destroy(g_pm);
  GA_Destroy(g_q);
  GA_Destroy(g_r);
  GA_Destroy(g_c1);
  GA_Destroy(g_c2);
  GA_Destroy(g_tmp);
  GA_Destroy(g_am);
  GA_Destroy(g_p);
  GA_Destroy(g_qm);
  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
