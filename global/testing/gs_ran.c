#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"


#define NDIM 8

#define GA_PRINT

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
  int g_tmp, g_am, g_p, g_qm, g_km;
  int dims[2], chunk[2], one, two;
  int lo[2], hi[2], ld[2];
  int i, j, idx, idim, jdim, kdim, ihi, ilo, klo, khi;
  int it, jt;
  int iter;
  int me, nproc;
  int heap=10000000, stack=10000000;
  double x, snorm, rone,rmone;
  double *dptr;
  double dmin, dmax, omin, omax;
  void *ptr;
  char cmax[4],cmin[4];

  /*Initialize MPI*/
  MP_INIT(argc,argv);
  /* Initialize GA */
  NGA_Initialize();
  me = GA_Nodeid();
  nproc = GA_Nnodes();
  heap /= nproc;
  stack /= nproc;
  if(! MA_init(MT_F_DBL, stack, heap))
    GA_Error("MA_init failed",stack+heap);  /* initialize memory allocator*/
  /* initialize random number generator */

  NGA_Rand(332182+me);
  //NGA_Rand(432182+me);

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
      dptr[idx] = 2.0*NGA_Rand(0)-1.0;
    }
  }
  NGA_Release(g_ran,lo,hi);
#ifdef GA_PRINT
  if (me == 0) printf("Original random matrix\n");
  GA_Print(g_ran);
#endif

  /* Create sparse matrix representing projection of g_ran onto a sketch matrix */
  kdim = NDIM/2;
  s_sk = NGA_Sprs_array_create(kdim,idim,C_DBL);
  /* Add elements to sparse projection */
  ilo = me*idim/nproc;
  ihi = (me+1)*idim/nproc-1;
  if (me == nproc-1) ihi = idim-1;
#if 0
  for (i=ilo; i<=ihi; i++) {
    int k;
    k = (int)(((double)kdim)*NGA_Rand(0));
    x = 1.0;
    double rx = NGA_Rand(0);
    if (rx > 0.5) x = -1.0;
    /* printf("p[%d] add element %d %d %f %f\n",me,k,i,x,rx); */
    NGA_Sprs_array_add_element(s_sk, k, i, &x);
  }
#else
  {
    int *kbuf = (int*)malloc(idim*sizeof(int));
    int kcnt;
    for (i=0; i<idim; i++) kbuf[i] = -1;
    if (me == 0) {
      kcnt = 0;
      while (kcnt < kdim) {
        i = (int)(((double)idim)*NGA_Rand(0));
        if (kbuf[i] == -1) {
          kbuf[i] = kcnt;
          kcnt++;
        }
      }
    }
    strcpy(cmax,"max");
    GA_Igop(kbuf,idim,cmax);
    if (me == 0) {
      for (i=0; i<idim; i++) {
        printf("p[%d] kbuf[%d]: %d\n",me,i,kbuf[i]);
      }
    }
    for (i=ilo; i<=ihi; i++) {
      int k;
      if (kbuf[i] > -1) {
        k = kbuf[i];
      } else {
        k = (int)(((double)kdim)*NGA_Rand(0));
      }
      x = 1.0;
      double rx = NGA_Rand(0);
      if (rx > 0.5) x = -1.0;
      printf("p[%d] add element %d %d %f %f\n",me,k,i,x,rx);
      NGA_Sprs_array_add_element(s_sk, k, i, &x);
    }
    free(kbuf);
  }
#endif
  NGA_Sprs_array_assemble(s_sk);
#ifdef GA_PRINT
  NGA_Sprs_array_export(s_sk,"sketch.m");
#endif

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
    GA_Error("Could not allocate Q matrix",0);
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
  dims[0] = kdim;
  dims[1] = 1;
  g_km = NGA_Create_handle();
  NGA_Set_data(g_km, two, dims, C_DBL);
  if (!NGA_Allocate(g_km))
    GA_Error("Could not allocate km vector",0);
  NGA_Zero(g_am);
  /* Initialize system and calculate first vector */
  lo[0] = 0;
  hi[0] = idim-1;
  lo[1] = 0;
  hi[1] = 0;
  copy_to_vector(g_ran,g_a,lo,hi);
  GA_Sync();
#ifdef GA_PRINT
  if (me == 0) printf("Initial vector a_1\n");
  GA_Print(g_a);
#endif
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
  hi[0] = kdim-1;
  lo[1] = 0;
  hi[1] = 0;
  copy_to_matrix(g_ss,g_s,lo,hi);
#ifdef GA_PRINT
  if (me == 0) printf("Value of S_1\n");
  GA_Print(g_ss);
#endif
  GA_Scale(g_a,&snorm);
#ifdef GA_PRINT
  if (me == 0) printf("Value of q_1\n");
  GA_Print(g_a);
#endif
  lo[0] = 0;
  hi[0] = idim-1;
  copy_to_matrix(g_q,g_a,lo,hi);
#ifdef GA_PRINT
  if (me == 0) printf("Initial Q\n");
  GA_Print(g_q);
#endif
  /* Iterate over remaining columns of g_ran */
  rone = 1.0; 
  rmone = -1.0; 
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
    hi[0] = jdim-1;
    lo[1] = 0;
    hi[1] = 0;
    copy_to_matrix(g_pm,g_p,lo,hi);
#ifdef GA_PRINT
    if (me == 0) printf("Value of p_%d\n",iter+1);
    GA_Print(g_pm);
    if (me == 0) printf("Value of Sm_%d\n",iter+1);
    GA_Print(g_ss);
#endif

    GA_Dgemm('t','n',jdim,1,kdim,1.0,g_ss,g_pm,0.0,g_c1);
#ifdef GA_PRINT
    if (me == 0) printf("Value of c1 (%d)\n",iter+1);
    GA_Print(g_c1);
#endif
    /* Update projected vector */
    GA_Dgemm('n','n',kdim,1,jdim,-1.0,g_ss,g_c1,1.0,g_pm);
#ifdef GA_PRINT
    if (me == 0) printf("Value of pm_%d\n handle: %d",iter+1,g_pm);
    GA_Print(g_pm);
#endif
    /* Calculate column in R */
    GA_Dgemm('t','n',jdim,1,kdim,1.0,g_ss,g_pm,0.0,g_c2);
    GA_Sync();
#ifdef GA_PRINT
    if (me == 0) printf("Value of c2 (%d) handle: %d\n",iter+1,g_c2);
    GA_Print(g_c2);
#endif
    GA_Add(&rone,g_c1,&rone,g_c2,g_tmp);
#ifdef GA_PRINT
    if (me == 0) printf("Value of c1+c2 (%d)\n",iter+1);
    GA_Print(g_tmp);
#endif
    lo[0] = 0;
    hi[0] = idim-1;
    lo[1] = iter;
    hi[1] = iter;
    copy_to_matrix(g_r,g_tmp,lo,hi);
    /* calculate new orthogonal vector q_i */
    lo[0] = 0;
    hi[0] = idim-1;
    lo[1] = 0;
    hi[1] = 0;
    copy_to_matrix(g_am,g_a,lo,hi);
    GA_Dgemm('n','n',idim,1,jdim,-1.0,g_q,g_tmp,1.0,g_am);
#ifdef GA_PRINT
    if (me == 0) printf("Value of am_%d\n",iter+1);
    GA_Print(g_am);
#endif
    copy_to_vector(g_am,g_a,lo,hi);
    NGA_Sprs_array_matvec_multiply(s_sk,g_a,g_s);
#ifdef GA_PRINT
    if (me == 0) printf("Value of (1) S_%d\n",iter+1);
    GA_Print(g_s);
#endif
    snorm = GA_Ddot(g_s,g_s);
    snorm  = sqrt(snorm);
    NGA_Distribution(g_r,me,lo,hi);
    if (iter >= lo[0] && iter <= hi[0] && iter >= lo[1] && iter <= hi[1]) {
      lo[0] = iter;
      hi[0] = iter;
      lo[1] = iter;
      hi[1] = iter;
      NGA_Put(g_r,lo,hi,&snorm,&one);
    }
#ifdef GA_PRINT
    if (me == 0) printf("Value of R_%d\n",iter+1);
    GA_Print(g_r);
#endif
    snorm = 1.0/snorm;
    if (me == 0) printf("sqrt(S.S)[%d]: %e\n",iter+1,snorm);
    GA_Scale(g_s,&snorm);
#ifdef GA_PRINT
    if (me == 0) printf("Value of (2) S_%d\n",iter+1);
    GA_Print(g_s);
#endif
    lo[0] = 0;
    hi[0] = kdim-1;
    lo[1] = iter;
    hi[1] = iter;
    copy_to_matrix(g_ss,g_s,lo,hi);
    GA_Scale(g_a,&snorm);
    lo[0] = 0;
    hi[0] = idim-1;
    lo[1] = iter;
    hi[1] = iter;
#ifdef GA_PRINT
//  GA_Print(g_a);
#endif
    copy_to_matrix(g_q,g_a,lo,hi);
#ifdef GA_PRINT
//  GA_Print(g_q);
#endif
  }
  if (me == 0) {
    printf("Completed Gram-Schmidt orthogonalization\n");
  }
  /* Clean up all global arrays, except those used for verification */
  NGA_Sprs_array_destroy(s_sk);
  GA_Destroy(g_ran);
  GA_Destroy(g_s);
  GA_Destroy(g_ss);
  GA_Destroy(g_st);
  GA_Destroy(g_pm);
  GA_Destroy(g_c1);
  GA_Destroy(g_c2);
  GA_Destroy(g_tmp);
  GA_Destroy(g_am);
  GA_Destroy(g_p);
  GA_Destroy(g_qm);

  g_s = NGA_Create_handle();
  NGA_Set_data(g_s, one, &idim, C_DBL);
  if (!NGA_Allocate(g_s))
    GA_Error("Could not allocate second S vector",0);
  NGA_Zero(g_s);
  g_qm = NGA_Create_handle();
  dims[0] = idim;
  dims[1] = idim;
  NGA_Set_data(g_qm, two, dims, C_DBL);
  if (!NGA_Allocate(g_qm))
    GA_Error("Could not allocate matrix of dot products",0);
  NGA_Zero(g_qm);
#if 0
  for (i=0; i<idim; i++) {
    lo[0] = 0;
    hi[0] = idim;
    lo[1] = i;
    hi[1] = i;
    copy_to_vector(g_q,g_a,lo,hi);
    for (j=0; j<idim; j++) {
      lo[0] = 0;
      hi[0] = idim;
      lo[1] = j;
      hi[1] = j;
      copy_to_vector(g_q,g_s,lo,hi);
      snorm = GA_Ddot(g_a,g_s);
      lo[0] = i;
      hi[0] = i;
      lo[1] = j;
      hi[1] = j;
      if ((i*idim+j)%nproc == me) {
        NGA_Put(g_qm,lo,hi,&snorm,&one);
      }
    }
  }
  GA_Sync();
#else
  GA_Dgemm('t','n',idim,idim,idim,1.0,g_q,g_q,0.0,g_qm);
#endif
//  GA_Print(g_qm);
  /* Check min and max values of dot product matrix elements for diagonal
   * and off diagonal elements. Start by everyone initializing to element
   * 0,0 and 0,1 */
  lo[0] = 0;
  hi[0] = 0;
  lo[1] = 0;
  hi[1] = 0;
  NGA_Get(g_qm,lo,hi,&snorm,&one);
  dmin = snorm;
  dmax = snorm;
  lo[1] = 1;
  hi[1] = 1;
  NGA_Get(g_qm,lo,hi,&snorm,&one);
  omin = snorm;
  omax = snorm;
  /* Get pointers to local data */
  NGA_Distribution(g_qm,me,lo,hi);
  NGA_Access(g_qm,lo,hi,&dptr,ld);

  it = hi[0]-lo[0]+1;
  jt = hi[1]-lo[1]+1;
  for (i=0; i<it; i++) {
    for (j=0; j<jt; j++) {
      idx = j+i*ld[0];
      if (i+lo[0] == j+lo[1]) {
        if (dptr[idx] > dmax) dmax = dptr[idx];
        if (dptr[idx] < dmin) dmin = dptr[idx];
      } else {
        if (dptr[idx] > omax) omax = dptr[idx];
        if (dptr[idx] < omin) omin = dptr[idx];
      }
    }
  }
  NGA_Release(g_qm,lo,hi);
  strcpy(cmax,"max");
  strcpy(cmin,"min");
  GA_Dgop(&omin,1,cmin);
  GA_Dgop(&omax,1,cmax);
  GA_Dgop(&dmin,1,cmin);
  GA_Dgop(&dmax,1,cmax);
  if (me == 0) {
    printf("Minimum and maximum values of off-diagonal elements %16.8e %16.8e\n",
        omin,omax);
    printf("Minimum and maximum values of diagonal elements     %16.8e %16.8e\n",
        dmin,dmax);
  }

  /* Clean up remaining global arrays */
  GA_Destroy(g_a);
  GA_Destroy(g_q);
  GA_Destroy(g_r);
  GA_Destroy(g_s);
  GA_Destroy(g_km);
  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
