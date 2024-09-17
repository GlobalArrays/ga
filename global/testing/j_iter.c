#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

#define OLD_XDIST

#define SYNCHRONOUS 0

#define WRITE_VTK
#define NDIM 64
#define MAX_ITERATIONS 500

#define NUM_BINS 40

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

/**
 * Subroutine for doing a serial sparse matrix (CSR format) matrix-vector
 * multiply
 * @param ilo,ihi bounding indices for rows
 * @param jlo,jhi bounding indices for columns
 * @param idx array containing bounds for each set of column indices
 * @param jdx array containing column indices
 * @param vals array containing matrix values
 * @param x array containing vector values
 * @param ax array containing matrix vector product
 */
void lmatvec(int64_t ilo, int64_t ihi, int64_t jlo, int64_t jhi, int64_t *idx,
    int64_t *jdx, double *vals, double *x, double *ax)
{
  int64_t i, j;
  int64_t ncols;
  /* don't bother to initialize inner product to zero. Assume that this has
   * already been done or you are accumulating results */
  for (i=ilo; i<=ihi; i++) {
    ncols = idx[i+1-ilo]-idx[i-ilo];
    for (j=0; j<ncols; j++) {
      ax[i-ilo] += vals[idx[i-ilo]+j]*x[jdx[idx[i-ilo]+j]-jlo];
    }
  }
}

/**
 * Return estimate of dot product of two vectors
 * @param g_dot handle of global array holding partial results from all
 *              processes
 * @param dot_ptr pointer to local portion of g_dot
 * @param nproc number of processors in the system
 * @param nvals number of locally held elements in vectors
 * @param x, y arrays holding vector values
 * @param sync if true, synchronize dot product
 */
double ldot(int g_dot, double *dot_ptr, int nproc, int64_t nvals, double *x, double *y, int sync)
{
  int64_t i;
  double *buf = (double*)malloc(nproc*sizeof(double));
  double acc = 0.0;
  double ret = 0.0;
  int lo = 0;
  int hi = nproc-1;
  int ld = 1;
  for (i=0; i<nvals; i++) {
    acc += x[i]*y[i];
  }
  *dot_ptr = acc;
  if (sync) GA_Sync();
  NGA_Get(g_dot,&lo,&hi,buf,&ld);
  for (i=0; i<(int64_t)nproc; i++) {
    ret += buf[i];
  }
  free(buf);
  return ret;
}

/**
 * Iterative asynchronous Jacobi solver
 * @param s_a handle for sparse matrix
 * @param g_b handle for right hand side vector
 * @param g_ref handle for reference solution calculated using CG
 * @param g_x handle for solution vector
 */
void j_solve(int s_a, int g_b, int g_ref, int *g_x)
{
  int i, j, nb, icnt, g_cvg, g_maxr, g_c, g_d, s_j;
  int g_p;
  int nblocks;
  int *list;
  int my_nb;
  int *offblocks;
  int converged = 0;
  int converged_tracker = 0;
  int recheck_converged = 0;
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  double residual;
  int64_t *idx;
  int64_t *jdx;
  double *xbuf;
  double *rbuf;
  double *pbuf;
  double *tbuf;
  double *my_vals;
  double *my_rhs;
  double *vals;
  double *xptr;
  double *cptr;
  int64_t my_lo, my_hi, my_n, ld64;
  int64_t lo, hi;
  int64_t *my_idx;
  int64_t *my_jdx;
  int64_t *m_idx;
  int64_t *m_jdx;
  int64_t maxlen;
  void *ptr;
  double r_one = 1.0;
  double mr_one = -1.0;

  double r2;

  double *maxr;
  double *resbuf;
  double *refptr;
  double *ebuf;
  double *maxe;
  double *eptr;
  double *dot_tp;
  double *dot_rr;
  double g_err;
  double error;

  int iteration_max = MAX_ITERATIONS;
  int iter = 0;
  int one = 1;
  int zero = 0;
  double tolerance = 1.0e-5;
  int dbgcnt;

  /* Copy sparse array */
  s_j = NGA_Sprs_array_duplicate(s_a);
  /* Construct matrix (1-D^-1*M) where M is the original matrix represented by
   * s_a and D is the diagonal of M */
  NGA_Sprs_array_get_diag(s_a, &g_d);
  NGA_Distribution64(g_d,me,&lo,&hi);
  NGA_Access64(g_d,&lo,&hi,&vals,&ld64);
  for (i=lo; i<=hi; i++) {
    vals[i-lo] = -1.0/vals[i-lo];
  }
  NGA_Sprs_array_diag_left_multiply(s_j,g_d);
  NGA_Sprs_array_shift_diag(s_j, &r_one);

  /* Create shift vector D^-1*b where b is the right hand side */
  for (i=lo; i<=hi; i++) {
    vals[i-lo] = -vals[i-lo];
  }
  NGA_Release_update64(g_d, &lo, &hi);
  g_c = GA_Duplicate(g_b, "dup_c");
  GA_Elem_multiply(g_b,g_d,g_c);
  NGA_Destroy(g_d);
  NGA_Distribution64(g_c,me,&lo,&hi);
  NGA_Access64(g_c,&lo,&hi,&cptr,&ld64);

  /* Create solution vector */
  *g_x = GA_Duplicate(g_b, "dup_x");
  g_p = GA_Duplicate(g_b, "dup_p");
  NGA_Distribution64(g_p,me,&lo,&hi);
  NGA_Access64(g_p,&lo,&hi,&pbuf,&ld64);
  GA_Zero(*g_x);
  /* Create atomic counter for testing convergence */
  g_cvg = NGA_Create_handle();
  NGA_Set_data(g_cvg,one,&one,C_INT);
  NGA_Allocate(g_cvg);
  GA_Zero(g_cvg);
  /* Create array for holding maximum residual on each process*/ 
  g_maxr = NGA_Create_handle();
  NGA_Set_data(g_maxr,one,&nprocs,C_DBL);
  NGA_Set_chunk(g_maxr,&one);
  NGA_Allocate(g_maxr);
  GA_Zero(g_maxr);
  NGA_Distribution64(g_maxr,me,&lo,&hi);
  NGA_Access64(g_maxr,&lo,&hi,&maxr,&ld64);
  /* Create array for holding norm of difference between solution and CG
   * solution */
  g_err = NGA_Create_handle();
  NGA_Set_data(g_err,one,&nprocs,C_DBL);
  NGA_Set_chunk(g_err,&one);
  NGA_Allocate(g_err);
  GA_Zero(g_err);
  NGA_Distribution64(g_err,me,&lo,&hi);
  NGA_Access64(g_err,&lo,&hi,&maxe,&ld64);

  NGA_Distribution64(g_ref,me,&lo,&hi);
  NGA_Access64(g_ref,&lo,&hi,&eptr,&ld64);

  /* Allocate buffers to check stopping criteria */
  resbuf = (double*)malloc(nprocs*sizeof(double));
  ebuf = (double*)malloc(nprocs*sizeof(double));

  /* Find out which column blocks are non-zero */
  NGA_Sprs_array_col_block_list(s_a, &list, &nblocks);
  if (nblocks>1) {
    offblocks = (int*)malloc((nblocks-1)*sizeof(int));
  }
  icnt = 0;
  maxlen = 0; 
  for (i=0; i<nblocks; i++) {
    if (list[i] != me) {
      NGA_Sprs_array_row_distribution64(s_a, list[i], &lo, &hi);
      if (maxlen < hi-lo+1) maxlen = hi-lo+1;
      offblocks[icnt] = list[i];
      icnt++;
    }
  }

  /* Find out what rows I own */
  NGA_Sprs_array_row_distribution64(s_a, me, &my_lo, &my_hi);
  if (maxlen < my_hi-my_lo+1) maxlen = my_hi-my_lo+1;
  my_n = my_hi-my_lo+1;
  /* Find maximum number of rows over all processors */
  GA_Lgop(&maxlen,1,"max");
  NGA_Sprs_array_access_col_block64(s_a, me, &my_idx, &my_jdx, &ptr);
  my_vals = (double*)ptr;
  /* Create buffer for holding solution vector from other processors */
  xbuf = (double*)malloc(maxlen*sizeof(double));
  rbuf = (double*)malloc(maxlen*sizeof(double));
  tbuf = (double*)malloc(maxlen*sizeof(double));
  /* Get pointers to local portions of solution vector and RHS vector */
  NGA_Access64(*g_x,&my_lo,&my_hi,&xptr,&ld64);
  NGA_Access64(g_b,&my_lo,&my_hi,&my_rhs,&ld64);
  /* Initial value of residual */
  GA_Norm_infinity(g_b,&residual);
  while (iter < iteration_max && !recheck_converged) {
    double resmax;
    double alpha, beta;

#if SYNCHRONOUS
    NGA_Sprs_array_matvec_multiply(s_j,*g_x,g_p);
    GA_Add(&r_one,g_p,&r_one,g_c,*g_x);
#else
    /* initialize tbuf with vector g_c */
    for (i=0; i<my_n; i++) {
      tbuf[i] = cptr[i];
    }
    /* evaluate contributions from off-diagonal blocks */
    for (i = 0; i<nblocks-1; i++) {
      NGA_Sprs_array_column_distribution64(s_j, offblocks[i], &lo, &hi);
      NGA_Get64(*g_x,&lo,&hi,xbuf,&ld64);
      /* Get pointers to local matrix block */
      NGA_Sprs_array_access_col_block64(s_j, offblocks[i], &m_idx, &m_jdx,
          &ptr);
      vals = (double*)ptr;
      lmatvec(my_lo, my_hi, lo, hi, m_idx, m_jdx, vals, xbuf, tbuf);
    }
    /* evaluate contribution from diagonal block */
    NGA_Sprs_array_access_col_block64(s_j, me, &m_idx, &m_jdx, &ptr);
    vals = (double*)ptr;
    lmatvec(my_lo, my_hi, my_lo, my_hi, my_idx, my_jdx, vals, xptr, tbuf);
    for (i=0; i<my_n; i++) {
      xptr[i] = tbuf[i];
    }
#endif

#if SYNCHRONOUS
    NGA_Sprs_array_matvec_multiply(s_a,*g_x,g_p);
    GA_Add(&mr_one,g_b,&r_one,g_p,g_p);
#else
    /* Calculate maximum residual on this process */
    for (i=0; i<my_n; i++) {
      tbuf[i] = -my_rhs[i];
    }
    /* evaluate contributions from off-diagonal blocks */
    for (i = 0; i<nblocks-1; i++) {
      NGA_Sprs_array_column_distribution64(s_a, offblocks[i], &lo, &hi);
      NGA_Get64(*g_x,&lo,&hi,xbuf,&ld64);
      /* Get pointers to local matrix block */
      NGA_Sprs_array_access_col_block64(s_a, offblocks[i], &m_idx, &m_jdx,
          &ptr);
      vals = (double*)ptr;
      lmatvec(my_lo, my_hi, lo, hi, m_idx, m_jdx, vals, xbuf, tbuf);
    }
    /* evaluate contribution from diagonal block */
    lmatvec(my_lo, my_hi, my_lo, my_hi, my_idx, my_jdx, my_vals, xptr, tbuf);
#endif

    resmax = 0.0;
    for (i=0; i<my_n; i++) {
#if SYNCHRONOUS
      if (fabs(pbuf[i]) > resmax) {
        resmax = fabs(pbuf[i]);
      }
#else
      if (fabs(tbuf[i]) > resmax) {
        resmax = fabs(tbuf[i]);
      }
#endif
    }

    /* Check for convergence
     * Only call read-increment if local convergence has changed
     * since the last cycle of row updates */
    *maxr = resmax;
#if SYNCHRONOUS
    GA_Sync();
    {
      int icnv;
      char op[2];
      if (resmax < tolerance) {
        icnv = 1;
      } else {
        icnv = 0;
      }
      op[0] = '*';
      op[1] = '\0';
      GA_Igop(&icnv,1,op);
      if (icnv == 1) {
        converged = 1;
        recheck_converged = 1;
      } else {
        converged = 0;
        recheck_converged = 0;
      }
    }
#else
    if (resmax < tolerance && !converged_tracker) {
      NGA_Read_inc(g_cvg,&zero,1);
      converged_tracker = 1;
    } else if (resmax > tolerance && converged_tracker) {
      NGA_Read_inc(g_cvg,&zero,-1);
      converged_tracker = 0;
    }
    /* Only check for global convergence if locally converged */
    if (converged_tracker) {
      int ld;
      NGA_Get(g_cvg, &zero, &zero, &icnt, &ld);
      if (!converged) {
        if (icnt >= nprocs) converged = 1;
      } else {
        if (icnt >= nprocs) {
          recheck_converged = 1;
        } else {
          converged = 0;
        }
      }
    } else {
      converged = 0;
    }
#endif
    /* estimate absolute error */
    {
      int imax=nprocs-1;
      error = 0.0;
      for (i=0; i<my_n; i++) {
        if (error < fabs(xptr[i]-eptr[i])) error = fabs(xptr[i]-eptr[i]);
      }
      *maxe = error;
#if SYNCHRONOUS
      GA_Sync();
#endif
      NGA_Get(g_err,&zero,&imax,ebuf,&one);
      error = 0.0;
      for (i=0; i<nprocs; i++)  {
        if (error < ebuf[i]) error = ebuf[i];
      }
    }
    resmax = 0.0;
    if (me == 0) {
      int imax=nprocs-1;
      NGA_Get(g_maxr,&zero,&imax,resbuf,&one);
      for (i=0; i<nprocs; i++) {
        if (resmax < resbuf[i]) resmax = resbuf[i];
      }
      printf("Iteration: %d Residual: %f error: %f\n",iter,resmax,error);
    }
    iter++;
  }

  NGA_Release64(g_b,&my_lo, &my_hi);
  NGA_Release_update64(*g_x,&my_lo, &my_hi);
  NGA_Release(g_maxr,&me, &me);
  NGA_Release(g_err,&me, &me);
  NGA_Destroy(g_cvg);
  NGA_Destroy(g_maxr);
  NGA_Destroy(g_err);
  NGA_Destroy(g_p);
  free(resbuf);
  free(ebuf);
  free(list);
  if (nblocks>1) {
    free(offblocks);
  }
  free(xbuf);
  free(rbuf);
  free(tbuf);
}

/**
 * Conjugate gradient solver for A.x = b
 * s_a: handle for sparse matrix A
 * g_b: handle for right hand side vector b
 * g_x: handle for solution vector x
 */
void cg_solve(int s_a, int g_b, int *g_x)
{
  double alpha, beta, tol;
  int g_r, g_p, g_t;
  int me = GA_Nodeid();
  double one_r;
  double m_one_r;
  double residual;
  int ncnt;
  int iterations = 10000;
  *g_x = GA_Duplicate(g_b, "dup_x");
  g_r = GA_Duplicate(g_b, "dup_r");
  g_p = GA_Duplicate(g_b, "dup_p");
  g_t = GA_Duplicate(g_b, "dup_t");
  /* accumulate boundary values to right hand side vector */
  if (me == 0) {
    printf("\nRight hand side vector completed. Starting\n");
    printf("conjugate gradient iterations.\n\n");
  }

  /* Solve Laplace's equation using conjugate gradient method */
  one_r = 1.0;
  m_one_r = -1.0;
  GA_Zero(*g_x);
  /* Initial guess is zero, so Ax = 0 and r = b */
  GA_Copy(g_b, g_r);
  GA_Copy(g_r, g_p);
  residual = GA_Ddot(g_r,g_r);
  GA_Norm_infinity(g_r, &tol);
  ncnt = 0;
  /* Start iteration loop */
  while (tol > 1.0e-5 && ncnt < iterations) {
    NGA_Sprs_array_matvec_multiply(s_a, g_p, g_t);
    alpha = GA_Ddot(g_t,g_p);
    alpha = residual/alpha;
    GA_Add(&one_r,*g_x,&alpha,g_p,*g_x);
    alpha = -alpha;
    GA_Add(&one_r,g_r,&alpha,g_t,g_r);
    GA_Norm_infinity(g_r, &tol);
    beta = residual;
    residual = GA_Ddot(g_r,g_r);
    beta = residual/beta;
    GA_Add(&one_r,g_r,&beta,g_p,g_p);
    if (me==0) printf("Iteration: %d Tolerance: %e\n",(int)ncnt+1,tol);
    ncnt++;
  }
  if (ncnt == iterations) {
    if (me==0) printf("Conjugate gradient solution failed to converge\n");
  } else {
    if (me==0) printf("Conjugate gradient solution converged after %d iterations\n",ncnt);
  }
  NGA_Destroy(g_r);
  NGA_Destroy(g_p);
  NGA_Destroy(g_t);
}

int idim, jdim, kdim;
int64_t *sizes, *offsets;
int64_t *_ilo, *_ihi, *_jlo, *_jhi, *_klo, *_khi;
int64_t ilo, ihi, jlo, jhi, klo, khi;
int64_t xdim, ydim, zdim;
int ipx, ipy, ipz, idx, idy, idz;

int getIndex(int i, int j, int k)
{
#ifndef OLD_DIST
  int ix, iy, iz, pdx, index;
  int ldx, ldy;
  int ii, jj, kk;
  ix = idx;
  if (i > ihi) ix++;
  if (i < ilo) ix--;
  iy = idy;
  if (j > jhi) iy++;
  if (j < jlo) iy--;
  iz = idz;
  if (k > khi) iz++;
  if (k < klo) iz--;
  pdx = ix + iy*ipx + iz*ipx*ipy;
  ldx = _ihi[pdx]-_ilo[pdx]+1;
  ldy = _jhi[pdx]-_jlo[pdx]+1;
  ii = i-_ilo[pdx];
  jj = j-_jlo[pdx];
  kk = k-_klo[pdx];
  index = ii + jj*ldx + kk*ldx*ldy;
  return index+offsets[pdx];
#else
  /*
  if (i < 0) i = xdim-1;
  if (i >= xdim) i = 0;
  if (j < 0) j = ydim-1;
  if (j >= ydim) j = 0;
  if (k < 0) k = zdim-1;
  if (k >= zdim) k = 0;
  */
  return i + xdim*j + k*xdim*ydim;
#endif
}

int main(int argc, char **argv) {
  int s_a, g_b, g_x, g_ref, g_dist;
  int one;
  int64_t one_64;
  int me, nproc;
  int64_t rdim, cdim, rdx, cdx;
  int64_t ldx, ldxy;
  int64_t i, j, k, ncnt;
  int  iproc, ld;
  double x, y, z, val, h, rxdim, rydim, rzdim;
  long *lmap;
  int64_t *ibuf, **iptr, *imap;
  int64_t nproc64;
  double *vptr;
  double *vbuf;
  int ok;
  double one_r = 1.0;
  double m_one_r = -1.0;
  double ir, jr, ldr;
  double xinc_p, yinc_p, zinc_p;
  double xinc_m, yinc_m, zinc_m;
  int64_t tlo, thi;
  double alpha, beta, rho, rho_m, omega, m_omega, residual;
  double rv,ts,tt;
  int nsave;
  int heap=10000000, stack=10000000;
  int iterations = 10000;
  double tol, twopi;
  FILE *PHI;
  double tbeg, t_cgsolve, t_acgsolve;
  /* Intitialize a message passing library */
  one = 1;
  one_64 = 1;
  MP_INIT(argc,argv);

  /* Initialize GA */
  NGA_Initialize();
  /* initialize random number generator */

  /* Interior points of the grid run from 0 to NDIM-1, boundary points are located
   * at -1 and NDIM for each of the axes */
  idim = NDIM;
  jdim = NDIM;
  kdim = NDIM;
  xdim = NDIM;
  ydim = NDIM;
  zdim = NDIM;
  rxdim = 1.0;
  rydim = 1.0;
  rzdim = 1.0;
  h = rxdim/((double)NDIM);
  me = GA_Nodeid();
  nproc = GA_Nnodes();
  twopi = 8.0*atan(1.0);

  t_cgsolve = 0.0;
  t_acgsolve = 0.0;
  heap /= nproc;
  stack /= nproc;
  if(! MA_init(MT_F_DBL, stack, heap))
    GA_Error("MA_init failed",stack+heap);  /* initialize memory allocator*/

  /* factor array */
  grid_factor(nproc, idim, jdim, kdim, &ipx, &ipy, &ipz);
  if (me == 0) {
    printf("Solving Laplace's equation on %d processors\n",nproc);
    printf("\n    Using %d X %d X %d processor grid\n",ipx,ipy,ipz);
    printf("\n    Grid size is %d X %d X %d\n",idim,jdim,kdim);
#ifdef OLD_DIST
    printf("\n    Using slab decomposition\n");
#else
    printf("\n    Using block decomposition\n");
#endif

  }
  /* figure out process location in proc grid */
  i = me;
  idx = me%ipx;
  i = (i-idx)/ipx;
  idy = i%ipy;
  idz = (i-idy)/ipy;
  /* find bounding indices for this processor */
  ilo = (xdim*idx)/ipx;
  if (idx < ipx-1) {
    ihi = (xdim*(idx+1))/ipx-1;
  } else {
    ihi = xdim-1;
  }
  jlo = (ydim*idy)/ipy;
  if (idy < ipy-1) {
    jhi = (ydim*(idy+1))/ipy-1;
  } else {
    jhi = ydim-1;
  }
  klo = (zdim*idz)/ipz;
  if (idz < ipz-1) {
    khi = (zdim*(idz+1))/ipz-1;
  } else {
    khi = zdim-1;
  }
  /* redefine idim, jdim, kdim */
  idim = ihi-ilo+1;
  jdim = jhi-jlo+1;
  kdim = khi-klo+1;
  sizes = (int64_t*)malloc(nproc*sizeof(int64_t));
  offsets = (int64_t*)malloc(nproc*sizeof(int64_t));
  _ilo = (int64_t*)malloc(nproc*sizeof(int64_t));
  _ihi = (int64_t*)malloc(nproc*sizeof(int64_t));
  _jlo = (int64_t*)malloc(nproc*sizeof(int64_t));
  _jhi = (int64_t*)malloc(nproc*sizeof(int64_t));
  _klo = (int64_t*)malloc(nproc*sizeof(int64_t));
  _khi = (int64_t*)malloc(nproc*sizeof(int64_t));

  for (i=0; i<nproc; i++) {
    sizes[i] = 0;
    offsets[i] = 0;
    _ilo[i] = 0;
    _ihi[i] = 0;
    _jlo[i] = 0;
    _jhi[i] = 0;
    _klo[i] = 0;
    _khi[i] = 0;
  }
  sizes[me] = (ihi-ilo+1)*(jhi-jlo+1)*(khi-klo+1);
  _ilo[me] = ilo;
  _ihi[me] = ihi;
  _jlo[me] = jlo;
  _jhi[me] = jhi;
  _klo[me] = klo;
  _khi[me] = khi;
  GA_Lgop(sizes,nproc,"+");
  GA_Lgop(_ilo,nproc,"+");
  GA_Lgop(_ihi,nproc,"+");
  GA_Lgop(_jlo,nproc,"+");
  GA_Lgop(_jhi,nproc,"+");
  GA_Lgop(_klo,nproc,"+");
  GA_Lgop(_khi,nproc,"+");
  offsets[0] = 0; 
  for (i=1; i<nproc; i++) offsets[i] = offsets[i-1]+sizes[i-1];
 
  /* create sparse array */
  rdim = xdim*ydim*zdim;
  cdim = xdim*ydim*zdim;
  ldx = xdim;
  ldxy = xdim*ydim;
  s_a = NGA_Sprs_array_create64(rdim, cdim, C_DBL);
  ncnt = 0;
  /* Set elements of Laplace operator. Use a global indexing scheme and don't
   * worry about setting elements locally. Count up values associated with
   * boundaries in variable ncnt */
  for (i=ilo; i<=ihi; i++) {
    if (i == 0) {
      xinc_m = 2.0;
      xinc_p = 1.0;
    } else if (i == NDIM - 1) {
      xinc_m = 1.0;
      xinc_p = 2.0;
    } else {
      xinc_p = 1.0;
      xinc_m = 1.0;
    }
    for (j=jlo; j<=jhi; j++) {
      if (j == 0) {
        yinc_m = 2.0;
        yinc_p = 1.0;
      } else if (j == NDIM - 1) {
        yinc_m = 1.0;
        yinc_p = 2.0;
      } else {
        yinc_p = 1.0;
        yinc_m = 1.0;
      }
      for (k=klo; k<=khi; k++) {
        if (k == 0) {
          zinc_m = 2.0;
          zinc_p = 1.0;
        } else if (k == NDIM - 1) {
          zinc_m = 1.0;
          zinc_p = 2.0;
        } else {
          zinc_p = 1.0;
          zinc_m = 1.0;
        }
        rdx = getIndex(i,j,k);
        val = -(xinc_p+xinc_m+yinc_p+yinc_m+zinc_p+zinc_m)/(h*h);
        NGA_Sprs_array_add_element64(s_a,rdx,rdx,&val);
        if (i+1 < xdim) {
          cdx = getIndex(i+1,j,k);
          val = xinc_p/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (i-1 >= 0) {
          cdx = getIndex(i-1,j,k);
          val = xinc_m/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (j+1 < ydim) {
          cdx = getIndex(i,j+1,k);
          val = yinc_p/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (j-1 >= 0) {
          cdx = getIndex(i,j-1,k);
          val = yinc_m/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (k+1 < zdim) {
          cdx = getIndex(i,j,k+1);
          val = zinc_p/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (k-1 >= 0) {
          cdx = getIndex(i,j,k-1);
          val = zinc_m/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
      }
    }
  }
  if (NGA_Sprs_array_assemble(s_a) && me == 0) {
    printf("\n    Sparse array assembly completed\n");
  }
  if (NDIM <= 16) {
    NGA_Sprs_array_export(s_a,"matrix.dat");
  }

  /* Construct RHS vector. Assume points on boundary are given by
   * the equation f(x,y,z) = cos(twopi*x) + cos(twopi*y) + cos(twopi*z) */
  ibuf = (int64_t*)malloc(ncnt*sizeof(int64_t));
  iptr = (int64_t**)malloc(ncnt*sizeof(int64_t*));
  vbuf = (double*)malloc(ncnt*sizeof(double));
  for (i=0; i<ncnt; i++) {
    iptr[i] = ibuf+i;
  }
  nsave = ncnt;
  ncnt = 0;
  /* Evaluate contributions for faces parallel to xy plane */
  if (klo == 0) {
    for (i=ilo; i<=ihi; i++) {
      for (j=jlo; j<=jhi; j++) {
        x = ((double)i+0.5)*h;
        y = ((double)j+0.5)*h;
        z = 0.0;
        vbuf[ncnt] = -2.0*(cos(twopi*x)+cos(twopi*y)+cos(twopi*z))/(h*h);
        ibuf[ncnt] = getIndex(i,j,0);
        ncnt++;
      }
    }
  }
  if (khi == zdim-1) {
    for (i=ilo; i<=ihi; i++) {
      for (j=jlo; j<=jhi; j++) {
        x = ((double)i+0.5)*h;
        y = ((double)j+0.5)*h;
        z = 1.0;
        vbuf[ncnt] = -2.0*(cos(twopi*x)+cos(twopi*y)+cos(twopi*z))/(h*h);
        ibuf[ncnt] = getIndex(i,j,zdim-1);
        ncnt++;
      }
    }
  }
  /* Evaluate contributions for faces parallel to xz plane */
  if (jlo == 0) {
    for (i=ilo; i<=ihi; i++) {
      for (k=klo; k<=khi; k++) {
        x = ((double)i+0.5)*h;
        y = 0.0;
        z = ((double)k+0.5)*h;
        vbuf[ncnt] = -2.0*(cos(twopi*x)+cos(twopi*y)+cos(twopi*z))/(h*h);
        ibuf[ncnt] = getIndex(i,0,k);
        ncnt++;
      }
    }
  }
  if (jhi == ydim-1) {
    for (i=ilo; i<=ihi; i++) {
      for (k=klo; k<=khi; k++) {
        x = ((double)i+0.5)*h;
        y = 1.0;
        z = ((double)k+0.5)*h;
        vbuf[ncnt] = -2.0*(cos(twopi*x)+cos(twopi*y)+cos(twopi*z))/(h*h);
        ibuf[ncnt] = getIndex(i,ydim-1,k);
        ncnt++;
      }
    }
  }
  /* Evaluate contributions for faces parallel to yz plane */
  if (ilo == 0) {
    for (j=jlo; j<=jhi; j++) {
      for (k=klo; k<=khi; k++) {
        x = 0.0;
        y = ((double)j+0.5)*h;
        z = ((double)k+0.5)*h;
        vbuf[ncnt] = -2.0*(cos(twopi*x)+cos(twopi*y)+cos(twopi*z))/(h*h);
        ibuf[ncnt] = getIndex(0,j,k);
        ncnt++;
      }
    }
  }
  if (ihi == xdim-1) {
    for (j=jlo; j<=jhi; j++) {
      for (k=klo; k<=khi; k++) {
        x = 1.0;
        y = ((double)j+0.5)*h;
        z = ((double)k+0.5)*h;
        vbuf[ncnt] = -2.0*(cos(twopi*x)+cos(twopi*y)+cos(twopi*z))/(h*h);
        ibuf[ncnt] = getIndex(xdim-1,j,k);
        ncnt++;
      }
    }
  }

  /* allocate global array representing right hand side vector. Make sure
   * that vector is partitioned in the same way that rows in the sparse
   * matrix are */
  imap = (int64_t*)malloc(nproc*sizeof(int64_t));
  lmap = (long*)malloc(nproc*sizeof(long));
  for (i=0; i<nproc; i++) lmap[i] = 0;
  NGA_Sprs_array_row_distribution64(s_a, me, &tlo, &thi);
  lmap[me] = (long)tlo;
  GA_Lgop(lmap, nproc, "+");
  for (i=0; i<nproc; i++) imap[i] = (int64_t)lmap[i];
  nproc64 = nproc;
  g_b = NGA_Create_handle();
  NGA_Set_data64(g_b,one,&cdim,C_DBL);
  NGA_Set_irreg_distr64(g_b,imap,&nproc64);
  NGA_Allocate(g_b);
  free(lmap);
  free(imap);
  GA_Zero(g_b);
  NGA_Scatter_acc64(g_b,vbuf,iptr,ncnt,&one_r);
  GA_Sync();
  free(ibuf);
  free(iptr);
  free(vbuf);
  free(sizes);
  free(offsets);
  free(_ilo);
  free(_ihi);
  free(_jlo);
  free(_jhi);
  free(_klo);
  free(_khi);
  GA_Norm_infinity(g_b, &x);
  if (me == 0) {
    printf("\n    Right hand side completed. Maximum value of RHS: %e\n",x);
  }

  /* Matrix and right hand side have been constructed. Begin solution
   * loop using asynchronous conjugate gradient algorithm */

  tbeg = GA_Wtime();
  cg_solve(s_a, g_b, &g_ref);
  t_cgsolve = GA_Wtime()-tbeg;
  tbeg = GA_Wtime();
  j_solve(s_a, g_b, g_ref, &g_x);
  t_acgsolve = GA_Wtime()-tbeg;
  GA_Dgop(&t_cgsolve,1,"+");
  t_cgsolve /= ((double)nproc);
  GA_Dgop(&t_acgsolve,1,"+");
  t_acgsolve /= ((double)nproc);
  if (me == 0) {
    printf("Elapsed time in CG solver:              %16.4f\n",t_cgsolve);
    printf("Elapsed time in Asynchronous CG solver: %16.4f\n",t_acgsolve);
  }

  /* Gather some information on difference between Kaczmarz and exact
   * solution. Start by finding maximum and minimum difference for all
   * elements in solution vector*/
  {
    double *kptr, *xptr;
    int64_t nelem;
    double diff_min, diff_max;
    int diff_min_zero = 0;
    double lmin, lmax, bin_size;
    int *bins;
    int ibin;
    NGA_Distribution64(g_x, me, &tlo, &thi);
    NGA_Access64(g_ref, &tlo, &thi, &kptr, &one_64);
    nelem = thi-tlo+1;
    NGA_Distribution64(g_ref, me, &tlo, &thi);
    if (nelem != thi-tlo+1) {
      printf("p[%d] Elements in exact and async CG solutions differ!\n",me);
      printf("p[%d] Exact: %ld Async CG: %ld\n",me,thi-tlo+1,nelem);
      GA_Error("Cannot compute difference vector properties\n",0);
    }
    NGA_Access64(g_x, &tlo, &thi, &xptr, &one_64);
    diff_min = fabs(xptr[0]-kptr[0]);
    diff_max = fabs(xptr[0]-kptr[0]);
    for (i=1; i<nelem; i++) {
      double diff = fabs(xptr[i]-kptr[i]);
      if (diff > diff_max) diff_max = diff;
      /* Shouldn't happen, but try and make sure diff_min > 0.0 */
      if (diff_min == 0.0) {
        diff_min_zero = 1;
        diff_min = diff;
      } else if(diff < diff_min && diff > 0.0) {
        diff_min = diff;
      }
    }
    if (diff_min_zero) {
      printf("p[%d] found minimum difference of zero\n",me);
    }
    /* find global minimum and maximum difference */
    GA_Dgop(&diff_min,1,"min");
    GA_Dgop(&diff_max,1,"max");
    bins = (int*)malloc(NUM_BINS*sizeof(int));
    for (i=0; i<NUM_BINS; i++) {
      bins[i] = 0;
    }
    lmin = log10(diff_min);
    lmax = log10(diff_max);
    bin_size = (lmax-lmin)/((double)NUM_BINS);
    /* Bin up differences locally */
    for (i=0; i<nelem; i++) {
      double diff = fabs(xptr[i]-kptr[i]);
      ibin = (int)((log10(diff)-lmin)/bin_size+0.5);
      if (ibin >= NUM_BINS) ibin--;
      bins[ibin]++;
    }
    /* Bin up differences globally */
    GA_Igop(bins,NUM_BINS,"+");
    if (me == 0) {
      printf("Log_10 minimum difference: %f\n",lmin);
      printf("Log_10 maximum difference: %f\n",lmax);
      for (i=0; i<NUM_BINS; i++) {
        printf("  Bin center: %f number of entries %d\n",
            lmin+((double)i+0.5)*bin_size,bins[i]);
      }
      PHI = fopen("edist.dat","w");
      for (i=0; i<NUM_BINS; i++) {
        fprintf(PHI,"%f %d\n",lmin+((double)i+0.5)*bin_size,bins[i]);
      }
      fclose(PHI);
    }
   
  }

  /* Write solution to file */
#ifdef WRITE_VTK
#ifndef OLD_DIST
  {
    int g_v, g_ex;
    int64_t lo[3], hi[3];
    int64_t dims[3];
    int ndim = 3;
    double *transpose;
    int ix, iy, iz;
    int64_t nelem = idim*jdim*kdim;
    double *xptr;
    int64_t lld[3];

    /* create global array to reformat data */
    NGA_Distribution64(g_x, me, &tlo, &thi);
    NGA_Access64(g_x, &tlo, &thi, &xptr, &one_64);
    dims[0] = xdim;
    dims[1] = ydim;
    dims[2] = zdim;
    lo[0] = ilo;
    lo[1] = jlo;
    lo[2] = klo;
    hi[0] = ihi;
    hi[1] = jhi;
    hi[2] = khi;
    g_v = NGA_Create_handle();
    NGA_Set_data64(g_v,ndim,dims,C_DBL);
    NGA_Allocate(g_v);
    /* transpose data */
    transpose = (double*)malloc(idim*jdim*kdim*sizeof(double));
    for (i=0; i<nelem; i++) {
      int n = i;
      ix = n%idim;
      n = (n-ix)/idim;
      iy = n%jdim;
      iz = (n-iy)/jdim;
      transpose[ix*jdim*kdim + iy*kdim + iz] = xptr[i];
    }
    NGA_Release64(g_x, &tlo, &thi);
    lld[0] = hi[1]-lo[1]+1;
    lld[1] = hi[2]-lo[2]+1;
    NGA_Put64(g_v,lo,hi,transpose,lld);

    NGA_Access64(g_ref, &tlo, &thi, &xptr, &one_64);
    g_ex = NGA_Create_handle();
    NGA_Set_data64(g_ex,ndim,dims,C_DBL);
    NGA_Allocate(g_ex);
    /* transpose data */
    for (i=0; i<nelem; i++) {
      int n = i;
      ix = n%idim;
      n = (n-ix)/idim;
      iy = n%jdim;
      iz = (n-iy)/jdim;
      transpose[ix*jdim*kdim + iy*kdim + iz] = xptr[i];
    }
    NGA_Release64(g_ref, &tlo, &thi);
    lld[0] = hi[1]-lo[1]+1;
    lld[1] = hi[2]-lo[2]+1;
    NGA_Put64(g_ex,lo,hi,transpose,lld);
    GA_Sync();
    free(transpose);
    if (me == 0) {
      vbuf = (double*)malloc(xdim*ydim*sizeof(double));
      double *exbuf = (double*)malloc(xdim*ydim*sizeof(double));
      PHI = fopen("phi.vtk","w");
      fprintf(PHI,"# vtk DataFile Version 3.0\n");
      fprintf(PHI,"Laplace Equation Solution\n");
      fprintf(PHI,"ASCII\n");
      fprintf(PHI,"DATASET STRUCTURED_POINTS\n");
      fprintf(PHI,"DIMENSIONS %ld %ld %ld\n",xdim,ydim,zdim);
      fprintf(PHI,"ORIGIN %12.6f %12.6f %12.6f\n",0.5*h,0.5*h,0.5*h);
      fprintf(PHI,"SPACING %12.6f %12.6f %12.6f\n",h,h,h);
      fprintf(PHI," \n");    
      fprintf(PHI,"POINT_DATA %ld\n",xdim*ydim*zdim);
      fprintf(PHI,"SCALARS Phi float\n");
      fprintf(PHI,"LOOKUP_TABLE default\n");
      lo[0] = 0;
      lo[1] = 0;
      hi[0] = xdim-1;
      hi[1] = ydim-1;
      lld[0] = ydim;
      lld[1] = 1;
      for (k=0; k<zdim; k++) {
        int crtcnt = 0;
        lo[2] = k;
        hi[2] = k;
        NGA_Get64(g_v,lo,hi,vbuf,lld);
        for (j=0; j<ydim; j++) {
          for (i=0; i<xdim; i++) {
            fprintf(PHI," %12.6f",vbuf[i+j*xdim]);
            crtcnt++;
            if (crtcnt%5 == 0) fprintf(PHI,"\n");
          }
        }
        if (crtcnt%5 != 0) fprintf(PHI,"\n");
      }
      fprintf(PHI,"SCALARS Diff float\n");
      fprintf(PHI,"LOOKUP_TABLE default\n");
      for (k=0; k<zdim; k++) {
        int crtcnt = 0;
        lo[2] = k;
        hi[2] = k;
        NGA_Get64(g_v,lo,hi,vbuf,lld);
        NGA_Get64(g_ex,lo,hi,exbuf,lld);
        for (j=0; j<ydim; j++) {
          for (i=0; i<xdim; i++) {
            fprintf(PHI," %12.6f",fabs(vbuf[i+j*xdim]-exbuf[i+j*xdim]));
            crtcnt++;
            if (crtcnt%5 == 0) fprintf(PHI,"\n");
          }
        }
        if (crtcnt%5 != 0) fprintf(PHI,"\n");
      }
      fclose(PHI);
      free(vbuf);
      free(exbuf);
    }
    GA_Destroy(g_v);
    GA_Destroy(g_ex);
  }
#else
    if (me == 0) {
      vbuf = (double*)malloc(xdim*ydim*sizeof(double));
      PHI = fopen("phi.vtk","w");
      fprintf(PHI,"# vtk DataFile Version 3.0\n");
      fprintf(PHI,"Laplace Equation Solution\n");
      fprintf(PHI,"ASCII\n");
      fprintf(PHI,"DATASET STRUCTURED_POINTS\n");
      fprintf(PHI,"DIMENSIONS %ld %ld %ld\n",xdim,ydim,zdim);
      fprintf(PHI,"ORIGIN %12.6f %12.6f %12.6f\n",0.5*h,0.5*h,0.5*h);
      fprintf(PHI,"SPACING %12.6f %12.6f %12.6f\n",h,h,h);
      fprintf(PHI," \n");    
      fprintf(PHI,"POINT_DATA %ld\n",xdim*ydim*zdim);
      fprintf(PHI,"SCALARS Phi float\n");
      fprintf(PHI,"LOOKUP_TABLE default\n");
      for (k=0; k<zdim; k++) {
        int crtcnt = 0;
        ilo = k*xdim*ydim;
        ihi = ilo + xdim*ydim - 1;
        NGA_Get64(g_x,&ilo,&ihi,vbuf,&one_64);
        for (j=0; j<ydim; j++) {
          for (i=0; i<xdim; i++) {
            fprintf(PHI," %12.6f",vbuf[i+j*xdim]);
            crtcnt++;
            if (crtcnt%5 == 0) fprintf(PHI,"\n");
          }
        }
        if (crtcnt%5 != 0) fprintf(PHI,"\n");
      }

      fclose(PHI);
      free(vbuf);
    }
#endif
#endif

  NGA_Sprs_array_destroy(s_a);
  NGA_Destroy(g_b);
  NGA_Destroy(g_x);
  NGA_Destroy(g_ref);

  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
