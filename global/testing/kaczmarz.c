#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

#define OLD_XDIST

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

/* Randomly reorder the sequence of integers [0,1,2,...,N-1] by
 * apply N random pair permutations */
void reorder(int *seq, int N)
{
  int i, j, idx, jdx;
  double rn;
  int me = GA_Nodeid();
  for (i=0; i<N; i++) seq[i] = i;
  for (i=0; i<N; i++) {
    /* randomly pick location at higher index value */
    rn = (double)(N-1 - i);  
    j = (int)(rn*NGA_Rand(0)+0.5);
    if (j+i >= N) j--;
    idx = seq[i];
    jdx = seq[j+i];
    seq[i] = jdx;
    seq[j+i] = idx;
  }
}

void k_solve(int s_a, int g_b, int g_ref, int *g_x)
{
  int nblocks;
  int *list;
  int i, j, nb, icnt, g_cvg, g_maxr, g_ax;
  int my_nb;
  int converged = 0;
  int converged_tracker = 0;
  int recheck_converged = 0;
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int64_t my_lo, my_hi, my_n, ld64;
  double *my_vals;
  double *my_rhs;
  int64_t total;
  double *xblocks;
  double **blk_ptrs;
  double *diag_ptr;
  double *normA;
  double *maxr;
  int64_t *b_n;
  int64_t **m_idx;
  int64_t **m_jdx;
  int64_t *m_jlo;
  int64_t ilast;
  double **m_vals;
  double *resbuf;
  double *refptr;
  double *ebuf;
  double *maxe;
  double *eptr;
  double g_err;
  double error;
  int tlo, thi, tld;
  int iteration_max = MAX_ITERATIONS;
  int iter;
  int *seq;
  int irow;
  int one = 1;
  int zero = 0;
  double tolerance = 1.0e-5;
  double maxax;
  double axmb;
  double axtmp;
  double maxinc;
  int maxrow;
  /* Create solution vector */
  *g_x = GA_Duplicate(g_b, "dup_x");
  GA_Zero(*g_x);
  /* Create atomic counter for testing convergence */
  g_cvg = NGA_Create_handle();
  NGA_Set_data(g_cvg,one,&one,C_INT);
  NGA_Allocate(g_cvg);
  GA_Zero(g_cvg);
  g_maxr = NGA_Create_handle();
  NGA_Set_data(g_maxr,one,&nprocs,C_DBL);
  NGA_Allocate(g_maxr);
  GA_Zero(g_maxr);
  g_ax = NGA_Create_handle();
  NGA_Set_data(g_ax,one,&nprocs,C_DBL);
  NGA_Allocate(g_ax);
  GA_Zero(g_ax);
  NGA_Distribution(g_maxr,me,&tlo,&thi);
  NGA_Access(g_maxr,&tlo,&thi,&maxr,&tld);

  g_err = NGA_Create_handle();
  NGA_Set_data(g_err,one,&nprocs,C_DBL);
  NGA_Allocate(g_err);
  GA_Zero(g_err);
  NGA_Distribution(g_err,me,&tlo,&thi);
  NGA_Access(g_err,&tlo,&thi,&maxe,&tld);
  NGA_Distribution(g_ref,me,&tlo,&thi);
  NGA_Access(g_ref,&tlo,&thi,&eptr,&tld);


  /* Find out which column blocks are non-zero */
  NGA_Sprs_array_col_block_list(s_a, &list, &nblocks);
  if (nblocks>1) {
    blk_ptrs = (double**)malloc((nblocks-1)*sizeof(double*));
  }
  b_n = (int64_t*)malloc(nblocks*sizeof(int64_t*));
  m_idx = (int64_t**)malloc(nblocks*sizeof(int64_t*));
  m_jdx = (int64_t**)malloc(nblocks*sizeof(int64_t*));
  m_jlo = (int64_t*)malloc(nblocks*sizeof(int64_t));
  m_vals = (double**)malloc(nblocks*sizeof(double*));
  resbuf = (double*)malloc(nprocs*sizeof(double));
  ebuf = (double*)malloc(nprocs*sizeof(double));
  /* Find total data on processes corresponding to non-zero column blocks
   * (but not including block on diagonal) */
  total = 0;
  for (i=0; i<nblocks; i++) {
    int lo, hi;
    NGA_Distribution(*g_x, list[i], &lo, &hi);
    b_n[i] = hi-lo+1;
    if (list[i] != me) {
      total += (int64_t)(hi-lo+1);
    } else {
      my_nb = i;
    }
  }
  xblocks = (double*)malloc(total*sizeof(double));
  /* set up blk_ptrs */
  if (nblocks > 1) {
    blk_ptrs[0] = NULL;
  }
  icnt = 0;
  for (i=0; i<nblocks; i++) {
    int lo, hi;
    if (list[i] != me) {
      NGA_Distribution(*g_x, list[i], &lo, &hi);
      if (icnt == 0) {
        blk_ptrs[icnt] = xblocks;
      } else {
        blk_ptrs[icnt] = blk_ptrs[icnt-1] + ilast;
      }
      ilast = (int64_t)(hi-lo+1);
      icnt++;
    }
  }
  /* set up pointers to sparse blocks in matrix */
  for (i=0; i<nblocks; i++) {
    void *tmp_ptr;
    int64_t ilo, ihi;
    NGA_Sprs_array_column_distribution64(s_a, list[i], &ilo, &ihi);
    m_jlo[i] = ilo;
    NGA_Sprs_array_access_col_block64(s_a, list[i], &m_idx[i], &m_jdx[i], &tmp_ptr);
    if (tmp_ptr == NULL) printf("Null values pointer for row block %d"
        " column block %d\n",me,list[i]);
    m_vals[i] = (double*)tmp_ptr;
  }
  /* find the norm of each row of matrix s_a */
  NGA_Sprs_array_row_distribution64(s_a, me, &my_lo, &my_hi);
  //printf("p[%d] my_lo: %ld my_hi: %ld\n",me,my_lo,my_hi);
  my_n = my_hi - my_lo + 1;
  normA = (double*)malloc(my_n*sizeof(double));
  for (i=0; i<my_n; i++) {
    normA[i] = 0.0;
    for (nb = 0; nb<nblocks; nb++) {
      int64_t *idx = m_idx[nb];
      int64_t *jdx = m_jdx[nb];
      double *vals = m_vals[nb];
      int64_t jnum = idx[i+1]-idx[i];
      int64_t jstart = idx[i];
      for (j=0; j<jnum; j++) {
        double val = vals[jstart+j];
        normA[i] += val*val;
      }
    }
  }

  /* get pointers etc. to my slice of the solution vector */
  NGA_Access64(*g_x, &my_lo, &my_hi, &my_vals, &ld64);
  NGA_Access64(g_b, &my_lo, &my_hi, &my_rhs, &ld64);
  /* Normalize A and b by row norm of A */
#if 0
  for (i=0; i<my_n; i++) {
    icnt = 0;
    normA[i] = sqrt(normA[i]);
    for (nb = 0; nb<nblocks; nb++) {
      int64_t *idx = m_idx[nb];
      int64_t *jdx = m_jdx[nb];
      double *vals = m_vals[nb];
      int64_t jnum = idx[i+1]-idx[i];
      int64_t jstart = idx[i];
      for (j=0; j<jnum; j++) {
        int64_t icol = jdx[jstart+j]-m_jlo[nb];
        vals[jstart+j] /= normA[i];
      }
    }
    my_rhs[i] /= normA[i];
    normA[i] = 1.0;
  }
#endif
  seq = (int*)malloc(my_n*sizeof(int));
  /* start iteration loop */
  iter = 0;
  while (!converged && iter < iteration_max && !recheck_converged) {
    double residual;
    double res_max;
    /* Update values of solution */
    icnt = 0;
    for (nb = 0; nb<nblocks; nb++) {
      if (list[nb] != me) {
        int lo, hi, ld; 
        NGA_Distribution(*g_x, list[nb], &lo, &hi);
        NGA_Get(*g_x, &lo, &hi, blk_ptrs[icnt], &ld);
        icnt++;
      }
    }

    reorder(seq, my_n);
    maxax = 0.0;
    axmb = 0.0;
    maxinc = 0.0;
    /* loop over rows in my row block */
    for (i=0; i<my_n; i++) {
      int irow = seq[i];
      double axdot = 0.0;
      double xmb = 0.0;
      /* loop through all blocks and calculate inner product of x and
       * row irow of sparse matrix */
      icnt = 0;
      for (nb = 0; nb<nblocks; nb++) {
        int64_t *idx = m_idx[nb];
        int64_t *jdx = m_jdx[nb];
        double *vals = m_vals[nb];
        double *xptr;
        if (list[nb] != me) {
          xptr = blk_ptrs[icnt];
          icnt++;
        } else {
          xptr = my_vals;
        }
        int64_t jnum = idx[irow+1]-idx[irow];
        int64_t jstart = idx[irow];
        for (j=0; j<jnum; j++) {
          int64_t icol = jdx[jstart+j]-m_jlo[nb];
          if (icol < 0 || icol >= b_n[nb])
            printf("p[%d] Illegal column index: %ld\n",me,icol);
          double val = vals[jstart+j];
          axdot += val*xptr[icol];
        }
      }
      /* Finish calculating update to x[irow] */
      if (normA[irow] != 0.0) {
        axtmp = axdot;
        axdot -= my_rhs[irow];
        axdot /= normA[irow];
        if (fabs(axdot) >  axmb) {
          maxax = axtmp;
          axmb = fabs(axdot);
          maxrow = irow;
        }
      } else {
        GA_Error("Row of matrix is all zeros!",irow);
      }
      /* Update vector elements */
#if 1
      {
        double *xptr = my_vals;
        int64_t *idx = m_idx[my_nb];
        int64_t *jdx = m_jdx[my_nb];
        double *vals = m_vals[my_nb];
        int64_t jnum = idx[irow+1]-idx[irow];
        int64_t jstart = idx[irow];
        for (j=0; j<jnum; j++) {
          int64_t icol = jdx[jstart+j]-m_jlo[my_nb];
          double val = vals[jstart+j];
          if (icol < 0 || icol >= my_n) {
            printf("p[%d] icol out of bounds icol: %ld my_n: %ld\n",me,icol,my_n);
          }
          xptr[icol] -= val*axdot;
          if (fabs(val*axdot) > maxinc) maxinc = fabs(val*axdot);
        }
      }
#else
      icnt = 0;
      for (nb = 0; nb<nblocks; nb++) {
        double *xptr;
        int64_t *idx = m_idx[nb];
        int64_t *jdx = m_jdx[nb];
        double *vals = m_vals[nb];
        int64_t jnum = idx[irow+1]-idx[irow];
        int64_t jstart = idx[irow];
        if (list[nb] != me) {
          xptr = blk_ptrs[icnt];
          icnt++;
        } else {
          xptr = my_vals;
        }
        for (j=0; j<jnum; j++) {
          int64_t icol = jdx[jstart+j]-m_jlo[nb];
          double val = vals[jstart+j];
          if (icol < 0) {
            printf("p[%d] icol out of bounds icol: %ld\n",me,icol);
          }
          xptr[icol] -= val*axdot;
          if (fabs(val*axdot) > maxinc) maxinc = fabs(val*axdot);
        }
      }
#endif
    }
    /* Calculate maximum residual on this process */
    residual = 0.0;
    for (i=0; i<my_n; i++) {
      double xi=0.0;
      icnt = 0;
      for (nb = 0; nb<nblocks; nb++) {
        int64_t *idx = m_idx[nb];
        int64_t *jdx = m_jdx[nb];
        double *vals = m_vals[nb];
        double *xptr;
        if (list[nb] != me) {
          xptr = blk_ptrs[icnt];
          icnt++;
        } else {
          xptr = my_vals;
        }
        int64_t jnum = idx[i+1]-idx[i];
        int64_t jstart = idx[i];
        for (j=0; j<jnum; j++) {
          int64_t icol = jdx[jstart+j]-m_jlo[nb];
          double val = vals[jstart+j];
          xi += val*xptr[icol];
        }
      }
      if (fabs(xi-my_rhs[i]) > residual) {
        residual = fabs(xi-my_rhs[i]);
      }
    }
    *maxr = residual;
    /* Check for convergence
     * Only call read-increment if local convergence has changed
     * since the last cycle of row updates */
    if (residual < tolerance && !converged_tracker) {
      NGA_Read_inc(g_cvg,&zero,1);
      converged_tracker = 1;
    } else if (residual > tolerance && converged_tracker) {
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
    /* estimate absolute error */
    {
      int imax=nprocs-1;
      error = 0.0;
      for (i=0; i<my_n; i++) {
        error += pow((my_vals[i]-eptr[i]),2);
      }
      *maxe = error;
      NGA_Get(g_err,&zero,&imax,ebuf,&one);
      error = 0.0;
      for (i=0; i<nprocs; i++) error += ebuf[i];
      error = sqrt(error);
    }
    if (me == 0) {
      int imax=nprocs-1;
      double resmax = 0.0;
      NGA_Get(g_maxr,&zero,&imax,resbuf,&one);
      for (i=0; i<nprocs; i++) {
        if (resmax < resbuf[i]) resmax = resbuf[i];
      }
      printf("Iteration: %d Residual: %f error: %f\n",iter,resmax,error);
    }
    //GA_Sync();
    iter++;
  }
#if 0
  {
    int max = nprocs-1;
    NGA_Put(g_ax,&me,&me,&maxax,&one);
    GA_Sync();
    if (me==0) {
      double *abuf = (double*)malloc(nprocs*sizeof(double));
      NGA_Get(g_ax,&zero,&max,abuf,&one);
      for (nb=0; nb<nprocs; nb++) {
        printf("p[%d] iteration: %d amax[%d]: %f\n",me,iter,nb,abuf[nb]);
      }
      free(abuf);
    }
    GA_Sync();
    NGA_Put(g_ax,&me,&me,&axmb,&one);
    GA_Sync();
    if (me==0) {
      double *abuf = (double*)malloc(nprocs*sizeof(double));
      NGA_Get(g_ax,&zero,&max,abuf,&one);
      for (nb=0; nb<nprocs; nb++) {
        printf("p[%d] iteration: %d max (ax-b)/a2[%d]: %f\n",me,iter,nb,abuf[nb]);
      }
      free(abuf);
    }
    GA_Sync();
    axtmp = fabs(maxax-my_rhs[maxrow]);
    NGA_Put(g_ax,&me,&me,&axtmp,&one);
    GA_Sync();
    if (me==0) {
      double *abuf = (double*)malloc(nprocs*sizeof(double));
      NGA_Get(g_ax,&zero,&max,abuf,&one);
      for (nb=0; nb<nprocs; nb++) {
        printf("p[%d] iteration: %d max ax-b[%d]: %f\n",me,iter,nb,abuf[nb]);
      }
      free(abuf);
    }
    GA_Sync();
    NGA_Put(g_ax,&me,&me,&maxinc,&one);
    GA_Sync();
    if (me==0) {
      double *abuf = (double*)malloc(nprocs*sizeof(double));
      NGA_Get(g_ax,&zero,&max,abuf,&one);
      for (nb=0; nb<nprocs; nb++) {
        printf("p[%d] iteration: %d max increment[%d]: %f\n",me,iter,nb,abuf[nb]);
      }
      free(abuf);
    }
    GA_Sync();
    NGA_Put(g_ax,&me,&me,&my_rhs[maxrow],&one);
    GA_Sync();
    if (me==0) {
      double *abuf = (double*)malloc(nprocs*sizeof(double));
      NGA_Get(g_ax,&zero,&max,abuf,&one);
      for (nb=0; nb<nprocs; nb++) {
        printf("p[%d] iteration: %d max RHS[%d]: %f\n",me,iter,nb,abuf[nb]);
      }
      free(abuf);
    }
  }
#endif

  NGA_Release64(g_b,&my_lo, &my_hi);
  NGA_Release_update64(*g_x,&my_lo, &my_hi);
  NGA_Release(g_maxr,&me, &me);
  NGA_Release(g_err,&me, &me);
  NGA_Destroy(g_cvg);
  NGA_Destroy(g_maxr);
  NGA_Destroy(g_err);
  free(resbuf);
  free(ebuf);
  free(b_n);
  free(m_idx);
  free(m_jdx);
  free(m_jlo);
  free(m_vals);
  if (nblocks > 1) {
    free(blk_ptrs);
  }
  free(xblocks);
  free(seq);
  free(list);
}

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
    if (me==0) printf("Iteration: %d Tolerance: %e\n",(int)ncnt+1,tol);
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
int64_t *offsets;
long *sizes;
long *_ilo, *_ihi, *_jlo, *_jhi, *_klo, *_khi;
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
  double tbeg, t_cgsolve, t_ksolve;
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
  x = NGA_Rand(32823+me);

  t_cgsolve = 0.0;
  t_ksolve = 0.0;
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
  sizes = malloc(nproc*sizeof(int64_t));
  offsets = (int64_t*)malloc(nproc*sizeof(int64_t));
  _ilo = malloc(nproc*sizeof(int64_t));
  _ihi = malloc(nproc*sizeof(int64_t));
  _jlo = malloc(nproc*sizeof(int64_t));
  _jhi = malloc(nproc*sizeof(int64_t));
  _klo = malloc(nproc*sizeof(int64_t));
  _khi = malloc(nproc*sizeof(int64_t));

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
  if (NDIM <= 32) {
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
   * loop using Kaczmarz algorithm */

  tbeg = GA_Wtime();
  cg_solve(s_a, g_b, &g_ref);
  t_cgsolve = GA_Wtime()-tbeg;
  tbeg = GA_Wtime();
  k_solve(s_a, g_b, g_ref, &g_x);
  t_ksolve = GA_Wtime()-tbeg;
  GA_Dgop(&t_cgsolve,1,"+");
  t_cgsolve /= ((double)nproc);
  GA_Dgop(&t_ksolve,1,"+");
  t_ksolve /= ((double)nproc);
  if (me == 0) {
    printf("Elapsed time in CG solver:       %16.4f\n",t_cgsolve);
    printf("Elapsed time in Kaczmarz solver: %16.4f\n",t_ksolve);
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
      printf("p[%d] Elements in exact and Kacmarz solutions differ!\n",me);
      printf("p[%d] Exact: %ld Kacmarz: %ld\n",me,thi-tlo+1,nelem);
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
