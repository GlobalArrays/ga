#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

#define WRITE_VTK
#define CG_SOLVE 0
#define NDIM 256

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
  ip = (int)(sqrt((double)p))+1;
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

#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC (1.0/MBIG)

double ran3(long idum)
{
  static int inext,inextp;
  static long ma[56];
  static int iff=0;
  long mj,mk;
  int i,ii,k;

  if (idum < 0 || iff == 0) {
    iff=1;
    mj=MSEED-(idum < 0 ? -idum : idum);
    mj %= MBIG;
    ma[55]=mj;
    mk=1;
    for (i=1;i<=54;i++) {
      ii=(21*i) % 55;
      ma[ii]=mk;
      mk=mj-mk;	
        if (mk < MZ) mk += MBIG;
      mj=ma[ii];
    }
    for (k=1;k<=4;k++)
      for (i=1;i<=55;i++) {
        ma[i] -= ma[1+(i+30) % 55];
        if (ma[i] < MZ) ma[i] += MBIG;
      }
    inext=0;
    inextp=31;
    idum=1;
  }
  if (++inext == 56) inext=1;
  if (++inextp == 56) inextp=1;
  mj=ma[inext]-ma[inextp];
  if (mj < MZ) mj += MBIG;
  ma[inext]=mj;
  return mj*FAC;
}
            
#undef MBIG
#undef MSEED
#undef MZ
#undef FAC

/* Randomly reorder the sequence of integers [0,1,2,...,N-1] by
 * apply N random pair permutations */
void reorder(int *seq, int N)
{
  int i, j, idx, jdx;
  double rn;
  int me = GA_Nodeid();
  for (i=0; i<N; i++) seq[i] = i;
#if 1
  for (i=0; i<N; i++) {
    /* randomly pick location at higher index value */
    rn = (double)(N-1 - i);  
    j = (int)(rn*ran3(0)+0.5);
    if (j+i >= N) j--;
    idx = seq[i];
    jdx = seq[j+i];
    seq[i] = jdx;
    seq[j+i] = idx;
  }
#else
  for (i=0; i<N; i++) {
    /* randomly pick two locations in seq */
    idx = (int)(rn*ran3());
    if (idx>=N) idx=N-1;
    jdx = (int)(rn*ran3());
    if (jdx>=N) jdx=N-1;
    itmp = seq[idx];
    seq[idx] = seq[jdx];
    seq[jdx] = itmp;
  }
#endif
}

void k_solve(int s_a, int g_b, int *g_x)
{
  int nblocks;
  int *list;
  int i, j, nb, icnt, g_cvg;
  int converged = 0;
  int converged_tracker = 0;
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
  int64_t **m_idx;
  int64_t **m_jdx;
  int64_t *m_jlo;
  double **m_vals;
  int iteration_max = 10000;
  int iter;
  int *seq;
  int irow;
  int one = 1;
  int zero = 0;
  double tolerance = 1.0e-5;
  /* Create solution vector */
  *g_x = GA_Duplicate(g_b, "dup_x");
  GA_Zero(*g_x);
  /* Create atomic counter for testing convergence */
  g_cvg = NGA_Create_handle();
  NGA_Set_data(g_cvg,one,&one,C_INT);
  NGA_Allocate(g_cvg);
  GA_Zero(g_cvg);
  /* Find out which column blocks are non-zero */
  NGA_Sprs_array_col_block_list(s_a, &list, &nblocks);
  blk_ptrs = (double**)malloc((nblocks-1)*sizeof(double*));
  m_idx = (int64_t**)malloc(nblocks*sizeof(int64_t*));
  m_jdx = (int64_t**)malloc(nblocks*sizeof(int64_t*));
  m_jlo = (int64_t*)malloc(nblocks*sizeof(int64_t));
  m_vals = (double**)malloc(nblocks*sizeof(double*));
  /* Find total data on processes corresponding to non-zero column blocks
   * (but not including block on diagonal) */
  total = 0;
  for (i=0; i<nblocks; i++) {
    int lo, hi;
    if (list[i] != me) {
      NGA_Distribution(*g_x, list[i], &lo, &hi);
      total += (int64_t)(hi-lo+1);
    }
  }
  xblocks = (double*)malloc(total*sizeof(double));
  /* set up blk_ptrs */
  blk_ptrs[0] = NULL;
  icnt = 0;
  for (i=0; i<nblocks; i++) {
    int lo, hi;
    if (list[i] != me) {
      NGA_Distribution(*g_x, list[i], &lo, &hi);
      if (icnt == 0) {
        blk_ptrs[icnt] = xblocks;
      } else {
        blk_ptrs[icnt] = blk_ptrs[icnt-1] + (hi-lo+1);
      }
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
    if (tmp_ptr == NULL) printf("Null values pointer for row block %d column block %d\n",
        me,list[i]);
    m_vals[i] = (double*)tmp_ptr;
  }
  /* find the norm of each row of matrix s_a */
  NGA_Sprs_array_row_distribution64(s_a, me, &my_lo, &my_hi);
  my_n = my_hi - my_lo + 1;
  normA = (double*)malloc(my_n*sizeof(double));
  for (i=0; i<my_n; i++) {
    normA[i] = 0.0;
    for (nb = 0; nb<nblocks; nb++) {
      int64_t *idx = m_idx[nb];
      int64_t *jdx = m_jdx[nb];
      double *vals = m_vals[nb];
      int64_t jnum = idx[i+1]-idx[i]+1;
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
  seq = (int*)malloc(my_n*sizeof(int));
  ran3(-323892+me);
  /* start iteration loop */
  iter = 0;
  while (!converged && iter < iteration_max) {
    double residual;
    double res_max;
    /* Update values of solution */
    icnt = 0;
    for (nb = 0; nb<nblocks; nb++) {
      if (list[nb] != me) {
        int lo, hi, ld; 
        NGA_Distribution(g_b, list[nb], &lo, &hi);
        NGA_Get(g_b, &lo, &hi, blk_ptrs[icnt], &ld);
        icnt++;
      }
    }

    reorder(seq, my_n);
    for (i=0; i<my_n; i++) {
      int irow = seq[i];
      double axdot = 0.0;
      int jmax = 0;
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
        if (jnum > jmax) jmax = jnum;
        for (j=0; j<jnum; j++) {
          int64_t icol = jdx[jstart+j]-m_jlo[nb];
          if (icol < 0 || icol >= my_n)
            printf("p[%d] Illegal column index: %ld\n",me,icol);
          double val = vals[jstart+j];
          axdot += val*xptr[icol];
        }
      }
      /* Finish calculating update to x[irow] */
      if (normA[irow] != 0.0) {
        axdot -= my_rhs[irow];
        axdot /= normA[irow];
      } else {
        GA_Error("Row of matrix is all zeros!",irow);
      }
      /* Update vector elements */
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
          double val = vals[jstart+j];
          xptr[icol] -= val*axdot;
        }
      }
    }
    /* Check for convergence */
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
    /* Only call read-increment if local convergence has changed
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
      if (icnt >= nprocs) converged = 1;
    }
//    res_max = residual;
//    GA_Dgop(&res_max, 1, "max");
    if (me == 0) {
      printf("Iteration: %d Residual: %f\n",iter,residual);
    }
//    GA_Sync();
    iter++;
  }

  NGA_Release64(g_b,&my_lo, &my_hi);
  NGA_Release_update64(*g_x,&my_lo, &my_hi);
  free(m_idx);
  free(m_jdx);
  free(m_jlo);
  free(m_vals);
  free(blk_ptrs);
  free(xblocks);
  free(seq);
  free(list);
}

int main(int argc, char **argv) {
  int s_a, g_b, g_x, g_p, g_r, g_t;
  int g_s, g_v, g_rm;
  int one;
  int64_t one_64;
  int me, nproc;
  int idim, jdim, kdim;
  int64_t xdim, ydim, zdim;
  int64_t rdim, cdim, rdx, cdx;
  int64_t ldx, ldxy;
  int ipx, ipy, ipz, idx, idy, idz;
  int64_t ilo, ihi, jlo, jhi, klo, khi;
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
  double alpha, beta, rho, rho_m, omega, m_omega, residual;
  double rv,ts,tt;
  int nsave;
  int heap=10000000, stack=10000000;
  int iterations = 10000;
  double tol, twopi;
  FILE *PHI;
  /* Intitialize a message passing library */
  one = 1;
  one_64 = 1;
  MP_INIT(argc,argv);

  /* Initialize GA */
  NGA_Initialize();

  /* initialize random number generator */
  x = ran3(-32823);

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
        rdx = i + j*ldx + k*ldxy;
        val = -(xinc_p+xinc_m+yinc_p+yinc_m+zinc_p+zinc_m)/(h*h);
        NGA_Sprs_array_add_element64(s_a,rdx,rdx,&val);
        if (i+1 < xdim) {
          cdx = i+1 + j*ldx + k*ldxy;
          val = xinc_p/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (i-1 >= 0) {
          cdx = i-1 + j*ldx + k*ldxy;
          val = xinc_m/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (j+1 < ydim) {
          cdx = i + (j+1)*ldx + k*ldxy;
          val = yinc_p/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (j-1 >= 0) {
          cdx = i + (j-1)*ldx + k*ldxy;
          val = yinc_m/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (k+1 < zdim) {
          cdx = i + j*ldx + (k+1)*ldxy;
          val = zinc_p/(h*h);
          NGA_Sprs_array_add_element64(s_a,rdx,cdx,&val);
        } else {
          ncnt++;
        }
        if (k-1 >= 0) {
          cdx = i + j*ldx + (k-1)*ldxy;
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
        ibuf[ncnt] = i + j*ldx;
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
        ibuf[ncnt] = i + j*ldx + (zdim-1)*ldxy;
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
        ibuf[ncnt] = i + k*ldxy;
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
        ibuf[ncnt] = i + (ydim-1)*ldx + k*ldxy;
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
        ibuf[ncnt] = j*ldx + k*ldxy;
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
        ibuf[ncnt] = (xdim-1) + j*ldx + k*ldxy;
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
  NGA_Sprs_array_row_distribution64(s_a, me, &ilo, &ihi);
  lmap[me] = (long)ilo;
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
  if (me == 0) {
    printf("\n    Right hand side completed\n");
  }

  /* Matrix and right hand side have been constructed. Begin solution
   * loop using Kaczmarz algorithm */

  k_solve(s_a, g_b, &g_x);

  /* Write solution to file */
#ifdef WRITE_VTK
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
      ilo = k*xdim*ydim;
      ihi = ilo + xdim*ydim - 1;
      NGA_Get64(g_x,&ilo,&ihi,vbuf,&one_64);
      for (j=0; j<ydim; j++) {
        for (i=0; i<xdim; i++) {
          fprintf(PHI," %12.6f",vbuf[i+j*xdim]);
          if (i%5 == 0) fprintf(PHI,"\n");
        }
        if ((xdim-1)%5 != 0) fprintf(PHI,"\n");
      }
    }
    fclose(PHI);
    free(vbuf);
  }
#endif

  NGA_Sprs_array_destroy(s_a);
  NGA_Destroy(g_b);
  NGA_Destroy(g_x);

  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
