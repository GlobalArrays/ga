#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

//#define WRITE_VTK
#define CG_SOLVE 0
#define NDIM 64

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

/**
 * variables for executing asynchronous dot products
 */
typedef struct {
  double *a_vec_old; /* old values of vector A on this processor */
  double *b_vec_old; /* old values of vector B on this processor */
  double *a_vec_new; /* new values of vector A on this processor */
  double *b_vec_new; /* new values of vector B on this processor */
  double *dot_buf; /* array for holding increments */
  double *dot; /* current value of dot product */
  int g_a; /* handle of array A */
  int g_b; /* handle of array B */
  int g_dot;   /* global array containing dot product */
  int64_t vlen;    /* length of vectors A and B on this processor */
  int actv;  /* flag indicating whether this handle is active */
} async_dot_struct;


/**
 * Array of asynchronous dot product structures
 */
#define MAX_DOT_HANDLES 100
async_dot_struct _dot_handles[MAX_DOT_HANDLES];

/**
 * Initialize asynchronous dot product functionality
 */
void init_async_dot()
{
  int i;
  for (i=0; i<MAX_DOT_HANDLES; i++) {
    _dot_handles[i].actv = 0;
    _dot_handles[i].a_vec_old = NULL;
    _dot_handles[i].b_vec_old = NULL;
    _dot_handles[i].a_vec_new = NULL;
    _dot_handles[i].b_vec_new = NULL;
    _dot_handles[i].dot_buf = NULL;
    _dot_handles[i].dot = NULL;
  }
}

/**
 * g_a, g_b: handles of 1D global arrays that will be used for dot product
 * dot: initial value of dot product
 * return: handle for asynchronous dot product object
 */
int new_async_dot(int g_a, int g_b, double *dot)
{
  int64_t alo, ahi, blo, bhi, ld;
  double *aptr_new;
  double *bptr_new;
  double *aptr_old;
  double *bptr_old;
  double *dot_buf;
  int64_t vlen;
  int g_dot;
  int64_t i;
  int j;
  int me = GA_Nodeid();
  int64_t nprocs = GA_Nnodes();
  int64_t one = 1;
  int handle = -1;
  int64_t lme = me;
  /* Find an unused handle */
  for (j=0; j<MAX_DOT_HANDLES; j++) {
    if (_dot_handles[j].actv == 0) {
      handle = j;
      _dot_handles[j].actv = 1;
      break;
    }
  }
  if (handle == -1) {
    _dot_handles[handle].actv = 0;
    GA_Error("No handles available for asynchronous dot product",0);
  }
  _dot_handles[handle].g_a = g_a;
  _dot_handles[handle].g_b = g_b;

  NGA_Distribution64(g_a, me, &alo, &ahi);
  NGA_Distribution64(g_b, me, &blo, &bhi);
  _dot_handles[handle].vlen = ahi-alo+1;
  vlen = _dot_handles[handle].vlen;
  if (vlen != bhi-blo+1) {
    GA_Error("Error (init_async_doc): Vector distributions must be the same",0);
  }
  aptr_old = (double*)malloc(vlen*sizeof(double));
  bptr_old = (double*)malloc(vlen*sizeof(double));
  aptr_new = (double*)malloc(vlen*sizeof(double));
  bptr_new = (double*)malloc(vlen*sizeof(double));
  dot_buf = (double*)malloc(nprocs*sizeof(double));
  _dot_handles[handle].a_vec_old = aptr_old;
  _dot_handles[handle].b_vec_old = bptr_old;
  _dot_handles[handle].a_vec_new = aptr_new;
  _dot_handles[handle].b_vec_new = bptr_new;
  _dot_handles[handle].dot_buf = dot_buf;
  
  NGA_Get64(g_a,&alo,&ahi,aptr_new,&ld);
  NGA_Get64(g_b,&blo,&bhi,bptr_new,&ld);
  for (i=0; i<vlen; i++) {
    aptr_old[i] = aptr_new[i];
    bptr_old[i] = bptr_new[i];
  }
  /* create a global array with one element per processor */
  g_dot = NGA_Create_handle();
  NGA_Set_data64(g_dot, 1, &nprocs, C_DBL);
  NGA_Set_chunk64(g_dot, &one);
  NGA_Allocate(g_dot);
  NGA_Access64(g_dot, &lme, &lme, &_dot_handles[handle].dot, &ld);
  _dot_handles[handle].g_dot = g_dot;
  *_dot_handles[handle].dot = GA_Ddot(g_a, g_b);
  *dot = *_dot_handles[handle].dot;
  GA_Sync();
  return handle;
}

/**
 * Calculate current value of asynchronous dot product
 * handle: asynchronous dot product handle
 * dot: estimated value of current dot product
 */
void async_dot(int handle, double *dot)
{
  double *a_vec_old, *b_vec_old, *dot_buf;
  double *a_vec_new, *b_vec_new;
  int64_t alo, ahi, blo, bhi, ld;
  int64_t vlen;
  int lo, hi, dd;
  int me = GA_Nodeid();
  int nprocs = GA_Nnodes();
  int i;
  int g_dot, g_a, g_b;
  double dot_inc, AB, ab;
  double one = 1.0;

  a_vec_old = _dot_handles[handle].a_vec_old;
  b_vec_old = _dot_handles[handle].b_vec_old;
  a_vec_new = _dot_handles[handle].a_vec_new;
  b_vec_new = _dot_handles[handle].b_vec_new;
  dot_buf = _dot_handles[handle].dot_buf;
  vlen = _dot_handles[handle].vlen;
  g_dot = _dot_handles[handle].g_dot;
  g_a = _dot_handles[handle].g_a;
  g_b = _dot_handles[handle].g_b;
#if 0
  *dot = GA_Ddot(g_a,g_b);
#else
  /*
  {
    double tmpbuf[4];
    int ilo = 0;
    int ihi = nprocs-1;
    NGA_Get(g_a,&ilo,&ihi,tmpbuf,&dd);
    printf("p[%d] g_a:",me);
    for (i=0; i<nprocs; i++) {
      printf(" %16.8e",tmpbuf[i]);
    }
    printf("\n");
    NGA_Get(g_a,&ilo,&ihi,tmpbuf,&dd);
    printf("p[%d] g_b:",me);
    for (i=0; i<nprocs; i++) {
      printf(" %16.8e",tmpbuf[i]);
    }
    printf("\n");
  }
  */

  NGA_Distribution64(g_a,me,&alo,&ahi);
  NGA_Distribution64(g_b,me,&blo,&bhi);
  NGA_Get64(g_a, &alo, &ahi, a_vec_new, &ld);
  NGA_Get64(g_b, &blo, &bhi, b_vec_new, &ld);
  dot_inc = 0.0;
  AB = 0.0;
  ab = 0.0;
  for (i=0; i<vlen; i++) {
    AB += a_vec_old[i]*b_vec_old[i];
    ab += a_vec_new[i]*b_vec_new[i];
    a_vec_old[i] = a_vec_new[i];
    b_vec_old[i] = b_vec_new[i];
  }
  dot_inc = ab-AB;
  /*
  printf("p[%d] nprocs: %d handle: %d ab: %e AB: %e dot_inc: %e\n",me,nprocs,handle,ab,AB,dot_inc);
  printf("p[%d]",me);
  */
  for (i=0; i<nprocs; i++) {
    dot_buf[i] = dot_inc;
    /*
    printf(" %e",dot_buf[i]);
    */
  }
  /*
  printf(" one: %f\n",one);
  */
  lo = 0;
  hi = nprocs-1;
  /* Add increment to copy of dot product on every processor */
  NGA_Acc(g_dot,&lo,&hi,dot_buf,&dd,&one);
  //GA_Sync();
  *dot = *_dot_handles[handle].dot;
  /*
  printf("p[%d] *dot: %e\n",me,*dot);
  lo = me;
  hi = me;
  NGA_Get(g_dot,&lo,&hi,dot,&dd);
  printf("p[%d] lo: %d hi: %d dot: %e\n",me,lo,hi,*dot);
  */
#endif
  if (me==0) printf("p[%d] g_dot: %d dot product: %e\n",me,g_dot,*dot);
}

/**
 * Clean up asynchronous dot product object
 * handle: handle of asynchronous dot product
 */
void destroy_async_dot(int handle)
{
  int64_t me = GA_Nodeid();
  int g_dot = _dot_handles[handle].g_dot;
  free(_dot_handles[handle].a_vec_new);
  free(_dot_handles[handle].b_vec_new);
  free(_dot_handles[handle].a_vec_old);
  free(_dot_handles[handle].b_vec_old);
  free(_dot_handles[handle].dot_buf);
  NGA_Release64(g_dot, &me, &me);
  NGA_Destroy(g_dot);
  _dot_handles[handle].a_vec_new = NULL;
  _dot_handles[handle].b_vec_new = NULL;
  _dot_handles[handle].a_vec_old = NULL;
  _dot_handles[handle].b_vec_old = NULL;
  _dot_handles[handle].dot_buf = NULL;
  _dot_handles[handle].dot = NULL;
  _dot_handles[handle].actv = 0;
}

void terminate_async_dot()
{
  int i;
  for (i=0; i<MAX_DOT_HANDLES; i++) {
    if (_dot_handles[i].actv) {
      destroy_async_dot(i);
    }
  }
}

/**
 * Clean up asynchrous dot product module
 */

int main(int argc, char **argv) {
  int s_a, g_b, g_x, g_p, g_r, g_t;
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
  int64_t *ibuf, **iptr;
  double *vptr;
  double *vbuf;
  int ok;
  double one_r = 1.0;
  double m_one_r = -1.0;
  double ir, jr, ldr;
  double xinc_p, yinc_p, zinc_p;
  double xinc_m, yinc_m, zinc_m;
  double alpha, beta, residual;
  int nsave;
  int heap=10000000, stack=10000000;
  int iterations = 10000;
  double tol, twopi;
  FILE *PHI;
  int d_rr, d_tp, d_rho, d_rv, d_ts, d_tt;
  /* Intitialize a message passing library */
  one = 1;
  one_64 = 1;
  MP_INIT(argc,argv);

  /* Initialize GA */
  NGA_Initialize();
  init_async_dot();

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
  idx = i%ipx;
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
   * boundaries */
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

  /* allocate global array representing right hand side vector */
  g_b = NGA_Create_handle();
  NGA_Set_data64(g_b,one,&cdim,C_DBL);
  NGA_Allocate(g_b);
  GA_Zero(g_b);
  /* accumulate boundary values to right hand side vector */
  NGA_Scatter_acc64(g_b,vbuf,iptr,ncnt,&one_r);
  GA_Sync();
  free(ibuf);
  free(iptr);
  free(vbuf);
#define SYNCHED 0
#if CG_SOLVE
  g_x = GA_Duplicate(g_b, "dup_x");
  g_r = GA_Duplicate(g_b, "dup_r");
  g_p = GA_Duplicate(g_b, "dup_p");
  g_t = GA_Duplicate(g_b, "dup_t");
  if (me == 0) {
    printf("\nRight hand side vector completed. Starting\n");
    printf("conjugate gradient iterations.\n\n");
  }

  /* Solve Laplace's equation using conjugate gradient method */
  one_r = 1.0;
  m_one_r = -1.0;
  GA_Zero(g_x);
  GA_Zero(g_t);
  /* Initial guess is zero, so Ax = 0 and r = b */
  GA_Copy(g_b, g_r);
  GA_Copy(g_r, g_p);
#if SYNCHED
  residual = GA_Ddot(g_r,g_r);
  GA_Norm_infinity(g_r, &tol);
#else
  /* create asynchronous dot product handles */
  d_rr = new_async_dot(g_r,g_r,&residual);
  d_tp = new_async_dot(g_t,g_p,&alpha);
  tol =residual/((double)cdim);
#endif
  ncnt = 0;
  /* Start iteration loop */
  while (tol > 1.0e-5 && ncnt < iterations) {
    if (me==0) printf("Iteration: %d Tolerance: %e\n",(int)ncnt+1,tol);
    NGA_Sprs_array_matvec_multiply(s_a, g_p, g_t);
    //printf("p[%d] Got to 1\n",me);
#if SYNCHED
    alpha = GA_Ddot(g_t,g_p);
#else
    async_dot(d_tp,&alpha);
#endif
    //printf("p[%d] Alpha1: %e\n",me,alpha);
    alpha = residual/alpha;
    //printf("p[%d] Alpha2: %e\n",me,alpha);
    GA_Add(&one_r,g_x,&alpha,g_p,g_x);
    alpha = -alpha;
    GA_Add(&one_r,g_r,&alpha,g_t,g_r);
#if SYNCHED
    GA_Norm_infinity(g_r, &tol);
#endif
    beta = residual;
#if SYNCHED
    residual = GA_Ddot(g_r,g_r);
#else
    async_dot(d_rr,&residual);
#endif
    if (me==0) printf("p[%d] Residual: %e\n",me,residual);
    tol =residual/((double)cdim);
    beta = residual/beta;
    GA_Add(&one_r,g_r,&beta,g_p,g_p); 
    ncnt++;
  }
  /*
  if (me==0) printf("RHS Vector\n");
  GA_Print(g_b);
  if (me==0) printf("Solution Vector\n");
  GA_Print(g_x);
  */
  
  if (ncnt == iterations) {
    if (me==0) printf("Solution failed to converge\n");
  } else {
    if (me==0) printf("Solution converged\n");
  }
  destroy_async_dot(d_tp);
  destroy_async_dot(d_rr);
  NGA_Destroy(g_r);
  NGA_Destroy(g_p);
  NGA_Destroy(g_t);
#else
  /**
   * Based on algorithm described on page 136 in
   * Iterative Krylov Methods for Large Linear Systems, Henk A. van der Vorst,
   * Cambridge University Press, Cambridge, 2003.
   */
  g_x = GA_Duplicate(g_b, "dup_x");
  g_r = GA_Duplicate(g_b, "dup_r");
  g_rm = GA_Duplicate(g_b, "dup_rm");
  g_p = GA_Duplicate(g_b, "dup_p");
  g_v = GA_Duplicate(g_b, "dup_v");
  g_s = GA_Duplicate(g_b, "dup_s");
  g_t = GA_Duplicate(g_b, "dup_t");
  /* accumulate boundary values to right hand side vector */
  if (me == 0) {
    printf("\nRight hand side vector completed. Starting\n");
    printf("BiCG-STAB iterations.\n\n");
  }

  /* Solve Laplace's equation using conjugate gradient method */
  one_r = 1.0;
  m_one_r = -1.0;
  GA_Zero(g_x);
  /* Initial guess is zero, so Ax = 0 and r = b */
  GA_Copy(g_b, g_r);
  GA_Copy(g_b, g_rm);
  ncnt = 0;
#if SYNCHED
  GA_Norm_infinity(g_r, &tol);
#endif
  /* Start iteration loop */
  while (tol > 1.0e-5 && ncnt < iterations) {
    if (me==0) printf("Iteration: %d Tolerance: %e\n",(int)ncnt+1,tol);
#if SYNCHED
    rho = GA_Ddot(g_r,g_rm);
#else
    if (ncnt == 0) {
      d_rho = new_async_dot(g_r,g_rm,&rho);
    } else {
      async_dot(d_rho,&rho);
    }
#endif
    if (rho == 0.0) {
      GA_Error("BiCG-STAB method fails",0);
    }
    if (ncnt == 0) {
      GA_Copy(g_rm,g_p);
    } else {
      beta = (rho/rho_m)*(alpha/omega);
      m_omega = -omega;
      GA_Add(&one_r,g_p,&m_omega,g_v,g_p);
      GA_Add(&one_r,g_rm,&beta,g_p,g_p);
    }
    NGA_Sprs_array_matvec_multiply(s_a, g_p, g_v);
#if SYNCHED
    rv = GA_Ddot(g_r,g_v);
#else
    if (ncnt == 0) {
      d_rv = new_async_dot(g_r,g_v,&rv);
    } else {
      async_dot(d_rv,&rv);
    }
#endif
    alpha = -rho/rv;
    GA_Add(&one_r,g_rm,&alpha,g_v,g_s);
    alpha = -alpha;
    GA_Norm_infinity(g_s, &tol);
    if (tol < 1.0e-05) {
      GA_Add(&one_r,g_x,&alpha,g_p,g_x);
      break;
    }
    NGA_Sprs_array_matvec_multiply(s_a, g_s, g_t);
#if SYNCHED
    ts = GA_Ddot(g_t,g_s);
    tt = GA_Ddot(g_t,g_t);
#else
    if (ncnt == 0) {
      d_ts = new_async_dot(g_t,g_s,&ts);
      d_tt = new_async_dot(g_t,g_t,&tt);
    } else {
      async_dot(d_ts,&ts);
      async_dot(d_tt,&tt);
    }
#endif
    omega = ts/tt;
    m_omega = -omega;
    GA_Add(&one_r,g_x,&alpha,g_p,g_x);
    GA_Add(&one_r,g_x,&omega,g_s,g_x);
    GA_Add(&one_r,g_s,&m_omega,g_t,g_rm);
    GA_Norm_infinity(g_rm, &tol);
    if (tol < 1.0e-05) break;
    if (omega == 0) {
      GA_Error("BiCG-STAB method cannot continue",0);
    }
    ncnt++;
    rho_m = rho;
  }
  destroy_async_dot(d_rho);
  destroy_async_dot(d_rv);
  destroy_async_dot(d_ts);
  destroy_async_dot(d_tt);
  NGA_Destroy(g_x);
  NGA_Destroy(g_r);
  NGA_Destroy(g_rm);
  NGA_Destroy(g_p);
  NGA_Destroy(g_v);
  NGA_Destroy(g_s);
  NGA_Destroy(g_t);
#endif

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
  NGA_Destroy(g_r);
  NGA_Destroy(g_p);
  NGA_Destroy(g_t);

  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
