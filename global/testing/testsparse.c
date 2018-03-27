#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

#define NDIM 8

#define MAX_FACTOR 1024
void grid_factor(int p, int xdim, int ydim, int *idx, int *idy) {
  int i, j; 
  int ip, ifac, pmax, prime[MAX_FACTOR];
  int fac[MAX_FACTOR];
  int ix, iy, ichk;

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
 *    find two factors of p of approximately the same size
 */
  *idx = 1;
  *idy = 1;
  for (i = ifac-1; i >= 0; i--) {
    ix = xdim/(*idx);
    iy = ydim/(*idy);
    if (ix >= iy && ix > 1) {
      *idx = fac[i]*(*idx);
    } else if (iy >= ix && iy > 1) {
      *idy = fac[i]*(*idy);
    } else {
      printf("Too many processors in grid factoring routine\n");
    }
  }
}

int main(int argc, char **argv) {
  int s_a, g_a;
  int one;
  int me, nproc;
  int xdim, ydim, ipx, ipy, idx, idy;
  int ilo, ihi, jlo, jhi;
  int i, j, iproc, ld, ncols;
  double val;
  double *vptr;
  long *iptr = NULL, *jptr = NULL;
  /* Intitialize a message passing library */
  one = 1;
  MP_INIT(argc,argv);
  /* Initialize GA */
  NGA_Initialize();
  printf("p[%d] Got to 1\n",me);

  xdim = NDIM;
  ydim = NDIM;
  me = GA_Nodeid();
  nproc = GA_Nnodes();
  printf("p[%d] Got to 2\n",me);

  /* factor array */
  grid_factor(nproc, xdim, ydim, &ipx, &ipy);
  idx = ipy;
  ipy = ipx;
  ipx = idx;
  printf("p[%d] Got to 3 ipx: %d ipy: %d\n",me,ipx,ipy);
  /* figure out process location in proc grid */
  idx = me%ipx;
  idy = (me-idx)/ipx;
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
  printf("p[%d] Got to 4 ilo: %d ihi: %d jlo: %d jhi: %d\n",me,ilo,ihi,jlo,jhi);
 
  /* create sparse array */
  s_a = NGA_Sprs_array_create(xdim, ydim, C_DBL);
  printf("p[%d] Got to 5\n",me);
  if (ydim%2 == 0) {
    ld = ydim/2;
  } else {
    ld = (ydim-1)/2+1;
  }
  /* add elements to array. Every other element is zero */
  for (i=ilo; i<=ihi; i++) {
    for (j=jlo; j<=jhi; j++) {
      if (i%2 == 0 && j%2 == 0) {
        val = (double)((i/2)*ld+j/2);
        NGA_Sprs_array_add_element(s_a,i,j,&val);
      }
    }
  }
  printf("p[%d] Got to 6\n",me);
  NGA_Sprs_array_assemble(s_a);
  printf("p[%d] Got to 7\n",me);

  /* access array blocks an check values for correctness */
  NGA_Sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  printf("p[%d] Got to 8 sizeof(long): %d ilo: %d ihi: %d\n",me,sizeof(long),ilo,ihi);
  for (iproc=0; iproc<nproc; iproc++) {
    NGA_Sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    printf("p[%d] Got to 9 jlo: %d jhi: %d iproc: %d\n",me,jlo,jhi,iproc);
    NGA_Sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
    printf("p[%d] Got to 10 iptr: %p jptr: %p vptr: %p\n",me,iptr,jptr,vptr);
    for (i=0; i<4; i++) {
      printf("i: %d j:%d v: %f\n",iptr[i],jptr[i],vptr[i]);
    }
    if (vptr != NULL) {
      for (i=ilo; i<=ihi; i++) {
        ncols = iptr[i+1-ilo]-iptr[i-ilo];
        printf("p[%d] iptr[%d]: %d iptr[%d]: %d\n",me,i+1-ilo,iptr[i+1-ilo],
            i-ilo,iptr[i-ilo]);
        for (j=0; j<ncols; j++) {
          printf("p[%d] i: %d j: %d val: %f\n",me,i,jptr[iptr[i-ilo]+j],vptr[iptr[i-ilo]+j]);
        }
      }
    }
  }


  NGA_Sprs_array_destroy(s_a);

  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
