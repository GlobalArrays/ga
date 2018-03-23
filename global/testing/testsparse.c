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
  int i, j, ld;
  double val;
  /* Intitialize a message passing library */
  one = 1;
  MP_INIT(argc,argv);
  /* Initialize GA */
  NGA_Initialize();
  printf("Got to 1\n");

  xdim = NDIM;
  ydim = NDIM;
  me = GA_Nodeid();
  nproc = GA_Nnodes();
  printf("Got to 2\n");

  /* factor array */
  grid_factor(nproc, xdim, ydim, &ipx, &ipy);
  printf("Got to 3\n");
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
  printf("Got to 4\n");
 
  /* create sparse array */
  s_a = NGA_Sprs_array_create(xdim, ydim, C_DBL);
  printf("Got to 5\n");
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
  printf("Got to 6\n");
  NGA_Sprs_array_assemble(s_a);
  printf("Got to 7\n");


  NGA_Sprs_array_destroy(s_a);
  printf("Got to 8\n");

  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
