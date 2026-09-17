#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif
#include <stdlib.h>

#define DIM 128
#define MINBLOCK 10

#include "ga.h"
#include "macdecls.h"
#include "mp3.h"

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

int main( int argc, char **argv ) {
  int nblocks[3] = {0, 0, 0};
  int inc, offset, ncnt, icnt;
  int *mapc;
  int total_blocks;
  int g_a;
  int ndim = 3;
  int dims[3] = {DIM,DIM,DIM};
  int ipx, ipy, ipz;
  int proc_grid[3];
  int iblock;
  int lo[3], hi[3];
  int ld[2] = {DIM, DIM};
  int i, j, k;
  int *chkbuf;
  int ok;

  int heap=3000000, stack=2000000;
  int me, nproc;

  MP_INIT(argc,argv);

  GA_INIT(argc,argv);                            /* initialize GA */
  me=GA_Nodeid(); 
  nproc=GA_Nnodes();
  if(me==0) {
    if(GA_Uses_fapi())GA_Error("Program runs with C array API only",1);
    printf("Testing tiled irregular data distributions\n");
    printf("\nUsing %ld processes\n",(long)nproc);
    fflush(stdout);
  }
  heap /= nproc;
  stack /= nproc;
  if(! MA_init(MT_F_DBL, stack, heap)) 
    GA_Error("MA_init failed",stack+heap);  /* initialize memory allocator*/ 

  /* Create an irregular tiled array. */
  if(me==0)printf("\nCreating of size %d x %d x %d\n",DIM,DIM,DIM);
  /* Figure out how many blocks in each dimension */
  for (i=0; i<3; i++) {
    inc = i;
    offset = 0;
    while (offset<DIM) {
      nblocks[i]++;
      offset += MINBLOCK+inc;
      inc++;
      inc = inc%3;
    }
  }
  /* create map array */
  ncnt = 0;
  for (i=0; i<3; i++) ncnt += dims[i];
  mapc = (int*)malloc(ncnt*sizeof(int));
  ncnt = 0;
  for (i=0; i<3; i++) {
    inc = i;
    offset = 0;
    while (offset<DIM) {
      mapc[ncnt] = offset;
      offset += MINBLOCK+inc;
      inc++;
      inc = inc%3;
      ncnt++;
    }
  }
  grid_factor(nproc,DIM,DIM,DIM,&ipx,&ipy,&ipz);
  proc_grid[0] = ipx;
  proc_grid[1] = ipy;
  proc_grid[2] = ipz;
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a,ndim,dims,C_INT);
  NGA_Set_tiled_irreg_proc_grid(g_a,mapc,nblocks,proc_grid);
  NGA_Allocate(g_a);
  GA_Zero(g_a);

  if (me == 0) {
    printf("\nTesting PUT operation on individual tiles\n");
  }
  /* fill array with data. Each processor fills data held by next higher
   * processor */
  iblock = (me+1)%nproc;
  total_blocks = 1;
  for (i=0; i<ndim; i++) total_blocks *= nblocks[i];
  while (iblock < total_blocks) {
    int nelems = 1;
    int *buf;
    NGA_Distribution(g_a, iblock, lo, hi);
    for (i=0; i<ndim; i++) nelems *= (hi[i]-lo[i]+1);
    buf = (int*)malloc(nelems*sizeof(int)); 
    icnt = 0;
    for (i=lo[0]; i<=hi[0]; i++) {
      for (j=lo[1]; j<=hi[1]; j++) {
        for (k=lo[2]; k<=hi[2]; k++) {
          ncnt = k+DIM*j+DIM*DIM*i;
          buf[icnt] = ncnt;
          icnt++;
        }
      }
    }
    ld[0] = hi[1]-lo[1]+1;
    ld[1] = hi[2]-lo[2]+1;
    NGA_Put(g_a,lo,hi,buf,ld);
    free(buf);
    iblock += nproc;
  }
  NGA_Sync();

  /* copy entire array to local buffer and check that values
   * are correct */
  lo[0] = 0;
  lo[1] = 0;
  lo[2] = 0;
  hi[0] = DIM-1;
  hi[1] = DIM-1;
  hi[2] = DIM-1;
  ld[0] = DIM;
  ld[1] = DIM;
  chkbuf = (int*)malloc(DIM*DIM*DIM*sizeof(int));
  NGA_Get(g_a,lo,hi,chkbuf,ld);
  ok = 1;
  icnt = 0;
  for (i=lo[0]; i<=hi[0]; i++) {
    for (j=lo[1]; j<=hi[1]; j++) {
      for (k=lo[2]; k<=hi[2]; k++) {
        ncnt = k+DIM*j+DIM*DIM*i;
        if (chkbuf[icnt] != ncnt) {
          if (ok) {
            printf("p[%d] checking PUT expected: %d actual: %d\n", me,
                ncnt, chkbuf[icnt]);
            ok = 0;
          }
        }
        icnt++;
      }
    }
  }
  free(chkbuf);
  if (ok && me == 0) {
    printf("\nPUT is okay\n");
  } else if (!ok) {
    printf("\nPUT FAILS\n");
  }
  GA_Sync();

  if (me == 0) {
    printf("\nTesting GET operation on individual tiles\n");
  }
  /* get data from array and check for correctness */
  iblock = (me+1)%nproc;
  ok = 1;
  while (iblock < total_blocks) {
    int nelems = 1;
    int *buf;
    NGA_Distribution(g_a, iblock, lo, hi);
    for (i=0; i<ndim; i++) nelems *= (hi[i]-lo[i]+1);
    buf = (int*)malloc(nelems*sizeof(int)); 
    ld[0] = hi[1]-lo[1]+1;
    ld[1] = hi[2]-lo[2]+1;
    NGA_Get(g_a,lo,hi,buf,ld);
    icnt = 0;
    for (i=lo[0]; i<=hi[0]; i++) {
      for (j=lo[1]; j<=hi[1]; j++) {
        for (k=lo[2]; k<=hi[2]; k++) {
          ncnt = k+DIM*j+DIM*DIM*i;
          if (buf[icnt] != ncnt) {
            if (ok) {
              printf("p[%d] checking GET expected: %d actual: %d\n", me,
                  ncnt, buf[icnt]);
              ok = 0;
            }
          }
          icnt++;
        }
      }
    }
    free(buf);
    iblock += nproc;
  }
  if (ok && me == 0) {
    printf("\nGET is okay\n");
  } else if (!ok) {
    printf("\nGET FAILS\n");
  }
  GA_Sync();

  if (me == 0) {
    printf("\nTesting ACC operation on individual tiles\n");
  }
  /* Accumulate data */
  iblock = (me+1)%nproc;
  while (iblock < total_blocks) {
    int nelems = 1;
    int *buf;
    int scale = 1;
    NGA_Distribution(g_a, iblock, lo, hi);
    for (i=0; i<ndim; i++) nelems *= (hi[i]-lo[i]+1);
    buf = (int*)malloc(nelems*sizeof(int)); 
    icnt = 0;
    for (i=lo[0]; i<=hi[0]; i++) {
      for (j=lo[1]; j<=hi[1]; j++) {
        for (k=lo[2]; k<=hi[2]; k++) {
          ncnt = k+DIM*j+DIM*DIM*i;
          buf[icnt] = ncnt;
          icnt++;
        }
      }
    }
    ld[0] = hi[1]-lo[1]+1;
    ld[1] = hi[2]-lo[2]+1;
    NGA_Acc(g_a,lo,hi,buf,ld,&scale);
    free(buf);
    iblock += nproc;
  }
  GA_Sync();

  /* get data from array and check for correctness */
  iblock = (me+1)%nproc;
  ok = 1;
  while (iblock < total_blocks) {
    int nelems = 1;
    int *buf;
    NGA_Distribution(g_a, iblock, lo, hi);
    for (i=0; i<ndim; i++) nelems *= (hi[i]-lo[i]+1);
    buf = (int*)malloc(nelems*sizeof(int)); 
    ld[0] = hi[1]-lo[1]+1;
    ld[1] = hi[2]-lo[2]+1;
    NGA_Get(g_a,lo,hi,buf,ld);
    icnt = 0;
    for (i=lo[0]; i<=hi[0]; i++) {
      for (j=lo[1]; j<=hi[1]; j++) {
        for (k=lo[2]; k<=hi[2]; k++) {
          ncnt = 2*(k+DIM*j+DIM*DIM*i);
          if (buf[icnt] != ncnt) {
            if (ok) {
              printf("p[%d] checking GET expected: %d actual: %d\n", me,
                  ncnt, buf[icnt]);
              ok = 0;
            }
          }
          icnt++;
        }
      }
    }
    free(buf);
    iblock += nproc;
  }
  if (ok && me == 0) {
    printf("\nACC is okay\n");
  } else if (!ok) {
    printf("\nACC FAILS\n");
  }
  GA_Sync();

  GA_Destroy(g_a);
  if(me==0)printf("\nSuccess\n");
  GA_Terminate();

  MP_FINALIZE();

 return 0;
}
