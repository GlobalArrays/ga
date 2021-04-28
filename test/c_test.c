#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "mpi.h"
#include "ga.h"
#include "macdecls.h"

#define DIM 2
#define DIMSIZE 4
#define MAX_FACTOR 256

void factor(int p, int *idx, int *idy) {
  int i, j;                              
  int ip, ifac, pmax;                    
  int prime[MAX_FACTOR];                 
  int fac[MAX_FACTOR];                   
  int ix, iy;                            
  int ichk;                              

  i = 1;

 //find all prime numbers, besides 1, less than or equal to the square root of p
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

 //find all prime factors of p
  ip = p;
  ifac = 0;
  for (i=0; i<pmax; i++) {
    while(ip%prime[i] == 0) {
      ifac = ifac + 1;
      fac[ifac-1] = prime[i];
      ip = ip/prime[i];
    }
  }

 //when p itself is prime
  if (ifac==0) {
    ifac++;
    fac[0] = p;
  }


 //find two factors of p of approximately the same size
  *idx = 1;
  *idy = 1;
  for (i = ifac-1; i >= 0; i--) {
    ix = *idx;
    iy = *idy;
    if (ix <= iy) {
      *idx = fac[i]*(*idx);
    } else {
      *idy = fac[i]*(*idy);
    }
  }
}

int main(int argc, char **argv) {

  int g_a;
  int i, j, ii, jj, idx;
  int rank, nprocs;
  int ipx, ipy;
  int pdx, pdy;
  int xinc, yinc;
  int ndim, nsize;
  int dims[2], lo[2], hi[2];
  int tld, tlo[2], thi[2];
  int *buf;
  int ld;
  int ok;
  int *ptr;
  int nelem;
  

  MPI_Init(&argc, &argv);


  GA_Initialize();

  nprocs = GA_Nnodes();  
  rank = GA_Nodeid();   
  printf("Found GA rank %d\n",rank);

  /* Divide matrix up into pieces that are owned by each processor */
  factor(nprocs, &pdx, &pdy);

  ndim = 2;
  dims[0] = DIMSIZE;
  dims[1] = DIMSIZE;
  xinc = DIMSIZE/pdx;
  yinc = DIMSIZE/pdy;
  ipx = rank%pdx;
  ipy = (rank-ipx)/pdx;
  lo[0] = ipx*xinc;
  lo[1] = ipy*yinc;
  if (ipx<pdx-1) {
    hi[0] = (ipx+1)*xinc-1;
  } else {
    hi[0] = DIMSIZE-1;
  }
  if (ipy<pdy-1) {
    hi[1] = (ipy+1)*yinc-1;
  } else {
    hi[1] = DIMSIZE-1;
  }
  nelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  printf("p[%d] nelem: %d\n",rank,nelem);

  /* create a global array and initialize it to zero */
  g_a = NGA_Create_handle();
  if (rank == 0) printf("Created GA handle\n");
  NGA_Set_data(g_a, ndim, dims, C_INT);
  if (rank == 0) printf("Set GA data\n");
  NGA_Set_device(g_a, 1);
  if (rank == 0) printf("Set GA device\n");
  NGA_Allocate(g_a);
  if (rank == 0) printf("Completed GA allocate\n");
  GA_Zero(g_a);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  buf = (int*)malloc(nsize*sizeof(int));
  ld = (hi[1]-lo[1]+1);
  printf("p[%d] lo[0]: %d hi[0]: %d lo[1]: %d hi[1]: %d\n",rank,lo[0],hi[0],lo[1],hi[1]);
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      buf[idx] = ii*DIMSIZE+jj;
    }
  }
  printf("p[%d] buf:",rank);
  for (i=0; i<nelem; i++) {
    printf(" %d",buf[i]);
  }
  printf("\n");

  printf("p[%d] Call NGA_Put\n",rank);
  /* copy data to global array */
  NGA_Put(g_a, lo, hi, buf, &ld);
  printf("p[%d] Completed NGA_Put\n",rank);
  GA_Sync();
#if 0
  NGA_Distribution(g_a,rank,tlo,thi);
  printf("p[%d] Completed NGA_Distribution\n",rank);
  if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
    int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
    NGA_Access(g_a,tlo,thi,&ptr,&tld);
    printf("p[%d] Completed NGA_Access lo[0]: %d hi[0]: %d lo[1]: %d hi[1]: %d\n",
        rank,tlo[0],thi[0],tlo[1],thi[1]);
    ok = 1;
    for (ii = tlo[0]; ii<=thi[0]; ii++) {
      i = ii-tlo[0];
      for (jj=tlo[1]; jj<=thi[1]; jj++) {
        j = jj-tlo[1];
        idx = i*tld+j;
        if (ptr[idx] != ii*DIMSIZE+jj) {
          if (ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %d\n",rank,ii,jj,ii*DIMSIZE+jj,
              idx,ptr[idx]);
          ok = 0;
        }
      }
    }
    if (!ok) {
      printf("Mismatch found on process %d after Put\n",rank);
    } else {
      printf("Put is okay on process %d\n",rank);
    }
    printf("p[%d] ptr:",rank);
    for (i=0; i<tnelem; i++) {
      printf(" %d",ptr[i]);
    }
    printf("\n");
  }
#endif

  /* zero out local buffer */
  for (i=0; i<nsize; i++) buf[i] = 0;

  /* copy data from global array to local buffer */
  printf("p[%d] Call NGA_Get\n",rank);
  NGA_Get(g_a, lo, hi, buf, &ld);
  printf("p[%d] Completed NGA_Get\n",rank);
  GA_Sync();
#if 1

  printf("p[%d] local patch lo[0]: %d hi[0]: %d lo[1]: %d hi[1]: %d\n",rank,
      lo[0],hi[0],lo[1],hi[1]);
  ok = 1;
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      if (buf[idx] != ii*DIMSIZE+jj) {
        if (ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %d\n",rank,ii,jj,ii*DIMSIZE+jj,
            idx,buf[idx]);
        ok = 0;
      }
    }
  }
  if (!ok) {
    printf("Mismatch found on process %d after Get\n",rank);
  } else {
    if (rank == 0) printf("Get is okay\n");
  }

#endif
  GA_Destroy(g_a);
  if (rank == 0) printf("Completed GA_Destroy\n");
  GA_Terminate();
  if (rank == 0) printf("Completed GA_Terminate\n");
  MPI_Finalize();
}
