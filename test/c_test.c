#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "mpi.h"
#include "ga.h"
#include "macdecls.h"
#include <cuda_runtime.h>

#define DIM 1024
#define DIMSIZE 256
#define MAXCOUNT 10000
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
  int isx, isy;
  int pdx, pdy;
  int xinc, yinc;
  int ndim, nsize;
  int dims[2], lo[2], hi[2];
  int tld, tlo[2], thi[2];
  int *buf;
  double *rbuf;
  int ld;
  int ok;
  int *ptr;
  int nelem;
  int one;
  double one_r;
  int zero = 0;
  int icnt;
  

  MPI_Init(&argc, &argv);


  GA_Initialize();

  nprocs = GA_Nnodes();  
  rank = GA_Nodeid();   

  /* Divide matrix up into pieces that are owned by each processor */
  factor(nprocs, &pdx, &pdy);
  if (rank == 0) {
    printf("  Test run on %d procs configured on %d X %d grid\n",nprocs,pdx,pdy);
  }
  if (rank == 0) printf("  Testing integer arrays\n");

  ndim = 2;
  dims[0] = DIMSIZE;
  dims[1] = DIMSIZE;
  xinc = DIMSIZE/pdx;
  yinc = DIMSIZE/pdy;
  ipx = rank%pdx;
  ipy = (rank-ipx)/pdx;
  isx = (ipx+1)%pdx;
  isy = (ipy+1)%pdy;
  /* Guarantee some data exchange between nodes */
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
  if (rank == 0) printf("Completed GA zero\n");

  /* printf("p[%d] Process owns [%d:%d][%d:%d]\n",rank,lo[0],hi[0],lo[1],hi[1]); */
  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  buf = (int*)malloc(nsize*sizeof(int));
  ld = (hi[1]-lo[1]+1);
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      buf[idx] = ii*DIMSIZE+jj;
    }
  }

  /* copy data to global array */
  NGA_Put(g_a, lo, hi, buf, &ld);
  if (rank == 0) printf("Completed GA put\n");
  GA_Sync();
  if (rank == 0) printf("Completed GA sync\n");
  NGA_Distribution(g_a,rank,tlo,thi);
#if 0
  if (rank == 0) printf("Completed NGA_Distribution\n",rank);
  if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
    int *tbuf;
    int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
    tbuf = (int*)malloc(tnelem*sizeof(int));
    NGA_Access(g_a,tlo,thi,&ptr,&tld);
    for (i=0; i<tnelem; i++) tbuf[i] = 0;
    cudaMemcpy(tbuf, ptr, tnelem*sizeof(int), cudaMemcpyDeviceToHost);
    ok = 1;
    for (ii = tlo[0]; ii<=thi[0]; ii++) {
      i = ii-tlo[0];
      for (jj=tlo[1]; jj<=thi[1]; jj++) {
        j = jj-tlo[1];
        idx = i*tld+j;
        if (tbuf[idx] != ii*DIMSIZE+jj) {
          if (ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %d\n",rank,ii,jj,ii*DIMSIZE+jj,
              idx,tbuf[idx]);
          ok = 0;
        }
      }
    }
    if (!ok) {
      printf("Mismatch found for put on process %d after Put\n",rank);
    } else {
      if (rank==0) printf("Put is okay\n");
    }
    free(tbuf);
  }
  GA_Sync();
#endif

  /* zero out local buffer */
  for (i=0; i<nsize; i++) buf[i] = 0;

  /* copy data from global array to local buffer */
  if (rank == 0) printf("Calling GA get\n",rank);
  NGA_Get(g_a, lo, hi, buf, &ld);
  GA_Sync();

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
    printf("Mismatch found for get on process %d after Get\n",rank);
  } else {
    if (rank == 0) printf("Get is okay\n");
  }

  /* reset values in buf */
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      buf[idx] = ii*DIMSIZE+jj;
    }
  }

  /* accumulate data to global array */
  if (rank == 0) printf("Calling GA acc\n",rank);
  one = 1;
  NGA_Acc(g_a, lo, hi, buf, &ld, &one);
  GA_Sync();
#if 0
  if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
    int *tbuf;
    int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
    tbuf = (int*)malloc(tnelem*sizeof(int));
    NGA_Access(g_a,tlo,thi,&ptr,&tld);
    for (i=0; i<tnelem; i++) tbuf[i] = 0;
    cudaMemcpy(tbuf, ptr, tnelem*sizeof(int), cudaMemcpyDeviceToHost);
    if (tnelem > 0) {
      printf("p[%d] acc buffer:",rank);
      for (i=0; i<tnelem; i++) printf(" %d",tbuf[i]);
      printf("\n"); 
    }
    free(tbuf);
  }
#endif
  /* reset values in buf */
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      buf[idx] = 0;
    }
  }

  if (rank == 0) printf("Calling GA get\n",rank);
  NGA_Get(g_a, lo, hi, buf, &ld);
  GA_Sync();
  ok = 1;
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      if (buf[idx] != 2*(ii*DIMSIZE+jj)) {
        if (ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %d\n",rank,ii,jj,
            2*(ii*DIMSIZE+jj),idx,buf[idx]);
        ok = 0;
      }
    }
  }
  if (!ok) {
    printf("Mismatch found on process %d after Acc\n",rank);
  } else {
    if (rank == 0) printf("Acc is okay\n");
  }

  free(buf);
  GA_Destroy(g_a);
  if (rank == 0) printf("Completed GA destroy\n");

#if 1
  if (rank == 0) printf("  Testing double precision arrays\n");
  /* create a global array and initialize it to zero */
  g_a = NGA_Create_handle();
  if (rank == 0) printf("Created GA handle\n");
  NGA_Set_data(g_a, ndim, dims, C_DBL);
  if (rank == 0) printf("Set GA data\n");
  NGA_Set_device(g_a, 1);
  if (rank == 0) printf("Set GA device\n");
  NGA_Allocate(g_a);
  if (rank == 0) printf("Completed GA allocate\n");
  GA_Zero(g_a);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  rbuf = (double*)malloc(nsize*sizeof(double));
  ld = (hi[1]-lo[1]+1);
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      rbuf[idx] = (double)(ii*DIMSIZE+jj);
    }
  }

  /* copy data to global array */
  NGA_Put(g_a, lo, hi, rbuf, &ld);
  GA_Sync();
  if (rank == 0) printf("Completed GA put\n");
  NGA_Distribution(g_a,rank,tlo,thi);

  /* zero out local buffer */
  for (i=0; i<nsize; i++) rbuf[i] = 0.0;

  /* copy data from global array to local buffer */
  if (rank == 0) printf("Calling GA get\n",rank);
  NGA_Get(g_a, lo, hi, rbuf, &ld);
  GA_Sync();

  ok = 1;
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      if (rbuf[idx] != (double)(ii*DIMSIZE+jj)) {
        if (ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %f\n",rank,ii,jj,ii*DIMSIZE+jj,
            idx,rbuf[idx]);
        ok = 0;
      }
    }
  }
  if (!ok) {
    printf("Mismatch found on process %d after Get\n",rank);
  } else {
    if (rank == 0) printf("Get is okay\n");
  }

  /* reset values in buf */
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      rbuf[idx] = (double)(ii*DIMSIZE+jj);
    }
  }

  /* accumulate data to global array */
  if (rank == 0) printf("Calling GA acc\n",rank);
  one_r = 1.0;
  NGA_Acc(g_a, lo, hi, rbuf, &ld, &one_r);
  GA_Sync();
  if (rank == 0) printf("Calling GA get\n",rank);
  NGA_Get(g_a, lo, hi, rbuf, &ld);
  ok = 1;
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      if (rbuf[idx] != (double)(2*(ii*DIMSIZE+jj))) {
        if (ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %f\n",rank,ii,jj,
            2*(ii*DIMSIZE+jj),idx,rbuf[idx]);
        ok = 0;
      }
    }
  }
  if (!ok) {
    printf("Mismatch found on process %d after Acc\n",rank);
  } else {
    if (rank == 0) printf("Acc is okay\n");
  }

  free(rbuf);
  GA_Destroy(g_a);

#if 1
  if (rank == 0) printf("  Testing read-increment function\n");
  /* create a global array and initialize it to zero */
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, one, &one, C_INT);
  NGA_Set_device(g_a, 1);
  NGA_Allocate(g_a);
  GA_Zero(g_a);
  if (rank == 0) printf("Created and initialized global array with 1 element\n");
  icnt = 0;
  while (icnt<MAXCOUNT) {
    icnt = NGA_Read_inc(g_a,&zero,(long)one);
    if (icnt%1000 == 0) printf("p[%d] current value of counter: %d\n",rank,icnt);
  }
  GA_Sync();
  if (rank == 0) {
    NGA_Get(g_a,&zero,&zero,&i,&zero);
    if (i != MAXCOUNT+nprocs) {
      printf ("Mismatch found for read-increment expected: %d actual: %d\n",
          MAXCOUNT+nprocs,i);
    } else {
      printf("Read-increment is OK\n");
    }
  }

  
  GA_Destroy(g_a);
#endif
  if (rank == 0) printf("Completed GA destroy\n");
#endif
  GA_Terminate();
  if (rank == 0) printf("Completed GA terminate\n");
  MPI_Finalize();
}
