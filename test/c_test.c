#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "mpi.h"
#include "ga.h"
#include "macdecls.h"
#include <cuda_runtime.h>

/*
#define BLOCK1 1024*1024
#define BLOCK1 65536
*/
#define BLOCK1 65530
#define DIMSIZE 1024
#define MAXCOUNT 10000
#define MAX_FACTOR 256
#define NLOOP 10

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

int nprocs, rank;
int pdx, pdy;

double tput, tget, tacc, tinc;
int get_cnt,put_cnt,acc_cnt;
double put_bw, get_bw, acc_bw;

void test_int_array(int on_device)
{
  int g_a;
  int i, j, ii, jj, n, idx;
  int ipx, ipy;
  int isx, isy;
  int xinc, yinc;
  int ndim, nsize;
  int dims[2], lo[2], hi[2];
  int tld, tlo[2], thi[2];
  int *buf;
  int *tbuf;
  int nelem;
  int ld;
  int g_ok, p_ok, a_ok;
  int *ptr;
  int one;
  double tbeg;

  tput = 0.0;
  tget = 0.0;
  tacc = 0.0;
  put_cnt = 0;
  get_cnt = 0;
  acc_cnt = 0;
  p_ok = 1;
  g_ok = 1;
  a_ok = 1;

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
  lo[0] = isx*xinc;
  lo[1] = isy*yinc;
  if (isx<pdx-1) {
    hi[0] = (isx+1)*xinc-1;
  } else {
    hi[0] = DIMSIZE-1;
  }
  if (isy<pdy-1) {
    hi[1] = (isy+1)*yinc-1;
  } else {
    hi[1] = DIMSIZE-1;
  }
  nelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);

  /* create a global array and initialize it to zero */
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_INT);
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  buf = (int*)malloc(nsize*sizeof(int));

  for (n=0; n<NLOOP; n++) {
    GA_Zero(g_a);
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
    tbeg = GA_Wtime();
    NGA_Put(g_a, lo, hi, buf, &ld);
    tput += (GA_Wtime()-tbeg);
    put_cnt += nsize;
    GA_Sync();
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
      p_ok = 1;
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        for (jj=tlo[1]; jj<=thi[1]; jj++) {
          j = jj-tlo[1];
          idx = i*tld+j;
          if (tbuf[idx] != ii*DIMSIZE+jj) {
            if (ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %d\n",rank,ii,jj,ii*DIMSIZE+jj,
                idx,tbuf[idx]);
            p_ok = 0;
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
    tbeg = GA_Wtime();
    NGA_Get(g_a, lo, hi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    GA_Sync();

    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        if (buf[idx] != ii*DIMSIZE+jj) {
          if (g_ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %d\n",rank,ii,jj,ii*DIMSIZE+jj,
              idx,buf[idx]);
          g_ok = 0;
        }
      }
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
    one = 1;
    tbeg = GA_Wtime();
    NGA_Acc(g_a, lo, hi, buf, &ld, &one);
    tacc += (GA_Wtime()-tbeg);
    acc_cnt += nsize;
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

    tbeg = GA_Wtime();
    NGA_Get(g_a, lo, hi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    GA_Sync();
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        if (buf[idx] != 2*(ii*DIMSIZE+jj)) {
          if (a_ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %d\n",rank,ii,jj,
              2*(ii*DIMSIZE+jj),idx,buf[idx]);
          a_ok = 0;
        }
      }
    }
  }

  free(buf);
  GA_Destroy(g_a);

  if (!g_ok) {
    printf("Mismatch found for get on process %d after Get\n",rank);
  } else {
    if (rank == 0) printf("Get is okay\n");
  }
  if (!a_ok) {
    printf("Mismatch found on process %d after Acc\n",rank);
  } else {
    if (rank == 0) printf("Acc is okay\n");
  }

  GA_Igop(&put_cnt, 1, "+");
  GA_Igop(&get_cnt, 1, "+");
  GA_Igop(&acc_cnt, 1, "+");
  GA_Dgop(&tput, 1, "+");
  GA_Dgop(&tget, 1, "+");
  GA_Dgop(&tacc, 1, "+");
  put_bw = (double)(put_cnt*sizeof(int))/tput;
  get_bw = (double)(get_cnt*sizeof(int))/tget;
  acc_bw = (double)(acc_cnt*sizeof(int))/tacc;
}

void print_bw()
{
  int iB = 1<<20;
  double rB = 1.0/((double)iB);
  /*      12345678901234567890123456789012345678901234567890123456789012345 */
  if (rank != 0) return;

  printf("///////////////////////////////////////////////////////////////\n");
  printf("//      Put (MB/sec) //     Get (MB/sec) //     Acc (MB/sec) //\n");
  printf("//  %15.4e  // %15.4e  // %15.4e  //\n",put_bw*rB,get_bw*rB,acc_bw*rB);
  printf("///////////////////////////////////////////////////////////////\n");
}

void test_dbl_array(int on_device)
{
  int g_a;
  int i, j, ii, jj, idx;
  int ipx, ipy;
  int isx, isy;
  int xinc, yinc;
  int ndim, nsize;
  int dims[2], lo[2], hi[2];
  int tld, tlo[2], thi[2];
  double *buf;
  double *tbuf;
  int nelem;
  int ld;
  int ok;
  double *ptr;
  double one;

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
  lo[0] = isx*xinc;
  lo[1] = isy*yinc;
  if (isx<pdx-1) {
    hi[0] = (isx+1)*xinc-1;
  } else {
    hi[0] = DIMSIZE-1;
  }
  if (isy<pdy-1) {
    hi[1] = (isy+1)*yinc-1;
  } else {
    hi[1] = DIMSIZE-1;
  }
  nelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);

  /* create a global array and initialize it to zero */
  g_a = NGA_Create_handle();
  if (rank == 0) printf("Created GA handle\n");
  NGA_Set_data(g_a, ndim, dims, C_DBL);
  if (rank == 0) printf("Set GA data\n");
  NGA_Set_device(g_a, on_device);
  if (rank == 0) printf("Set GA device\n");
  NGA_Allocate(g_a);
  if (rank == 0) printf("Completed GA allocate\n");
  GA_Zero(g_a);
  if (rank == 0) printf("Completed GA zero\n");

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  buf = (double*)malloc(nsize*sizeof(double));
  ld = (hi[1]-lo[1]+1);
  for (ii = lo[0]; ii<=hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<=hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      buf[idx] = (double)(ii*DIMSIZE+jj);
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
    tbuf = (int*)malloc(tnelem*sizeof(double));
    NGA_Access(g_a,tlo,thi,&ptr,&tld);
    for (i=0; i<tnelem; i++) tbuf[i] = 0;
    cudaMemcpy(tbuf, ptr, tnelem*sizeof(double), cudaMemcpyDeviceToHost);
    ok = 1;
    for (ii = tlo[0]; ii<=thi[0]; ii++) {
      i = ii-tlo[0];
      for (jj=tlo[1]; jj<=thi[1]; jj++) {
        j = jj-tlo[1];
        idx = i*tld+j;
        if (tbuf[idx] != ii*DIMSIZE+jj) {
          if (ok) printf("p[%d] (%d,%d) expected: %f actual[%d]: %f\n",rank,ii,jj,
              (double)(ii*DIMSIZE+jj),idx,tbuf[idx]);
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
  for (i=0; i<nsize; i++) buf[i] = 0.0;

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
      if (buf[idx] != (double)(ii*DIMSIZE+jj)) {
        if (ok) printf("p[%d] (%d,%d) expected: %f actual[%d]: %f\n",rank,ii,jj,
            (double)(ii*DIMSIZE+jj),idx,buf[idx]);
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
      buf[idx] = (double)(ii*DIMSIZE+jj);
    }
  }

  /* accumulate data to global array */
  if (rank == 0) printf("Calling GA acc\n",rank);
  one = 1.0;
  NGA_Acc(g_a, lo, hi, buf, &ld, &one);
  GA_Sync();
#if 0
  if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
    int *tbuf;
    int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
    tbuf = (int*)malloc(tnelem*sizeof(double));
    NGA_Access(g_a,tlo,thi,&ptr,&tld);
    for (i=0; i<tnelem; i++) tbuf[i] = 0;
    cudaMemcpy(tbuf, ptr, tnelem*sizeof(double), cudaMemcpyDeviceToHost);
    if (tnelem > 0) {
      printf("p[%d] acc buffer:",rank);
      for (i=0; i<tnelem; i++) printf(" %f",tbuf[i]);
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
      buf[idx] = 0.0;
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
      if (buf[idx] != (double)(2*(ii*DIMSIZE+jj))) {
        if (ok) printf("p[%d] (%d,%d) expected: %d actual[%d]: %d\n",rank,ii,jj,
            (double)(2*(ii*DIMSIZE+jj)),idx,buf[idx]);
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

}

void test_read_inc(int on_device)
{
  int g_a;
  int one;
  int icnt, zero, i;
  /* create a global array and initialize it to zero */
  zero = 0;
  one = 1;
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, one, &one, C_INT);
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);
  GA_Zero(g_a);
  if (rank == 0) printf("Created and initialized global array with 1 element\n");
  icnt = 0;
  while (icnt<MAXCOUNT) {
    icnt = NGA_Read_inc(g_a,&zero,(long)one);
    if (icnt%1000 == 0) printf("  current value of counter: %d read on process %d\n",icnt,rank);
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
}

void test_int_1d_array(int on_device)
{
  int g_a;
  int i, ii, ld;
  int tlo[1], thi[1];
  int nelem;
  int nsize;
  int *buf;
  int one;
  int ok;

  one = 1;
  nelem = BLOCK1*nprocs;
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, one, &nelem, C_INT);
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);
  GA_Zero(g_a);
  if (rank == 0) printf("Created and initialized 1D global array of size %d\n",nelem);

  /* allocate a local buffer and initialize it with values*/
  i = (rank+1)%nprocs;
  tlo[0] = i*(nelem/nprocs);
  if (i<nprocs-1) {
    thi[0] = (i+1)*(nelem/nprocs)-1;
  } else {
    thi[0] = nelem-1;
  }
  nsize = thi[0]-tlo[0]+1;
  buf = (int*)malloc(nsize*sizeof(int));
  ld = 1;
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    buf[i] = ii;
  }

  /* copy data to global array */
  NGA_Put(g_a, tlo, thi, buf, &ld);
  if (rank == 0) printf("Completed GA put\n");
  GA_Sync();
  if (rank == 0) printf("Completed GA sync\n");

  /* zero out local buffer */
  for (i=0; i<nsize; i++) buf[i] = 0;

  /* copy data from global array to local buffer */
  if (rank == 0) printf("Calling GA get\n");
  NGA_Get(g_a, tlo, thi, buf, &ld);
  GA_Sync();

  ok = 1;
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    if (buf[i] != ii) {
      if (ok) printf("p[%d] index: %d expected: %f actual: %f\n",rank,i,ii,buf[i]);
      ok = 0;
    }
  }
  if (!ok) {
    printf("Mismatch found for get on process %d after Get\n",rank);
  } else {
    if (rank == 0) printf("Get is okay\n");
  }

  /* reset values in buf */
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    buf[i] = ii;
  }

  /* accumulate data to global array */
  if (rank == 0) printf("Calling GA acc\n");
  NGA_Acc(g_a, tlo, thi, buf, &ld, &one);
  GA_Sync();
#if 0
  NGA_Distribution(g_a,rank,lo,hi);
  if (lo[0]<=hi[0]) {
    int *tbuf;
    int *ptr;
    int tnelem = hi[0]-lo[0]+1;
    tbuf = (int*)malloc(tnelem*sizeof(int));
    NGA_Access(g_a,lo,hi,&ptr,&tld);
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
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    buf[i] = 0;
  }

  if (rank == 0) printf("Calling GA get\n");
  NGA_Get(g_a, tlo, thi, buf, &ld);
  GA_Sync();
  ok = 1;
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    if (buf[i] != 2*ii) {
      if (ok) printf("p[%d] index: %d expected: %d actual: %d\n",rank,i,2*ii,buf[i]);
      ok = 0;
    }
  }
  if (!ok) {
    printf("Mismatch found on process %d after Acc\n",rank);
  } else {
    if (rank == 0) printf("Acc is okay\n");
  }
  free(buf);

  GA_Destroy(g_a);
}

void test_dbl_1d_array(int on_device)
{
  int g_a;
  int i, ii, ld;
  int tlo[1], thi[1];
  int nelem;
  int nsize;
  double *buf;
  double one;
  int ok;

  one = 1.0;
  nelem = BLOCK1*nprocs;
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, one, &nelem, C_DBL);
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);
  GA_Zero(g_a);
  if (rank == 0) printf("Created and initialized 1D global array of size %d\n",nelem);

  /* allocate a local buffer and initialize it with values*/
  i = (rank+1)%nprocs;
  tlo[0] = i*(nelem/nprocs);
  if (i<nprocs-1) {
    thi[0] = (i+1)*(nelem/nprocs)-1;
  } else {
    thi[0] = nelem-1;
  }
  nsize = thi[0]-tlo[0]+1;
  buf = (double*)malloc(nsize*sizeof(double));
  ld = 1;
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    buf[i] = (double)ii;
  }

  /* copy data to global array */
  NGA_Put(g_a, tlo, thi, buf, &ld);
  if (rank == 0) printf("Completed GA put\n");
  GA_Sync();
  if (rank == 0) printf("Completed GA sync\n");

  /* zero out local buffer */
  for (i=0; i<nsize; i++) buf[i] = 0.0;

  /* copy data from global array to local buffer */
  if (rank == 0) printf("Calling GA get\n");
  NGA_Get(g_a, tlo, thi, buf, &ld);
  GA_Sync();

  ok = 1;
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    if (buf[i] != (double)ii) {
      if (ok) printf("p[%d] index: %d expected: %f actual: %f\n",rank,i,(double)ii,buf[i]);
      ok = 0;
    }
  }
  if (!ok) {
    printf("Mismatch found for get on process %d after Get\n",rank);
  } else {
    if (rank == 0) printf("Get is okay\n");
  }

  /* reset values in buf */
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    buf[i] = (double)ii;
  }

  /* accumulate data to global array */
  if (rank == 0) printf("Calling GA acc\n");
  NGA_Acc(g_a, tlo, thi, buf, &ld, &one);
  GA_Sync();
#if 0
  NGA_Distribution(g_a,rank,lo,hi);
  if (lo[0]<=hi[0]) {
    double *tbuf;
    double *ptr;
    int tnelem = hi[0]-lo[0]+1;
    tbuf = (double*)malloc(tnelem*sizeof(double));
    NGA_Access(g_a,lo,hi,&ptr,&tld);
    for (i=0; i<tnelem; i++) tbuf[i] = 0;
    cudaMemcpy(tbuf, ptr, tnelem*sizeof(double), cudaMemcpyDeviceToHost);
    if (tnelem > 0) {
      printf("p[%d] acc buffer:",rank);
      for (i=0; i<tnelem; i++) printf(" %f",tbuf[i]);
      printf("\n"); 
    }
    free(tbuf);
  }
#endif
  /* reset values in buf */
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    buf[i] = 0.0;
  }

  if (rank == 0) printf("Calling GA get\n");
  NGA_Get(g_a, tlo, thi, buf, &ld);
  GA_Sync();
  ok = 1;
  for (ii = tlo[0]; ii<=thi[0]; ii++) {
    i = ii-tlo[0];
    if (buf[i] != (double)(2*ii)) {
      if (ok) printf("p[%d] index: %d expected: %f actual: %f\n",rank,i,(double)(2*ii),buf[i]);
      ok = 0;
    }
  }
  if (!ok) {
    printf("Mismatch found on process %d after Acc\n",rank);
  } else {
    if (rank == 0) printf("Acc is okay\n");
  }
  free(buf);

  GA_Destroy(g_a);
}

int main(int argc, char **argv) {

  int g_a;
  double *rbuf;
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
  if (rank == 0) printf("  Testing integer array on device\n");
  test_int_array(1);
  print_bw();

  if (rank == 0) printf("  Testing integer array on host\n");
  test_int_array(0);
  print_bw();

  if (rank == 0) printf("  Testing double precision array on device\n");
  test_dbl_array(1);

  if (rank == 0) printf("  Testing double precision array on host\n");
  test_dbl_array(1);

  if (rank == 0) printf("  Testing read-increment function on device\n");
  test_read_inc(1);
  
  if (rank == 0) printf("  Testing read-increment function on host\n");
  test_read_inc(0);
  
  if (rank == 0) printf("  Testing contiguous one-sided operations for integers on device\n");
  test_int_1d_array(1);

  if (rank == 0) printf("  Testing contiguous one-sided operations for integer on host\n");
  test_int_1d_array(0);

  if (rank == 0) printf("  Testing contiguous one-sided operations for doubles on device\n");
  test_dbl_1d_array(1);

  if (rank == 0) printf("  Testing contiguous one-sided operations for doubles on host\n");
  test_dbl_1d_array(0);

  GA_Terminate();
  if (rank == 0) printf("Completed GA terminate\n");
  MPI_Finalize();
}
