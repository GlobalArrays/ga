#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include "mpi.h"
#include "ga.h"
#include "macdecls.h"

#if defined(ENABLE_HIP)
#include <hip/hip_runtime.h>
#elif defined(ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

#define DIMSIZE 2048
#define MAXCOUNT 10000
#define MAX_FACTOR 256
#define NLOOP 10

void set_device(int* devid) {
  #if defined(ENABLE_CUDA)
  cudaSetDevice(*devid);
  #elif defined(ENABLE_HIP)
  hipSetDevice(*devid);
  #endif    
}

void device_malloc(void **buf, size_t bsize){
  #if defined(ENABLE_CUDA)
  cudaMalloc(buf, bsize);
  #elif defined(ENABLE_HIP)
  hipMalloc(buf, bsize);
  #endif    
}

void memcpyH2D(void* dbuf, void* sbuf, size_t bsize){
  #if defined(ENABLE_CUDA)
  cudaMemcpy(dbuf, sbuf, bsize, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  #elif defined(ENABLE_HIP)
  hipMemcpy(dbuf, sbuf, bsize, hipMemcpyHostToDevice);
  hipDeviceSynchronize();
  #endif    
}

void memcpyD2H(void* dbuf, void* sbuf, size_t bsize){
  #if defined(ENABLE_CUDA)
  cudaMemcpy(dbuf, sbuf, bsize, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  #elif defined(ENABLE_HIP)
  hipMemcpy(dbuf, sbuf, bsize, hipMemcpyDeviceToHost);
  hipDeviceSynchronize();
  #endif    
}

void device_free(void *buf) {
  #if defined(ENABLE_CUDA)
  cudaFree(buf);
  #elif defined(ENABLE_HIP)
  hipFree(buf);
  #endif   
}


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

//int nprocs, rank, wrank;
int nprocs, rank;
int pdx, pdy;
int wrank;

int *list;
int *devIDs;
int ndev;
int my_dev;

double tput, tget, tcpy, tinc;
int get_cnt,put_cnt,cpy_cnt;
double put_bw, get_bw, cpy_bw;
double t_put, t_get, t_cpy, t_sync, t_chk, t_tot;
double t_create, t_free;

void test_int_array(int a_array_on_device, int b_array_on_device)
{
  int g_a, g_b;
  int i, j, ii, jj, n, idx;
  int ipx, ipy;
  int xinc, yinc;
  int ndim, nsize;
  int dims[2], lo[2], hi[2];
  int tld, tlo[2], thi[2];
  int *buf;
  int *tbuf;
  int nelem;
  int ld;
  int *ptr;
  int one;
  double tbeg;
  double zero = 0.0;
  int izero = 0;
  int ok;

  tput = 0.0;
  tget = 0.0;
  tcpy = 0.0;
  put_cnt = 0;
  get_cnt = 0;
  cpy_cnt = 0;
  ok = 1;

  t_tot = 0.0;

  factor(nprocs,&pdx,&pdy);
  ndim = 2;
  dims[0] = DIMSIZE;
  dims[1] = DIMSIZE;
  xinc = DIMSIZE/pdx;
  yinc = DIMSIZE/pdy;
  ipx = rank%pdx;
  ipy = (rank-ipx)/pdx;
  lo[0] = ipx*xinc;
  if (ipx < pdx-1) {
    hi[0] = (ipx+1)*xinc-1;
  } else {
    hi[0] = DIMSIZE-1;
  }
  lo[1] = ipy*yinc;
  if (ipy < pdy-1) {
    hi[1] = (ipy+1)*yinc-1;
  } else {
    hi[1] = DIMSIZE-1;
  }
  ld = (hi[1]-lo[1]+1);

  /* create a global array and initialize it to zero */
  tbeg = GA_Wtime();
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_INT);
  NGA_Set_device(g_a, a_array_on_device);
  NGA_Allocate(g_a);
  g_b = NGA_Create_handle();
  NGA_Set_data(g_b, ndim, dims, C_INT);
  NGA_Set_device(g_b, b_array_on_device);
  NGA_Allocate(g_b);
  t_create += (GA_Wtime()-tbeg);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  buf = (int*)malloc(nsize*sizeof(int));

  for (n=0; n<NLOOP; n++) {
    tbeg = GA_Wtime();
    GA_Zero(g_a);
    GA_Fill(g_a,&izero);
    ld = (hi[1]-lo[1]+1);
    /* Fill local array with values */
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        buf[idx] = ii*DIMSIZE+jj;
      }
    }
    t_chk += (GA_Wtime()-tbeg);
    /* copy data to global array */
    tbeg = GA_Wtime();
    NGA_Put(g_a, lo, hi, buf, &ld);
    tput += (GA_Wtime()-tbeg);
    t_put += (GA_Wtime()-tbeg);
    put_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
#if 1
    /* Verify that original distribution is correct */
    if (n == 0) {
      NGA_Distribution(g_a,rank,tlo,thi);
      if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
        int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
        if (a_array_on_device) {
          tbuf = (int*)malloc(tnelem*sizeof(int));
          NGA_Access(g_a,tlo,thi,&ptr,&tld);
          for (i=0; i<tnelem; i++) tbuf[i] = 0;
          memcpyD2H(tbuf, ptr, tnelem*sizeof(int));
        } else {
          NGA_Access(g_a,tlo,thi,&tbuf,&tld);
        }
        ok = 1;
        for (ii = tlo[0]; ii<=thi[0]; ii++) {
          i = ii-tlo[0];
          for (jj=tlo[1]; jj<=thi[1]; jj++) {
            j = jj-tlo[1];
            idx = i*tld+j;
            if (tbuf[idx] != ii*DIMSIZE+jj) {
              if (ok) printf("p[%d] (%d,%d) (put) expected: %d actual[%d]: %d\n",
                  rank,ii,jj,ii*DIMSIZE+jj,idx,tbuf[idx]);
              ok = 0;
            }
          }
        }
        if (!ok) {
          printf("Mismatch found in initial data on process %d\n",rank);
        } else if (n==0 && rank==0 && ok) {
          printf("Initial data is okay\n");
        }
        if (a_array_on_device) {
          free(tbuf);
        }
        NGA_Release(g_a,tlo,thi);
      }
      tbeg = GA_Wtime();
      GA_Sync();
      t_sync += (GA_Wtime()-tbeg);
    }
#endif


    /* copy data from global array a to global array b */
    tbeg = GA_Wtime();
    GA_Copy(g_a, g_b);
    tcpy += (GA_Wtime()-tbeg);
    t_cpy += (GA_Wtime()-tbeg);
    cpy_cnt += nsize;

    /* check values in g_b */
    tbeg = GA_Wtime();
    NGA_Get(g_b, lo, hi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    t_get += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    ok = 1;
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        if (buf[idx] != ii*DIMSIZE+jj) {
          if (ok) printf("p[%d] (%d,%d) (get) expected: %d"
              " actual[%d]: %d\n",rank,ii,jj,
              ii*DIMSIZE+jj,idx,buf[idx]);
          ok = 0;
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);
  }

  free(buf);

  tbeg = GA_Wtime();
  GA_Destroy(g_a);
  GA_Destroy(g_b);
  t_free += (GA_Wtime()-tbeg);

  if (!ok) {
    printf("Mismatch found for copy on process %d after copy\n",rank);
  } else {
    if (rank == 0) printf("Copy is okay\n");
  }

  GA_Igop(&put_cnt, 1, "+");
  GA_Igop(&get_cnt, 1, "+");
  GA_Igop(&cpy_cnt, 1, "+");
  GA_Dgop(&tput, 1, "+");
  GA_Dgop(&tget, 1, "+");
  GA_Dgop(&tcpy, 1, "+");
  put_bw = (double)(put_cnt*sizeof(int))/tput;
  get_bw = (double)(get_cnt*sizeof(int))/tget;
  cpy_bw = (double)(cpy_cnt*sizeof(int))/tcpy;
}

void print_bw()
{
  int iB = 1<<20;
  double rB = 1.0/((double)iB);
  /*      12345678901234567890123456789012345678901234567890123456789012345 */
  if (rank != 0) return;

  printf("///////////////////////////////////////////////////////////////\n");
  printf("//      Put (MB/sec) //     Get (MB/sec) //     Cpy (MB/sec) //\n");
  printf("//  %15.4e  // %15.4e  // %15.4e  //\n",put_bw*rB,get_bw*rB,cpy_bw*rB);
  printf("///////////////////////////////////////////////////////////////\n");
  printf("\n\n");
}

void test_dbl_array(int a_array_on_device, int b_array_on_device)
{
  int g_a, g_b;
  int i, j, ii, jj, n, idx;
  int ipx, ipy;
  int xinc, yinc;
  int ndim, nsize;
  int dims[2], lo[2], hi[2];
  int tld, tlo[2], thi[2];
  double *buf;
  double *tbuf;
  int nelem;
  int ld;
  int *ptr;
  int one;
  double tbeg;
  double zero = 0.0;
  int ok;

  tput = 0.0;
  tget = 0.0;
  tcpy = 0.0;
  put_cnt = 0;
  get_cnt = 0;
  cpy_cnt = 0;
  ok = 1;

  t_tot = 0.0;

  factor(nprocs,&pdx,&pdy);
  ndim = 2;
  dims[0] = DIMSIZE;
  dims[1] = DIMSIZE;
  xinc = DIMSIZE/pdx;
  yinc = DIMSIZE/pdy;
  ipx = rank%pdx;
  ipy = (rank-ipx)/pdx;
  lo[0] = ipx*xinc;
  if (ipx < pdx-1) {
    hi[0] = (ipx+1)*xinc-1;
  } else {
    hi[0] = DIMSIZE-1;
  }
  lo[1] = ipy*yinc;
  if (ipy < pdy-1) {
    hi[1] = (ipy+1)*yinc-1;
  } else {
    hi[1] = DIMSIZE-1;
  }
  ld = (hi[1]-lo[1]+1);

  /* create a global array and initialize it to zero */
  tbeg = GA_Wtime();
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_DBL);
  NGA_Set_device(g_a, a_array_on_device);
  NGA_Allocate(g_a);
  g_b = NGA_Create_handle();
  NGA_Set_data(g_b, ndim, dims, C_DBL);
  NGA_Set_device(g_b, b_array_on_device);
  NGA_Allocate(g_b);
  t_create += (GA_Wtime()-tbeg);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  buf = (double*)malloc(nsize*sizeof(double));

  for (n=0; n<NLOOP; n++) {
    tbeg = GA_Wtime();
    GA_Zero(g_a);
    GA_Fill(g_a,&zero);
    ld = (hi[1]-lo[1]+1);
    /* Fill local array with values */
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        buf[idx] = (double)(ii*DIMSIZE+jj);
      }
    }
    t_chk += (GA_Wtime()-tbeg);
    /* copy data to global array */
    tbeg = GA_Wtime();
    NGA_Put(g_a, lo, hi, buf, &ld);
    tput += (GA_Wtime()-tbeg);
    t_put += (GA_Wtime()-tbeg);
    put_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
#if 1
    /* Verify that original distribution is correct */
    if (n == 0) {
      NGA_Distribution(g_a,rank,tlo,thi);
      if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
        int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
        if (a_array_on_device) {
          tbuf = (double*)malloc(tnelem*sizeof(double));
          NGA_Access(g_a,tlo,thi,&ptr,&tld);
          for (i=0; i<tnelem; i++) tbuf[i] = 0;
          memcpyD2H(tbuf, ptr, tnelem*sizeof(double));
        } else {
          NGA_Access(g_a,tlo,thi,&tbuf,&tld);
        }
        ok = 1;
        for (ii = tlo[0]; ii<=thi[0]; ii++) {
          i = ii-tlo[0];
          for (jj=tlo[1]; jj<=thi[1]; jj++) {
            j = jj-tlo[1];
            idx = i*tld+j;
            if (tbuf[idx] != (double)(ii*DIMSIZE+jj)) {
              if (ok) printf("p[%d] (%d,%d) (put) expected: %f actual[%f]: %d\n",
                  rank,ii,jj,(double)(ii*DIMSIZE+jj),idx,tbuf[idx]);
              ok = 0;
            }
          }
        }
        if (!ok) {
          printf("Mismatch found in initial data on process %d\n",rank);
        } else if (n==0 && rank==0 && ok) {
          printf("Initial data is okay\n");
        }
        NGA_Release(g_a,tlo,thi);
        if (a_array_on_device) {
          free(tbuf);
        }
      }
      tbeg = GA_Wtime();
      GA_Sync();
      t_sync += (GA_Wtime()-tbeg);
    }
#endif


    /* copy data from global array a to global array b */
    tbeg = GA_Wtime();
    GA_Copy(g_a, g_b);
    tcpy += (GA_Wtime()-tbeg);
    t_cpy += (GA_Wtime()-tbeg);
    cpy_cnt += nsize;

    /* check values in g_b */
    tbeg = GA_Wtime();
    NGA_Get(g_b, lo, hi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    t_get += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        if (buf[idx] != ii*DIMSIZE+jj) {
          if (ok) printf("p[%d] (%d,%d) (get) expected: %f"
              " actual[%d]: %f\n",rank,ii,jj,
              ii*DIMSIZE+jj,idx,buf[idx]);
          ok = 0;
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);
  }

  free(buf);

  tbeg = GA_Wtime();
  GA_Destroy(g_a);
  GA_Destroy(g_b);
  t_free += (GA_Wtime()-tbeg);

  if (!ok) {
    printf("Mismatch found for copy on process %d after copy\n",rank);
  } else {
    if (rank == 0) printf("Copy is okay\n");
  }

  GA_Igop(&put_cnt, 1, "+");
  GA_Igop(&get_cnt, 1, "+");
  GA_Igop(&cpy_cnt, 1, "+");
  GA_Dgop(&tput, 1, "+");
  GA_Dgop(&tget, 1, "+");
  GA_Dgop(&tcpy, 1, "+");
  put_bw = (double)(put_cnt*sizeof(int))/tput;
  get_bw = (double)(get_cnt*sizeof(int))/tget;
  cpy_bw = (double)(cpy_cnt*sizeof(int))/tcpy;
}

void test_int_patch(int a_array_on_device, int b_array_on_device)
{
  int g_a, g_b;
  int i, j, ii, jj, n, idx;
  int ipx, ipy;
  int xinc, yinc;
  int ndim, nsize;
  int dims[2], lo[2], hi[2];
  int tld, tlo[2], thi[2];
  int plo[2], phi[2];
  int *buf;
  int *tbuf;
  int nelem;
  int ld;
  int *ptr;
  int one;
  double tbeg;
  double zero = 0.0;
  int izero = 0;
  int ok;

  tput = 0.0;
  tget = 0.0;
  tcpy = 0.0;
  put_cnt = 0;
  get_cnt = 0;
  cpy_cnt = 0;
  ok = 1;

  t_tot = 0.0;

  factor(nprocs,&pdx,&pdy);
  ndim = 2;
  dims[0] = DIMSIZE;
  dims[1] = DIMSIZE;
  xinc = DIMSIZE/pdx;
  yinc = DIMSIZE/pdy;
  ipx = rank%pdx;
  ipy = (rank-ipx)/pdx;
  lo[0] = ipx*xinc;
  if (ipx < pdx-1) {
    hi[0] = (ipx+1)*xinc-1;
  } else {
    hi[0] = DIMSIZE-1;
  }
  lo[1] = ipy*yinc;
  if (ipy < pdy-1) {
    hi[1] = (ipy+1)*yinc-1;
  } else {
    hi[1] = DIMSIZE-1;
  }
  ld = (hi[1]-lo[1]+1);

  /* calculate bounding indices for patch */
  plo[0] = DIMSIZE/4 + 1;
  phi[0] = (DIMSIZE*3)/4 - 1;
  plo[1] = DIMSIZE/4 + 1;
  phi[1] = (DIMSIZE*3)/4 - 1;

  /* create a global array and initialize it to zero */
  tbeg = GA_Wtime();
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_INT);
  NGA_Set_device(g_a, a_array_on_device);
  NGA_Allocate(g_a);
  g_b = NGA_Create_handle();
  NGA_Set_data(g_b, ndim, dims, C_INT);
  NGA_Set_device(g_b, b_array_on_device);
  NGA_Allocate(g_b);
  t_create += (GA_Wtime()-tbeg);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  buf = (int*)malloc(nsize*sizeof(int));

  for (n=0; n<NLOOP; n++) {
    tbeg = GA_Wtime();
    GA_Zero(g_a);
    GA_Fill(g_a,&izero);
    ld = (hi[1]-lo[1]+1);
    /* Fill local array with values */
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        buf[idx] = ii*DIMSIZE+jj;
      }
    }
    t_chk += (GA_Wtime()-tbeg);
    /* copy data to global array */
    tbeg = GA_Wtime();
    NGA_Put(g_a, lo, hi, buf, &ld);
    tput += (GA_Wtime()-tbeg);
    t_put += (GA_Wtime()-tbeg);
    put_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    /* Verify that original distribution is correct */
    if (n == 0) {
      NGA_Distribution(g_a,rank,tlo,thi);
      if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
        int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
        if (a_array_on_device) {
          tbuf = (int*)malloc(tnelem*sizeof(int));
          NGA_Access(g_a,tlo,thi,&ptr,&tld);
          for (i=0; i<tnelem; i++) tbuf[i] = 0;
          memcpyD2H(tbuf, ptr, tnelem*sizeof(int));
        } else {
          NGA_Access(g_a,tlo,thi,&tbuf,&tld);
        }
        ok = 1;
        for (ii = tlo[0]; ii<=thi[0]; ii++) {
          i = ii-tlo[0];
          for (jj=tlo[1]; jj<=thi[1]; jj++) {
            j = jj-tlo[1];
            idx = i*tld+j;
            if (tbuf[idx] != ii*DIMSIZE+jj) {
              if (ok) printf("p[%d] (%d,%d) (put) expected: %d actual[%d]: %d\n",
                  rank,ii,jj,ii*DIMSIZE+jj,idx,tbuf[idx]);
              ok = 0;
            }
          }
        }
        if (!ok) {
          printf("Mismatch found in initial data on process %d\n",rank);
        } else if (n==0 && rank==0 && ok) {
          printf("Initial data is okay\n");
        }
        if (a_array_on_device) {
          free(tbuf);
        }
        NGA_Release(g_a,tlo,thi);
      }
      tbeg = GA_Wtime();
      GA_Sync();
      t_sync += (GA_Wtime()-tbeg);
    }


    /* copy data from global array a to global array b */
    tbeg = GA_Wtime();
    NGA_Copy_patch('N',g_a,plo,phi,g_b,plo,phi);
    tcpy += (GA_Wtime()-tbeg);
    t_cpy += (GA_Wtime()-tbeg);
    cpy_cnt += nsize;

    /* check values in g_b */
    tbeg = GA_Wtime();
    NGA_Get(g_b, lo, hi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    t_get += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    ok = 1;
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        if (ii >= plo[0] && ii <= phi[0] && jj >= plo[1] && jj >= phi[1]) { 
          if (buf[idx] != ii*DIMSIZE+jj) {
            if (ok) printf("p[%d] (%d,%d) (get) expected: %d"
                " actual[%d]: %d\n",rank,ii,jj,
                ii*DIMSIZE+jj,idx,buf[idx]);
            ok = 0;
          }
        } else {
          if (buf[idx] != 0) {
            if (ok) printf("p[%d] (%d,%d) (get) expected: 0"
                " actual[%d]: %d\n",rank,ii,jj,idx,buf[idx]);
            ok = 0;
          }
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);
  }

  free(buf);

  tbeg = GA_Wtime();
  GA_Destroy(g_a);
  GA_Destroy(g_b);
  t_free += (GA_Wtime()-tbeg);

  if (!ok) {
    printf("Mismatch found for copy_patch on process %d after copy\n",rank);
  } else {
    if (rank == 0) printf("Copy_patch is okay\n");
  }

  GA_Igop(&put_cnt, 1, "+");
  GA_Igop(&get_cnt, 1, "+");
  GA_Igop(&cpy_cnt, 1, "+");
  GA_Dgop(&tput, 1, "+");
  GA_Dgop(&tget, 1, "+");
  GA_Dgop(&tcpy, 1, "+");
  put_bw = (double)(put_cnt*sizeof(int))/tput;
  get_bw = (double)(get_cnt*sizeof(int))/tget;
  cpy_bw = (double)(cpy_cnt*sizeof(int))/tcpy;
}

int main(int argc, char **argv) {

  int g_a;
  double *rbuf;
  double one_r;
  double t_sum;
  int zero = 0;
  int icnt;
  double tbeg;
  int local_buf_on_device;
  int i;
  
  t_put = 0.0;
  t_get = 0.0;
  t_cpy = 0.0;
  t_sync = 0.0;
  t_chk = 0.0;
  t_create = 0.0;
  t_free = 0.0;
  t_tot = 0.0;

  MPI_Init(&argc, &argv);


  MA_init(C_DBL, 2000000, 2000000);
  GA_Initialize();

  tbeg = GA_Wtime();
  nprocs = GA_Nnodes();  
  rank = GA_Nodeid();   
  // wrank = rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);

  int nodeid = GA_Cluster_nodeid();
  if (rank == 0) {
    printf("  Number of processors per node %d\n",GA_Cluster_nprocs(nodeid));
    printf("\n  Number of nodes %d\n",GA_Cluster_nnodes());
  }
  /* create list of GPU hosts */
  list = (int*)malloc(nprocs*sizeof(int));
  devIDs = (int*)malloc(nprocs*sizeof(int));
  NGA_Device_host_list(list, devIDs, &ndev, NGA_Pgroup_get_default());
  /* Determine if the process hosts a device */
  local_buf_on_device = 0;
  for (i=0; i<ndev; i++) {
     if (rank == list[i]) {
       local_buf_on_device = 1;
       my_dev = devIDs[i];
       break;
     }
  }

  /* Divide matrix up into pieces that are owned by each processor */
  factor(nprocs, &pdx, &pdy);
  if (rank == 0) {
    printf("\n  Test run on %d procs configured on %d X %d grid\n",nprocs,pdx,pdy);
    printf("\n  2D arrays are of size %d X %d\n",DIMSIZE,DIMSIZE);
    printf("\n  Number of loops in each test %d\n\n",NLOOP);
  }
  if (rank == 0) printf("  Testing integer array A on device, integer array B on host\n");
  test_int_array(1,0);
  print_bw();

  if (rank == 0) printf("  Testing integer array A on device, integer array B on device\n");
  test_int_array(1,1);
  print_bw();

  if (rank == 0) printf("  Testing integer array A on host, integer array B on host\n");
  test_int_array(0,0);
  print_bw();

  if (rank == 0) printf("  Testing integer array A on host, integer array B on device\n");
  test_int_array(0,1);
  print_bw();

  if (rank == 0) printf("  Testing double precision array A on device,"
      " double precision\n  array B on host\n");
  test_dbl_array(1,0);
  print_bw();

  if (rank == 0) printf("  Testing double precision array A on device,"
      " double precision\n  array B on device\n");
  test_dbl_array(1,1);
  print_bw();

  if (rank == 0) printf("  Testing double precision array A on host,"
      " double precision\n  array B on host\n");
  test_dbl_array(0,0);
  print_bw();

  if (rank == 0) printf("  Testing double precision array A on host,"
      " double precision\n  array B on device\n");
  test_dbl_array(0,1);
  print_bw();

  if (rank == 0) printf("  Testing copy patch on integer array A on device,\n"
      " integer  array B on host\n");
  test_dbl_array(1,0);
  print_bw();

  if (rank == 0) printf("  Testing copy patch on integer array A on device,\n"
      " integer  array B on device\n");
  test_dbl_array(1,1);
  print_bw();

  if (rank == 0) printf("  Testing copy patch on integer array A on host,\n"
      " integer  array B on host\n");
  test_dbl_array(0,0);
  print_bw();

  if (rank == 0) printf("  Testing copy patch on integer array A on host,\n"
      " integer  array B on device\n");
  test_dbl_array(0,1);
  print_bw();

  t_tot = GA_Wtime()-tbeg;
  /* Print out timing stats */
  GA_Dgop(&t_put,1,"+");
  GA_Dgop(&t_get,1,"+");
  GA_Dgop(&t_cpy,1,"+");
  GA_Dgop(&t_tot,1,"+");
  GA_Dgop(&t_sync,1,"+");
  GA_Dgop(&t_chk,1,"+");
  GA_Dgop(&t_create,1,"+");
  GA_Dgop(&t_free,1,"+");
  t_put /= ((double)nprocs);
  t_get /= ((double)nprocs);
  t_cpy /= ((double)nprocs);
  t_tot /= ((double)nprocs);
  t_sync /= ((double)nprocs);
  t_chk /= ((double)nprocs);
  t_create /= ((double)nprocs);
  t_free /= ((double)nprocs);
  t_sum = t_put + t_get + t_cpy + t_sync + t_chk;
  t_sum = t_sum + t_create + t_free;
  if (rank == 0) {
    printf("Total time in PUT:       %16.4e\n",t_put);
    printf("Total time in GET:       %16.4e\n",t_get);
    printf("Total time in COPY:      %16.4e\n",t_cpy);
    printf("Total time in SYNC :     %16.4e\n",t_sync);
    printf("Total time in CHECK:     %16.4e\n",t_chk);
    printf("Total time in CREATE:    %16.4e\n",t_create);
    printf("Total time in FREE :     %16.4e\n",t_free);
    printf("Total monitored time:    %16.4e\n",t_sum);
    printf("Total time:              %16.4e\n",t_tot);
  }

  free(list);
  free(devIDs);
  
  GA_Terminate();
  if (rank == 0) printf("Completed GA terminate\n");
  MPI_Finalize();
}
