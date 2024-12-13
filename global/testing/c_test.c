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

/*
#define BLOCK1 1024*1024
#define BLOCK1 65536
*/
#define BLOCK1 65530
#define DIMSIZE 512
#define SMLDIM 256
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

double tput, tget, tacc, tinc;
int get_cnt,put_cnt,acc_cnt;
double put_bw, get_bw, acc_bw;
double t_put, t_get, t_acc, t_sync, t_chk, t_tot;
double t_vput, t_rdinc;
double t_create, t_free;

void test_int_array(int on_device, int local_buf_on_device)
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
  double zero = 0.0;
  int ok;
  int me, lprocs, lpdx, lpdy;
 
  /* get local rank and default group size */
  me = GA_Nodeid();
  lprocs = GA_Nnodes();
  factor(lprocs,&lpdx,&lpdy);
  printf("p[%d] lprocs: %d lpdx: %d lpdy: %d\n",wrank,lprocs,lpdx,lpdy);
  tput = 0.0;
  tget = 0.0;
  tacc = 0.0;
  put_cnt = 0;
  get_cnt = 0;
  acc_cnt = 0;
  p_ok = 1;
  g_ok = 1;
  a_ok = 1;

  t_tot = 0.0;

  ndim = 2;
  dims[0] = DIMSIZE;
  dims[1] = DIMSIZE;
  xinc = DIMSIZE/lpdx;
  yinc = DIMSIZE/lpdy;
  ipx = me%lpdx;
  ipy = (me-ipx)/lpdx;
  isx = (ipx+1)%lpdx;
  isy = (ipy+1)%lpdy;
  /* Guarantee some data exchange between nodes */
  lo[0] = isx*xinc;
  lo[1] = isy*yinc;
  if (isx<lpdx-1) {
    hi[0] = (isx+1)*xinc-1;
  } else {
    hi[0] = DIMSIZE-1;
  }
  if (isy<lpdy-1) {
    hi[1] = (isy+1)*yinc-1;
  } else {
    hi[1] = DIMSIZE-1;
  }
  printf("p[%d] lo[0]: %d hi[0]: %d lo[1]: %d hi[i]: %d\n",
      wrank,lo[0],hi[0],lo[1],hi[1]);
  nelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);

  /* create a global array and initialize it to zero */
  tbeg = GA_Wtime();
  printf("p[%d] Got to 1\n",wrank);
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_INT);
  if (!on_device) {
    NGA_Set_restricted(g_a, list, ndev);
  }
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);
  t_create += (GA_Wtime()-tbeg);

  printf("p[%d] Got to 2\n",wrank);
  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  if (local_buf_on_device) {
    void *tbuf;
    set_device(&my_dev);
    device_malloc(&tbuf,(int)(nsize*sizeof(int)));
    buf = (int*)tbuf;
  } else {
    buf = (int*)malloc(nsize*sizeof(int));
  }

  for (n=0; n<NLOOP; n++) {
    tbeg = GA_Wtime();
  printf("p[%d] Got to 3\n",wrank);
    GA_Zero(g_a);
    GA_Fill(g_a,&zero);
  printf("p[%d] Got to 4\n",wrank);
    ld = (hi[1]-lo[1]+1);
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj = lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            tbuf[idx] = ii*DIMSIZE+jj;
          }
        }
        memcpyH2D(buf, tbuf, tnelem*sizeof(int));
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = ii*DIMSIZE+jj;
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);
    /* copy data to global array */
    tbeg = GA_Wtime();
  printf("p[%d] Got to 5\n",wrank);
    NGA_Put(g_a, lo, hi, buf, &ld);
  printf("p[%d] Got to 6\n",wrank);
    tput += (GA_Wtime()-tbeg);
    t_put += (GA_Wtime()-tbeg);
    put_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    NGA_Distribution(g_a,me,tlo,thi);
  printf("p[%d] Got to 7\n",wrank);
#if 1
    if (me == 0 && n == 0) printf("Completed NGA_Distribution\n");
    if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
      int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
      if (on_device) {
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
        printf("Mismatch found for put on process %d after Put\n",rank);
      } else if (n==0 && rank==0 && ok) {
        printf("Access function is okay\n");
      }
      NGA_Release(g_a,tlo,thi);
      if (on_device) {
        free(tbuf);
      }
    }
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
#endif
  printf("p[%d] Got to 8\n",wrank);

    /* zero out local buffer */
    if (local_buf_on_device) {
      int *tbuf = (int*)malloc(nsize*sizeof(int));
      for (i=0; i<nsize; i++) tbuf[i] = 0;
      memcpyH2D(buf, tbuf, nsize*sizeof(int));
      free(tbuf);
    } else {
      for (i=0; i<nsize; i++) buf[i] = 0;
    }
    t_chk += (GA_Wtime()-tbeg);

    /* copy data from global array to local buffer */
    tbeg = GA_Wtime();
    NGA_Get(g_a, lo, hi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    t_get += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (i=0; i<tnelem; i++) tbuf[i] = 0;
        memcpyD2H(tbuf, buf, tnelem*sizeof(int));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj=lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            if (tbuf[idx] != ii*DIMSIZE+jj) {
              if (g_ok) printf("p[%d] (%d,%d) (get) expected: %d"
                  " actual[%d]: %d\n",rank,ii,jj,ii*DIMSIZE+jj,idx,tbuf[idx]);
              g_ok = 0;
            }
          }
        }
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          if (buf[idx] != ii*DIMSIZE+jj) {
            if (g_ok) printf("p[%d] (%d,%d) (get) expected: %d"
                " actual[%d]: %d\n",rank,ii,jj,ii*DIMSIZE+jj,idx,buf[idx]);
            g_ok = 0;
          }
        }
      }
    }

#if 1
    /* reset values in buf */
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj = lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            tbuf[idx] = ii*DIMSIZE+jj;
          }
        }
        memcpyH2D(buf, tbuf, tnelem*sizeof(int));
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = ii*DIMSIZE+jj;
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    /* accumulate data to global array */
    one = 1;
    tbeg = GA_Wtime();
    NGA_Acc(g_a, lo, hi, buf, &ld, &one);
    tacc += (GA_Wtime()-tbeg);
    t_acc += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    acc_cnt += nsize;
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    /* reset values in buf */
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (i=0; i<nelem; i++) tbuf[i] = 0;
        memcpyH2D(buf, tbuf, tnelem*sizeof(int));
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = 0;
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    NGA_Get(g_a, lo, hi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    t_get += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (int*)malloc(tnelem*sizeof(int));
        for (i=0; i<tnelem; i++) tbuf[i] = 0;
        memcpyD2H(tbuf, buf, tnelem*sizeof(int));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj=lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            if (tbuf[idx] != 2*(ii*DIMSIZE+jj)) {
              if (a_ok) printf("p[%d] (%d,%d) (acc) expected: %d"
                  " actual[%d]: %d device: %d\n",rank,ii,jj,
                  2*(ii*DIMSIZE+jj),idx,tbuf[idx],on_device);
              a_ok = 0;
            }
          }
        }
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          if (buf[idx] != 2*(ii*DIMSIZE+jj)) {
            if (a_ok) printf("p[%d] (%d,%d) (acc) expected: %d"
                " actual[%d]: %d device: %d\n",rank,ii,jj,
                2*(ii*DIMSIZE+jj),idx,buf[idx],on_device);
            a_ok = 0;
          }
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);
#endif
  }

  if (local_buf_on_device) {
    device_free(buf);
  } else {
    free(buf);
  }
  tbeg = GA_Wtime();
  GA_Destroy(g_a);
  t_free += (GA_Wtime()-tbeg);

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
  int me = GA_Nodeid();
  /*      12345678901234567890123456789012345678901234567890123456789012345 */
  if (me != 0) return;

  printf("///////////////////////////////////////////////////////////////\n");
  printf("//      Put (MB/sec) //     Get (MB/sec) //     Acc (MB/sec) //\n");
  printf("//  %15.4e  // %15.4e  // %15.4e  //\n",put_bw*rB,get_bw*rB,acc_bw*rB);
  printf("///////////////////////////////////////////////////////////////\n");
  printf("\n\n");
}

void test_dbl_array(int on_device, int local_buf_on_device)
{
  int g_a;
  int i, j, ii, jj, idx, n;
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
  int p_ok, g_ok, a_ok;
  double *ptr;
  double one;
  double tbeg;
  int ok;

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
  tbeg = GA_Wtime();
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_DBL);
  if (!on_device) {
    NGA_Set_restricted(g_a, list, ndev);
  }
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);
  GA_Zero(g_a);
  t_create += (GA_Wtime()-tbeg);

  /* allocate a local buffer and initialize it with values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  if (local_buf_on_device) {
    void *tbuf;
    set_device(&my_dev);
    device_malloc(&tbuf,(int)(nsize*sizeof(double)));
    buf = (double*)tbuf;
  } else {
    buf = (double*)malloc(nsize*sizeof(double));
  }

  for (n=0; n<NLOOP; n++) {
    tbeg = GA_Wtime();
    GA_Zero(g_a);
    ld = (hi[1]-lo[1]+1);
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (double*)malloc(tnelem*sizeof(double));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj = lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            tbuf[idx] = (double)(ii*DIMSIZE+jj);
          }
        }
        memcpyH2D(buf, tbuf, tnelem*sizeof(double));
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = (double)(ii*DIMSIZE+jj);
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);
    /* copy data to global array */
    tbeg = GA_Wtime();
    NGA_Put(g_a, lo, hi, buf, &ld);
    tput += (GA_Wtime() - tbeg);
    t_put += (GA_Wtime() - tbeg);
    put_cnt += nelem;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    NGA_Distribution(g_a,rank,tlo,thi);
#if 1
    if (n == 0) {
      if (rank == 0) printf("Completed NGA_Distribution\n",rank);
      if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
        int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
        double *tbuf;
        if (on_device) {
          tbuf = (double*)malloc(tnelem*sizeof(double));
          NGA_Access(g_a,tlo,thi,&ptr,&tld);
          for (i=0; i<tnelem; i++) tbuf[i] = 0.0;
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
        NGA_Release(g_a,tlo,thi);
        if (on_device) {
          free(tbuf);
        }
      }
    }
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
#endif

    /* zero out local buffer */
    if (local_buf_on_device) {
      double *tbuf = (double*)malloc(nsize*sizeof(double));
      for (i=0; i<nsize; i++) tbuf[i] = 0.0;
      memcpyH2D(buf, tbuf, nsize*sizeof(double));
      free(tbuf);
    } else {
      for (i=0; i<nsize; i++) buf[i] = 0.0;
    }
    t_chk += (GA_Wtime()-tbeg);

    /* copy data from global array to local buffer */
    tbeg = GA_Wtime();
    NGA_Get(g_a, lo, hi, buf, &ld);
    tget += (GA_Wtime() - tbeg);
    t_get += (GA_Wtime() - tbeg);
    get_cnt += nelem;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    g_ok = 1;
    if (local_buf_on_device) {
      double *tbuf = (double*)malloc(nsize*sizeof(double));
      memcpyD2H(tbuf, buf, nsize*sizeof(double));
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          if (tbuf[idx] != (double)(ii*DIMSIZE+jj)) {
            if (g_ok) printf("p[%d] (%d,%d) expected: %f actual[%d]: %f\n",rank,ii,jj,
                (double)(ii*DIMSIZE+jj),idx,tbuf[idx]);
            g_ok = 0;
          }
        }
      }
      free(tbuf);
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          if (buf[idx] != (double)(ii*DIMSIZE+jj)) {
            if (g_ok) printf("p[%d] (%d,%d) expected: %f actual[%d]: %f\n",rank,ii,jj,
                (double)(ii*DIMSIZE+jj),idx,buf[idx]);
            g_ok = 0;
          }
        }
      }
    }

    /* reset values in buf */
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (double*)malloc(tnelem*sizeof(double));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj = lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            tbuf[idx] = (double)(ii*DIMSIZE+jj);
          }
        }
        memcpyH2D(buf, tbuf, tnelem*sizeof(double));
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = (double)(ii*DIMSIZE+jj);
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    /* accumulate data to global array */
    one = 1.0;
    tbeg = GA_Wtime();
    NGA_Acc(g_a, lo, hi, buf, &ld, &one);
    tacc += (GA_Wtime() - tbeg);
    t_acc += (GA_Wtime() - tbeg);
    acc_cnt += nelem;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    /* reset values in buf */
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (double*)malloc(tnelem*sizeof(double));
        for (i=0; i<nelem; i++) tbuf[i] = 0.0;
        memcpyH2D(buf, tbuf, tnelem*sizeof(double));
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          buf[idx] = 0.0;
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    NGA_Get(g_a, lo, hi, buf, &ld);
    tget += (GA_Wtime() - tbeg);
    t_get += (GA_Wtime() - tbeg);
    get_cnt += nelem;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    if (local_buf_on_device) {
      if (lo[0]<=hi[0] && lo[1]<=hi[1]) {
        int tnelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
        tbuf = (double*)malloc(tnelem*sizeof(double));
        for (i=0; i<tnelem; i++) tbuf[i] = 0;
        memcpyD2H(tbuf, buf, tnelem*sizeof(double));
        for (ii = lo[0]; ii<=hi[0]; ii++) {
          i = ii-lo[0];
          for (jj=lo[1]; jj<=hi[1]; jj++) {
            j = jj-lo[1];
            idx = i*ld+j;
            if (tbuf[idx] != (double)(2*(ii*DIMSIZE+jj))) {
              if (a_ok) printf("p[%d] (%d,%d) (acc) expected: %f"
                  " actual[%d]: %f device: %d\n",rank,ii,jj,
                  (double)(2*(ii*DIMSIZE+jj)),idx,buf[idx],on_device);
              a_ok = 0;
            }
          }
        }
        free(tbuf);
      }
    } else {
      for (ii = lo[0]; ii<=hi[0]; ii++) {
        i = ii-lo[0];
        for (jj = lo[1]; jj<=hi[1]; jj++) {
          j = jj-lo[1];
          idx = i*ld+j;
          if (buf[idx] != (double)(2*(ii*DIMSIZE+jj))) {
            if (a_ok) printf("p[%d] (%d,%d) (acc) expected: %f actual[%d]: %f\n",
                rank,ii,jj,(double)(2*(ii*DIMSIZE+jj)),idx,buf[idx]);
            a_ok = 0;
          }
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);
  }
  if (local_buf_on_device) {
    device_free(buf);
  } else {
    free(buf);
  }
  tbeg = GA_Wtime();
  GA_Destroy(g_a);
  t_free += (GA_Wtime()-tbeg);

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
  put_bw = (double)(put_cnt*sizeof(double))/tput;
  get_bw = (double)(get_cnt*sizeof(double))/tget;
  acc_bw = (double)(acc_cnt*sizeof(double))/tacc;
}

void test_read_inc(int on_device)
{
  int g_a;
  int one;
  int icnt, zero, i, nvals;
  int ri_cnt = 0;
  double tbeg;
  double t_ri = 0.0;
  int *bins;
  /* create a global array and initialize it to zero */
  zero = 0;
  one = 1;
  nvals = MAXCOUNT/1000;
  bins = (int*)malloc((nvals+1)*sizeof(int));
  for (i=0; i<nvals+1; i++) bins[i] = 0;
  tbeg = GA_Wtime();
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, one, &one, C_INT);
  if (!on_device) {
    NGA_Set_restricted(g_a, list, ndev);
  }
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);
  t_create += (GA_Wtime()-tbeg);

  GA_Zero(g_a);
  if (rank == 0) printf("Created and initialized global array with 1 element\n");
  icnt = 0;
  while (icnt<MAXCOUNT+1) {
    tbeg = GA_Wtime();
    icnt = NGA_Read_inc(g_a,&zero,(long)one);
    t_ri += (GA_Wtime()-tbeg);
    ri_cnt++;
    if (icnt%1000 == 0) {
      i = icnt/1000;
      if (i<=nvals) bins[i] = rank;
    }
  }

  tbeg = GA_Wtime();
  GA_Sync();
  t_sync += (GA_Wtime()-tbeg);
  GA_Igop(bins, nvals+1, "+");
  if (rank == 0) {
    for (i=0; i<=nvals; i++) {
      printf("  current value of counter: %d read on process %d\n",i*1000,bins[i]);
    }
  }
  if (rank == 0) {
    NGA_Get(g_a,&zero,&zero,&i,&zero);
    if (i != MAXCOUNT+1+nprocs) {
      printf ("Mismatch found for read-increment expected: %d actual: %d\n",
          MAXCOUNT+nprocs,i);
    } else {
      printf("Read-increment is OK\n");
    }
  }
  tbeg = GA_Wtime();
  GA_Destroy(g_a);
  t_free += (GA_Wtime()-tbeg);
  t_rdinc += t_ri;
  GA_Igop(&ri_cnt, 1, "+");
  GA_Dgop(&t_ri, 1, "+");
  t_ri /= ((double)ri_cnt);
  if (rank == 0) {
    printf("///////////////////////////////////////////////////////////////\n");
    printf("//                  micro seconds per read-increment         //\n");
    printf("//               %15.4f                             //\n",t_ri*1.0e6);
    printf("///////////////////////////////////////////////////////////////\n");
    printf("\n\n");
  }
  free(bins);
}

void test_int_1d_array(int on_device, int local_buf_on_device)
{
  int g_a;
  int i, ii, ld, n;
  int tlo[1], thi[1];
  int nelem;
  int nsize;
  int *buf;
  int one;
  int p_ok, g_ok, a_ok;
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

  one = 1;
  nelem = BLOCK1*nprocs;
  tbeg = GA_Wtime();
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, one, &nelem, C_INT);
  if (!on_device) {
    NGA_Set_restricted(g_a, list, ndev);
  }
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);
  t_create += (GA_Wtime()-tbeg);
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
  if (local_buf_on_device) {
    void *tbuf;
    set_device(&my_dev);
    device_malloc(&tbuf,(int)(nsize*sizeof(int)));
    buf = (int*)tbuf;
  } else {
    buf = (int*)malloc(nsize*sizeof(int));
  }
  ld = 1;
  for (n=0; n<NLOOP; n++) {
    tbeg = GA_Wtime();
    GA_Zero(g_a);
    if (local_buf_on_device) {
      int *tbuf = (int*)malloc(nsize*sizeof(int));
      for (ii= tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        tbuf[i] = ii;
      }
      memcpyH2D(buf, tbuf, nsize*sizeof(int));
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        buf[i] = ii;
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    /* copy data to global array */
    tbeg = GA_Wtime();
    NGA_Put(g_a, tlo, thi, buf, &ld);
    tput += (GA_Wtime()-tbeg);
    t_put += (GA_Wtime()-tbeg);
    put_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    /* zero out local buffer */
    if (local_buf_on_device) {
      int *tbuf = (int*)malloc(nsize*sizeof(int));
      for (i=0; i<nsize; i++) tbuf[i] = 0;
      memcpyH2D(buf, tbuf, nsize*sizeof(int));
      free(tbuf);
    } else {
      for (i=0; i<nsize; i++) buf[i] = 0;
    }
    t_chk += (GA_Wtime()-tbeg);

    /* copy data from global array to local buffer */
    tbeg = GA_Wtime();
    NGA_Get(g_a, tlo, thi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    t_get += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    g_ok = 1;
    if (local_buf_on_device) {
      int *tbuf = (int*)malloc(nsize*sizeof(int));
      memcpyD2H(tbuf, buf, nsize*sizeof(int));
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        if (tbuf[i] != ii) {
          if (g_ok) printf("p[%d] index: %d expected: %f actual: %f\n",rank,i,ii,tbuf[i]);
          g_ok = 0;
        }
      }
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        if (buf[i] != ii) {
          if (g_ok) printf("p[%d] index: %d expected: %f actual: %f\n",rank,i,ii,buf[i]);
          g_ok = 0;
        }
      }
    }

    /* reset values in buf */
    if (local_buf_on_device) {
      int *tbuf = (int*)malloc(nsize*sizeof(int));
      for (ii= tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        tbuf[i] = ii;
      }
      memcpyH2D(buf, tbuf, nsize*sizeof(int));
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        buf[i] = ii;
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    /* accumulate data to global array */
    tbeg = GA_Wtime();
    NGA_Acc(g_a, tlo, thi, buf, &ld, &one);
    tacc += (GA_Wtime()-tbeg);
    t_acc += (GA_Wtime()-tbeg);
    acc_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
#if 0
    NGA_Distribution(g_a,rank,lo,hi);
    if (lo[0]<=hi[0]) {
      int *tbuf;
      int *ptr;
      int tnelem = hi[0]-lo[0]+1;
      tbuf = (int*)malloc(tnelem*sizeof(int));
      NGA_Access(g_a,lo,hi,&ptr,&tld);
      for (i=0; i<tnelem; i++) tbuf[i] = 0;
      memcpyD2H(tbuf, ptr, tnelem*sizeof(int));
      if (tnelem > 0) {
        printf("p[%d] acc buffer:",rank);
        for (i=0; i<tnelem; i++) printf(" %d",tbuf[i]);
        printf("\n"); 
      }
      free(tbuf);
    }
#endif
    /* reset values in buf */
    if (local_buf_on_device) {
      int *tbuf = (int*)malloc(nsize*sizeof(int));
      for (ii= tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        tbuf[i] = 0;
      }
      memcpyH2D(buf, tbuf, nsize*sizeof(int));
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        buf[i] = 0;
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    NGA_Get(g_a, tlo, thi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    t_get += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    a_ok = 1;
    if (local_buf_on_device) {
      int *tbuf = (int*)malloc(nsize*sizeof(int));
      memcpyD2H(tbuf, buf, nsize*sizeof(int));
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        if (tbuf[i] != 2*ii) {
          if (a_ok) printf("p[%d] index: %d expected: %d actual: %d\n",
              rank,i,2*ii,tbuf[i]);
          a_ok = 0;
        }
      }
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        if (buf[i] != 2*ii) {
          if (a_ok) printf("p[%d] index: %d expected: %d actual: %d\n",rank,i,2*ii,buf[i]);
          a_ok = 0;
        }
      }
    } 
    t_chk += (GA_Wtime()-tbeg);
  }
  if (local_buf_on_device) {
    device_free(buf);
  } else {
    free(buf);
  }
  tbeg = GA_Wtime();
  GA_Destroy(g_a);
  t_free += (GA_Wtime()-tbeg);

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

void test_dbl_1d_array(int on_device, int local_buf_on_device)
{
  int g_a;
  int i, ii, ld, n;
  int tlo[1], thi[1];
  int nelem;
  int nsize;
  double *buf;
  double one;
  int p_ok, g_ok, a_ok;
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

  one = 1.0;
  nelem = BLOCK1*nprocs;
  tbeg = GA_Wtime();
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, one, &nelem, C_DBL);
  if (!on_device) {
    NGA_Set_restricted(g_a, list, ndev);
  }
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);
  t_create += (GA_Wtime()-tbeg);
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
  if (local_buf_on_device) {
    void *tbuf;
    set_device(&my_dev);
    device_malloc(&tbuf,(int)(nsize*sizeof(double)));
    buf = (double*)tbuf;
  } else {
    buf = (double*)malloc(nsize*sizeof(double));
  }
  ld = 1;
  for (n=0; n<NLOOP; n++) { 
    tbeg = GA_Wtime();
    GA_Zero(g_a);
    if (local_buf_on_device) {
      double *tbuf = (double*)malloc(nsize*sizeof(double));
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        tbuf[i] = (double)ii;
      }
      memcpyH2D(buf, tbuf, nsize*sizeof(double));
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        buf[i] = (double)ii;
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    /* copy data to global array */
    tbeg = GA_Wtime();
    NGA_Put(g_a, tlo, thi, buf, &ld);
    tput += (GA_Wtime()-tbeg);
    t_put += (GA_Wtime()-tbeg);
    put_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);

    /* zero out local buffer */
    if (local_buf_on_device) {
      double *tbuf = (double*)malloc(nsize*sizeof(double));
      for (i=0; i<nsize; i++) tbuf[i] = 0;
      memcpyH2D(buf, tbuf, nsize*sizeof(double));
      free(tbuf);
    } else {
      for (i=0; i<nsize; i++) buf[i] = 0.0;
    }

    /* copy data from global array to local buffer */
    tbeg = GA_Wtime();
    NGA_Get(g_a, tlo, thi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    t_get += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    g_ok = 1;
    if (local_buf_on_device) {
      double *tbuf = (double*)malloc(nsize*sizeof(double));
      memcpyD2H(tbuf, buf, nsize*sizeof(double));
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        if (tbuf[i] != (double)ii) {
          if (g_ok) printf("p[%d] index: %d expected: %f actual: %f\n",
              rank,i,(double)ii,tbuf[i]);
          g_ok = 0;
        }
      }
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        if (buf[i] != (double)ii) {
          if (g_ok) printf("p[%d] index: %d expected: %f actual: %f\n",
              rank,i,(double)ii,buf[i]);
          g_ok = 0;
        }
      }
    }

    /* reset values in buf */
    if (local_buf_on_device) {
      double *tbuf = (double*)malloc(nsize*sizeof(double));
      for (ii= tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        tbuf[i] = (double)ii;
      }
      memcpyH2D(buf, tbuf, nsize*sizeof(double));
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        buf[i] = (double)ii;
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    /* accumulate data to global array */
    tbeg = GA_Wtime();
    NGA_Acc(g_a, tlo, thi, buf, &ld, &one);
    tacc += (GA_Wtime()-tbeg);
    t_acc += (GA_Wtime()-tbeg);
    acc_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
#if 0
    NGA_Distribution(g_a,rank,lo,hi);
    if (lo[0]<=hi[0]) {
      double *tbuf;
      double *ptr;
      int tnelem = hi[0]-lo[0]+1;
      tbuf = (double*)malloc(tnelem*sizeof(double));
      NGA_Access(g_a,lo,hi,&ptr,&tld);
      for (i=0; i<tnelem; i++) tbuf[i] = 0;
      memcpyD2H(tbuf, ptr, tnelem*sizeof(double));
      if (tnelem > 0) {
        printf("p[%d] acc buffer:",rank);
        for (i=0; i<tnelem; i++) printf(" %f",tbuf[i]);
        printf("\n"); 
      }
      free(tbuf);
    }
#endif
    /* reset values in buf */
    if (local_buf_on_device) {
      double *tbuf = (double*)malloc(nsize*sizeof(double));
      for (ii= tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        tbuf[i] = 0.0;
      }
      memcpyH2D(buf, tbuf, nsize*sizeof(double));
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        buf[i] = 0.0;
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    NGA_Get(g_a, tlo, thi, buf, &ld);
    tget += (GA_Wtime()-tbeg);
    t_get += (GA_Wtime()-tbeg);
    get_cnt += nsize;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    a_ok = 1;
    if (local_buf_on_device) {
      double *tbuf = (double*)malloc(nsize*sizeof(double));
      memcpyD2H(tbuf, buf, nsize*sizeof(double));
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        if (tbuf[i] != (double)(2*ii)) {
          if (a_ok) printf("p[%d] index: %d expected: %d actual: %d\n",
              rank,i,(double)(2*ii),tbuf[i]);
          a_ok = 0;
        }
      }
      free(tbuf);
    } else {
      for (ii = tlo[0]; ii<=thi[0]; ii++) {
        i = ii-tlo[0];
        if (buf[i] != (double)(2*ii)) {
          if (a_ok) printf("p[%d] index: %d expected: %f actual: %f\n",
              rank,i,(double)(2*ii),buf[i]);
          a_ok = 0;
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);
  }
  if (local_buf_on_device) {
    device_free(buf);
  } else {
    free(buf);
  }
  tbeg = GA_Wtime();
  GA_Destroy(g_a);
  t_free += (GA_Wtime()-tbeg);
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
  put_bw = (double)(put_cnt*sizeof(double))/tput;
  get_bw = (double)(get_cnt*sizeof(double))/tget;
  acc_bw = (double)(acc_cnt*sizeof(double))/tacc;
}

void test_dbl_scatter(int on_device, int local_buf_on_device)
{
  int g_a;
  int i, j, ii, jj, idx, n;
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
  int p_ok, g_ok, a_ok;
  double *ptr;
  double one;
  double tbeg;
  double *vals;
  int *subsBuf;
  int **subsArray;
  int nvals;
  int arraysize;
  int icnt;

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
  dims[0] = SMLDIM;
  dims[1] = SMLDIM;
  xinc = SMLDIM/pdx;
  yinc = SMLDIM/pdy;
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
    hi[0] = SMLDIM-1;
  }
  if (isy<pdy-1) {
    hi[1] = (isy+1)*yinc-1;
  } else {
    hi[1] = SMLDIM-1;
  }
  nelem = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  /* Set up arrays for scattering values */
  arraysize = dims[0]*dims[1];
  nvals = arraysize/nprocs;
  if (nvals*nprocs != arraysize) {
    int delta = arraysize-nvals*nprocs;
    if (rank<delta) nvals++;
  }

  /* create a global array and initialize it to zero */
  tbeg = GA_Wtime();
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_DBL);
  if (!on_device) {
    NGA_Set_restricted(g_a, list, ndev);
  }
  NGA_Set_device(g_a, on_device);
  NGA_Allocate(g_a);
  GA_Zero(g_a);
  t_create += (GA_Wtime()-tbeg);

  /* allocate a local buffer for checking values*/
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  buf = (double*)malloc(nsize*sizeof(double));

  /* allocate buffers for scatter operation */
  tbuf = (double*)malloc(nvals*sizeof(double));
  subsBuf = (int*)malloc(2*nvals*sizeof(int));
  subsArray = (int**)malloc(nvals*sizeof(int*));
  if (local_buf_on_device) {
    void *vbuf;
    set_device(&my_dev);
    device_malloc(&vbuf, nvals*sizeof(double));
    vals = (double*)vbuf;
  } else {
    vals = (double*)malloc(nvals*sizeof(double));
  }
  /* initialize indices */
  icnt = 0;
  for (n=rank; n<arraysize; n+=nprocs)
  {
    j = n%dims[1]; 
    i = (n-j)/dims[1];
    tbuf[icnt] = (double)(i*dims[1]+j);
    subsArray[icnt] = &subsBuf[2*icnt];
    subsArray[icnt][0] = i;
    subsArray[icnt][1] = j;
    icnt++;
  }
  if (local_buf_on_device) {
    memcpyH2D(vals, tbuf, nvals*sizeof(double));
  } else {
    for (i=0; i<nvals; i++) vals[i] = tbuf[i];
  }

  for (n=0; n<NLOOP; n++) {
    tbeg = GA_Wtime();
    GA_Zero(g_a);
    ld = (hi[1]-lo[1]+1);
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        buf[idx] = 0.0;
      }
    }
    t_chk += (GA_Wtime()-tbeg);
    /* copy data to global array */
    tbeg = GA_Wtime();
    NGA_Scatter(g_a, vals, subsArray, nvals);
    tput += (GA_Wtime() - tbeg);
    t_put += (GA_Wtime() - tbeg);
    put_cnt += nvals;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
#if 1
    tbeg = GA_Wtime();
    NGA_Distribution(g_a,rank,tlo,thi);
    if (n == 0) {
      if (rank == 0) printf("Completed NGA_Distribution\n",rank);
      if (tlo[0]<=thi[0] && tlo[1]<=thi[1]) {
        int tnelem = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
        double *tbuf;
        if (on_device) {
          tbuf = (double*)malloc(tnelem*sizeof(double));
          NGA_Access(g_a,tlo,thi,&ptr,&tld);
          for (i=0; i<tnelem; i++) tbuf[i] = 0.0;
          memcpyD2H(tbuf, ptr, tnelem*sizeof(double));
        } else {
          NGA_Access(g_a,tlo,thi,&tbuf,&tld);
        }
        p_ok = 1;
        for (ii = tlo[0]; ii<=thi[0]; ii++) {
          i = ii-tlo[0];
          for (jj=tlo[1]; jj<=thi[1]; jj++) {
            j = jj-tlo[1];
            idx = i*tld+j;
            if (tbuf[idx] != (double)(ii*SMLDIM+jj)) {
              if (p_ok) printf("p[%d] (%d,%d) expected: %f actual[%d]: %f\n",rank,ii,jj,
                  (double)(ii*SMLDIM+jj),idx,tbuf[idx]);
              p_ok = 0;
            }
          }
        }
        if (!p_ok) {
          printf("Mismatch found for scatter on process %d after Put\n",rank);
        } else {
          if (rank==0) printf("Scatter is okay\n");
        }
        NGA_Release(g_a,tlo,thi);
        if (on_device) {
          free(tbuf);
        }
      }
    }
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);

    /* zero out local buffer */
    if (local_buf_on_device) {
      for (i=0; i<nvals; i++) tbuf[i] = 0.0;
      memcpyH2D(vals, tbuf, nvals*sizeof(double));
    } else {
      for (i=0; i<nvals; i++) vals[i] = 0.0;
    }
    t_chk += (GA_Wtime()-tbeg);

    /* gather data from global array to local buffer */
    tbeg = GA_Wtime();
    NGA_Gather(g_a, vals, subsArray, nvals);
    tget += (GA_Wtime() - tbeg);
    t_get += (GA_Wtime() - tbeg);
    get_cnt += nelem;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);

    /* check values */
    tbeg = GA_Wtime();
    g_ok = 1;
    icnt = 0;
    if (local_buf_on_device) {
      memcpyD2H(tbuf, vals, nvals*sizeof(double));
      for (n=rank; n<arraysize; n+=nprocs)
      {
        j = n%dims[1]; 
        i = (n-j)/dims[1];
        if (fabs(tbuf[icnt]-(double)(i*dims[1]+j)) > 1.0e-12) {
          if (g_ok) printf("p[%d] put/get (%d,%d) expected: %f actual[%d]: %f\n",wrank,i,j,
              (double)(i*SMLDIM+j),n,tbuf[icnt]);
          g_ok = 0;
        }
        icnt++;
      }
    } else {
      for (n=rank; n<arraysize; n+=nprocs)
      {
        j = n%dims[1]; 
        i = (n-j)/dims[1];
        if (fabs(vals[icnt]-(double)(i*dims[1]+j)) > 1.0e-12) {
          if (g_ok) printf("p[%d] put/get (%d,%d) expected: %f actual[%d]: %f\n",rank,i,j,
              (double)(i*SMLDIM+j),n,vals[icnt]);
          g_ok = 0;
        }
        icnt++;
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    /* accumulate data to global array */
    one = 1.0;
    tbeg = GA_Wtime();
    NGA_Scatter_acc(g_a, vals, subsArray, nvals, &one);
    tacc += (GA_Wtime() - tbeg);
    t_acc += (GA_Wtime() - tbeg);
    acc_cnt += nelem;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    /* reset values in buf */
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        buf[idx] = 0.0;
      }
    }
    t_chk += (GA_Wtime()-tbeg);

    tbeg = GA_Wtime();
    NGA_Get(g_a, lo, hi, buf, &ld);
    tget += (GA_Wtime() - tbeg);
    t_get += (GA_Wtime() - tbeg);
    get_cnt += nelem;
    tbeg = GA_Wtime();
    GA_Sync();
    t_sync += (GA_Wtime()-tbeg);
    tbeg = GA_Wtime();
    a_ok = 1;
    for (ii = lo[0]; ii<=hi[0]; ii++) {
      i = ii-lo[0];
      for (jj = lo[1]; jj<=hi[1]; jj++) {
        j = jj-lo[1];
        idx = i*ld+j;
        if (fabs(buf[idx]-(double)(2*(ii*dims[1]+jj))) > 1.0e-12) {
          if (a_ok) printf("p[%d] acc (%d,%d) expected: %f actual[%d]: %f\n",rank,ii,jj,
              (double)(2*(ii*SMLDIM+jj)),idx,buf[idx]);
          a_ok = 0;
        }
      }
    }
    t_chk += (GA_Wtime()-tbeg);
#endif
  }
  free(buf);
  free(tbuf);
  if (local_buf_on_device) {
    device_free(vals);
  } else {
    free(vals);
  }
  free(subsBuf);
  free(subsArray);
  tbeg = GA_Wtime();
  GA_Destroy(g_a);
  t_free += (GA_Wtime()-tbeg);

  if (!g_ok) {
    printf("Mismatch found for gather on process %d after Get\n",rank);
  } else {
    if (rank == 0) printf("Gather is okay\n");
  }
  if (!a_ok) {
    printf("Mismatch found on process %d after scatteracc\n",rank);
  } else {
    if (rank == 0) printf("Scatter_acc is okay\n");
  }

  GA_Igop(&put_cnt, 1, "+");
  GA_Igop(&get_cnt, 1, "+");
  GA_Igop(&acc_cnt, 1, "+");
  GA_Dgop(&tput, 1, "+");
  GA_Dgop(&tget, 1, "+");
  GA_Dgop(&tacc, 1, "+");
  put_bw = (double)(put_cnt*sizeof(double))/tput;
  get_bw = (double)(get_cnt*sizeof(double))/tget;
  acc_bw = (double)(acc_cnt*sizeof(double))/tacc;
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
  t_acc = 0.0;
  t_sync = 0.0;
  t_chk = 0.0;
  t_rdinc = 0.0;
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
  /* Determine if the process host a device */
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
    printf("\n  1D arrays are of size %d\n",BLOCK1*nprocs);
    printf("\n  Number of loops in each test %d\n\n",NLOOP);
  }
#if 0
  if (rank == 0) printf("  Testing integer array on device, local buffer on host\n");
  test_int_array(1,0);
  print_bw();

  if (rank == 0) printf("  Testing integer array on device, local buffer on device\n");
  test_int_array(1,local_buf_on_device);
  print_bw();

  if (rank == 0) printf("  Testing integer array on host, local buffer on host\n");
  test_int_array(0,0);
  print_bw();

  if (rank == 0) printf("  Testing integer array on host, local buffer on device\n");
  test_int_array(0,local_buf_on_device);
  print_bw();

  if (rank == 0) printf("  Testing double precision array on device,"
      " local buffer on host\n");
  test_dbl_array(1,0);
  print_bw();

  if (rank == 0) printf("  Testing double precision array on device,"
      " local buffer on device\n");
  test_dbl_array(1,local_buf_on_device);
  print_bw();

  if (rank == 0) printf("  Testing double precision array on host,"
      " local buffer on host\n");
  test_dbl_array(0,0);
  print_bw();

  if (rank == 0) printf("  Testing double precision array on host,"
      " local buffer on device\n");
  test_dbl_array(0,local_buf_on_device);
  print_bw();

  if (rank == 0) printf("  Testing read-increment function on device\n");
  test_read_inc(1);
  
  if (rank == 0) printf("  Testing read-increment function on host\n");
  test_read_inc(0);
  
  if (rank == 0) printf("  Testing contiguous one-sided operations"
      " for integer array\n  on device, local buffer on host\n");
  test_int_1d_array(1,0);
  print_bw();

  if (rank == 0) printf("  Testing contiguous one-sided operations"
      " for integer array\n  on device, local buffer on device\n");
  test_int_1d_array(1,local_buf_on_device);
  print_bw();

  if (rank == 0) printf("  Testing contiguous one-sided operations"
      " for integer array\n  on host, local buffer on host\n");
  test_int_1d_array(0,0);
  print_bw();

  if (rank == 0) printf("  Testing contiguous one-sided operations"
      " for integer array\n  on host, local buffer on device\n");
  test_int_1d_array(0,local_buf_on_device);
  print_bw();

  if (rank == 0) printf("  Testing contiguous one-sided operations"
      " for doubles array\n  on device, local buffer on host\n");
  test_dbl_1d_array(1,0);
  print_bw();

  if (rank == 0) printf("  Testing contiguous one-sided operations"
      " for doubles array\n  on device, local buffer on device\n");
  test_dbl_1d_array(1,local_buf_on_device);
  print_bw();

  if (rank == 0) printf("  Testing contiguous one-sided operations"
      " for doubles array\n  on host, local buffer on host\n");
  test_dbl_1d_array(0,0);
  print_bw();

  if (rank == 0) printf("  Testing contiguous one-sided operations"
      " for doubles array\n  on host, local buffer on device\n");
  test_dbl_1d_array(0,local_buf_on_device);
  print_bw();

  if (rank == 0) printf("  Testing scatter/gather operations for double"
      " array\n  on device, local buffer on host\n");
  test_dbl_scatter(1,0);
  print_bw();

  if (rank == 0) printf("  Testing scatter/gather operations for double"
      " array\n  on device, local buffer on device\n");
  test_dbl_scatter(1,local_buf_on_device);
  print_bw();

  if (rank == 0) printf("  Testing scatter/gather operations for double"
      " array\n  on host, local buffer on host\n");
  test_dbl_scatter(0,0);
  print_bw();

  if (rank == 0) printf("  Testing scatter/gather operations for double"
      " array\n  on host, local buffer on device\n");
  test_dbl_scatter(0,local_buf_on_device);
  print_bw();

#endif
#if 1
  /* Check to see if code runs on subgroups */
  /* Don't run this test on an odd number of processors */
  if (nprocs%2 == 0) {
    int grp_size = nprocs/2;
    int grp, world;
    int *llist, *devIDs;
    int ndev, icnt;
    int me;
    if (rank == 0) printf("  Test  operations on subgroup\n");
    free(list);
    list = (int*)malloc(grp_size*sizeof(int));
    for (i=0; i<grp_size; i++) {
      if (rank < grp_size) {
        list[i] = i;
      } else {
        list[i] = i + grp_size;
      }
    }
    grp = GA_Pgroup_create(list, grp_size);
    world = GA_Pgroup_get_default();
    GA_Pgroup_set_default(grp);
    me = GA_Nodeid();
    printf("p[%d] rank: %d me: %d\n",wrank,rank,me);
    printf("p[%d] world: %d grp_size: %d\n",wrank,world,grp_size);
    llist = (int*)malloc(grp_size*sizeof(int));
    devIDs = (int*)malloc(grp_size*sizeof(int));
    GA_Device_host_list(llist, devIDs, &ndev, grp);
    printf("p[%d] ndev: %d\n",wrank,ndev);
    for (i=0; i<ndev; i++) {
      printf("p[%d] list[%d]: %d devIDs[%d]: %d\n",wrank,i,llist[i],i,devIDs[i]);
    }
    free(llist);
    free(devIDs);
    if (ndev == grp_size) {
      for (i=0; i<grp_size; i++) {
        list[i] = i;
      }
#if 1
      if (rank < grp_size) {
        if (me == 0) printf("  Testing operations on group 1\n");
#if 0
        if (me == 0) printf("  Group 1: Testing integer array on"
            " device, local buffer on host\n");
        test_int_array(1,0);
        print_bw();

        if (me == 0) printf("  Group 1: Testing integer array on"
            " device, local buffer on device\n");
        test_int_array(1,local_buf_on_device);
        print_bw();

#endif
        if (me == 0) printf("  Group 1: Testing integer array on"
            " host, local buffer on host\n");
        test_int_array(0,0);
        print_bw();
#if 0

        if (me == 0) printf("  Group 1: Testing integer array on"
            " host, local buffer on device\n");
        test_int_array(0,local_buf_on_device);
        print_bw();
#endif
      }
#endif
      fflush(stdout);
      GA_Pgroup_sync(world);
#if 1
      if (rank >= grp_size) {
        if (me == 0) printf("  Testing operations on group 2\n");
#if 0
        if (me == 0) printf("  Group 2: Testing integer array on"
            " device, local buffer on host\n");
        test_int_array(1,0);
        print_bw();

        if (me == 0) printf("  Group 2: Testing integer array on"
            " device, local buffer on device\n");
        test_int_array(1,local_buf_on_device);
        print_bw();

#endif
        if (me == 0) printf("  Group 2: Testing integer array on"
            " host, local buffer on host\n");
        test_int_array(0,0);
        print_bw();

#if 0
        if (me == 0) printf("  Group 2: Testing integer array on"
            " host, local buffer on device\n");
        test_int_array(0,local_buf_on_device);
        print_bw();
#endif
      }
#endif
    } else {
      printf("p[%d] Number of processors in group does not match"
          " number of devices. Ndev: %d Grp_size: %d\n",rank,ndev,grp_size);
    }
    GA_Pgroup_set_default(world);
    free(list);
  }
#endif


  t_tot = GA_Wtime()-tbeg;
  /* Print out timing stats */
  GA_Dgop(&t_put,1,"+");
  GA_Dgop(&t_get,1,"+");
  GA_Dgop(&t_acc,1,"+");
  GA_Dgop(&t_tot,1,"+");
  GA_Dgop(&t_sync,1,"+");
  GA_Dgop(&t_chk,1,"+");
  GA_Dgop(&t_rdinc,1,"+");
  GA_Dgop(&t_create,1,"+");
  GA_Dgop(&t_free,1,"+");
  GA_Dgop(&t_chk,1,"+");
  t_put /= ((double)nprocs);
  t_get /= ((double)nprocs);
  t_acc /= ((double)nprocs);
  t_tot /= ((double)nprocs);
  t_sync /= ((double)nprocs);
  t_chk /= ((double)nprocs);
  t_rdinc /= ((double)nprocs);
  t_create /= ((double)nprocs);
  t_free /= ((double)nprocs);
  t_sum = t_put + t_get + t_acc + t_sync + t_chk;
  t_sum = t_sum + t_rdinc + t_create + t_free;
  if (rank == 0) {
    printf("Total time in PUT:       %16.4e\n",t_put);
    printf("Total time in GET:       %16.4e\n",t_get);
    printf("Total time in ACC:       %16.4e\n",t_acc);
    printf("Total time in SYNC :     %16.4e\n",t_sync);
    printf("Total time in CHECK:     %16.4e\n",t_chk);
    printf("Total time in READ/INC:  %16.4e\n",t_rdinc);
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
