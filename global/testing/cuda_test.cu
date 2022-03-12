#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*#include "cuda.h"*/
#include "mpi.h"
#include "ga.h"
#include "macdecls.h"

#include <cuda_runtime.h>

#define DIM 2
#define DIMSIZE 1024
#define SIZE DIMSIZE/2
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
  int *buf, *hbuf;
  int ld;
  int ok;

  MPI_Init(&argc, &argv);


  GA_Initialize();

  nprocs = GA_Nnodes();  
  rank = GA_Nodeid();   

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
  if (ipx+1<pdx-1) {
    hi[0] = (ipx+1)*xinc-1;
  } else {
    hi[0] = DIMSIZE-1;
  }
  if (ipy+1<pdy-1) {
    hi[1] = (ipy+1)*yinc-1;
  } else {
    hi[1] = DIMSIZE-1;
  }

  /* create a global array and initialize it to zero */
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_INT);
  NGA_Allocate(g_a);
  GA_Zero(g_a);

  /* discover how many GPU's I can see */
  int ngpus;
  cudaGetDeviceCount(&ngpus);
  printf("Process %d can see %d GPUs\n",rank,ngpus);
  cudaSetDevice(rank);
  /* allocate data on GPU */
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  cudaMallocManaged((int**)&buf, nsize*sizeof(int));
  //cudaMalloc((int**)&buf, nsize*sizeof(int));
  printf("p[%d] completed cudaMalloc operation\n",rank);
  ld = (hi[1]-lo[1]+1);


  /* allocate a local buffer and initialize it with values*/
  hbuf = (int*)malloc(nsize*sizeof(int));
  for (ii = lo[0]; ii<hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      hbuf[idx] = ii*DIMSIZE+jj;
    }
  }
  cudaMemcpy(buf, hbuf, nsize*sizeof(int), cudaMemcpyHostToDevice);
  printf("p[%d] completed cudaMemcpy operation\n",rank);

  /* copy data to global array */
  NGA_Put(g_a, lo, hi, buf, &ld);
  printf("p[%d] completed Put operation\n",rank);
  GA_Sync();

  /* zero out local buffers on host and device */
  for (i=0; i<nsize; i++) hbuf[i] = 0;
  cudaMemcpy(buf, hbuf, nsize*sizeof(int), cudaMemcpyHostToDevice);

  /* copy data from global array to local buffer */
  NGA_Get(g_a, lo, hi, buf, &ld);
  cudaMemcpy(hbuf, buf, nsize*sizeof(int), cudaMemcpyDeviceToHost);

  ok = 1;
  for (ii = lo[0]; ii<hi[0]; ii++) {
    i = ii-lo[0];
    for (jj = lo[1]; jj<hi[1]; jj++) {
      j = jj-lo[1];
      idx = i*ld+j;
      if (hbuf[idx] != ii*DIMSIZE+jj) ok = 0;
    }
  }
  if (!ok) {
    printf("Mismatch found on process %d\n",rank);
  } else {
    if (rank == 0) printf("Put is okay\n");
  }

  cudaFree(buf);
  free(hbuf);
  GA_Destroy(g_a);
  GA_Terminate();
  MPI_Finalize();
}
