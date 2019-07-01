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

#include "ga.h"
#include "macdecls.h"
#include "mp3.h"

#define DIM 2
#define DIMSIZE 1024
#define SIZE DIMSIZE/2
#define MAX_FACTOR 256

void grid_factor(int p, int *idx, int *idy) {
  int i, j;                              //indices
  int ip;                                //comparison value to find factors/primes
  int ifac;                              //how many are in the factor array
  int pmax;                              //how many are in the prime array?
  int prime[MAX_FACTOR];                 //prime numbers between 1 and the square root of p
  int fac[MAX_FACTOR];                   //prime factors of p
  int ix;
  int iy;                                //used to find closest factor numbers
  int ichk;                              //is prime bool

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

  //preinitialized variables needed?
  char *transa;
  char *transb;
  void *alpha;
  void *beta;

  //GAs
  int g_a;
  int g_b;
  int g_c;

  int one = 1;
  int rank;
  int nprocs;   //number of processors

  int kdim;
  int i;
  int j;
  int k;
  int ipx;
  int ipy;
  int pdx;
  int pdy;
  int xdim;
  int ydim;
  int xbl;
  int ybl;
  int xcdim;
  int ycdim;
  int xnbl;
  int ynbl;
  int xcnt;
  int ycnt;
  int nb;

  int ind; 
  int istart;

  int local_test;
  int ldtest = 5;

  int *local_A = NULL;
  int **array_A = NULL;

  int *local_B = NULL;
  int **array_B = NULL;

  double delta_t;

  //initiate MPI/MA/GA
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MA_init(C_INT, 1000, 1000);
  GA_Initialize();

  nprocs = GA_Nnodes();  //number of processors
  rank = GA_Nodeid();     //processorID

  int test;
  int dimsize;

  for (test = 0; test <= 1; test++) {

    for (dimsize = 4; dimsize <=4096; dimsize *= 2){
      int *abuf1, *bbuf1, *cbuf1;
      int *ctest;

      int full_size = dimsize*dimsize;

      int alo[DIM] = {dimsize-dimsize,dimsize-dimsize};
      int ahi[DIM] = {dimsize-1,dimsize-1};
      int blo[DIM] = {dimsize-dimsize,dimsize-dimsize};
      int bhi[DIM] = {dimsize-1,dimsize-1};
      int clo[DIM] = {dimsize-dimsize,dimsize-dimsize};
      int chi[DIM] = {dimsize-1,dimsize-1};

      //hi los
      int nlo;
      int loC[DIM] = {dimsize-dimsize,dimsize-dimsize};
      int hiC[DIM] = {dimsize-1,dimsize-1};
      int loA[DIM] = {dimsize-dimsize,dimsize-dimsize};
      int hiA[DIM] = {dimsize-1,dimsize-1};
      int loB[DIM] = {dimsize-dimsize,dimsize-dimsize};
      int hiB[DIM] = {dimsize-1,dimsize-1};

      int dims[DIM]={dimsize,dimsize};

      //leading dimensions

      int ldb = hiB[0]-loB[0]+1;
      int ldc = hiC[0]-loC[0]+1;
      int ldC;
      int lda = hiA[0]-loA[0]+1;

      local_A=(int*)malloc(full_size*sizeof(int));
      array_A=(int**)malloc(DIM*sizeof(int*));

      for (i=0;i<full_size;i++){
        *(local_A + i) = i;
      }

      local_B=(int*)malloc(full_size*sizeof(int));
      array_B=(int**)malloc(DIM*sizeof(int*));

      for (i=0; i<full_size; i++){
        *(local_B + i) = i;
      }

      g_a = NGA_Create(C_INT, DIM, dims, "array_A", NULL);
      g_b = NGA_Create(C_INT, DIM, dims, "array_B", NULL);
      g_c = NGA_Create(C_INT, DIM, dims, "array_C", NULL);

      //fill GAs a and b with values to be multipled

      GA_Zero(g_a);
      GA_Zero(g_b);
      GA_Zero(g_c);

      //GA_Print(g_a);

      //fprintf(stderr, "rank[%d] PUTBUFA loA[0]:%d,loA[1]:%d,hiA[0]:%d,hiA[1]:%d, ld:%d\n",rank, loA[0],loA[1],hiA[0],hiA[1], lda);

      NGA_Put(g_a, alo, ahi, local_A, &lda);
      NGA_Put(g_b, blo, bhi, local_B, &ldb);

      GA_Sync();

      GA_Zero(g_c);

      char* read_only = "read_only";
      char* read_cache = "read_cache";

      /*
      if (test == 0) {
      NGA_Set_property(g_a, read_only);
      NGA_Set_property(g_b, read_only);
      }
      */

      if (test == 1) {
        NGA_Set_property(g_a, read_cache);
        NGA_Set_property(g_b, read_cache);
      }

      GA_Sync();
      //GA_Print(g_a);
      //GA_Print(g_b);

      grid_factor(nprocs, &pdx, &pdy);

      //coordinates of processor for grid
      ipx = rank%pdx;
      ipy = (rank-ipx)/pdx; 

      xdim = dimsize;
      ydim = dimsize;

      if (dimsize <= 64) {
        xbl = dimsize/2;
        ybl = dimsize/2;
      } else {
        xbl = dimsize/2;
        ybl = dimsize/2;
      }

      //total number of blocks on each dimension
      xnbl = xdim/xbl;
      ynbl = ydim/ybl;

      if ((xnbl * xbl) < xdim)
        xnbl++;

      if ((ynbl * ybl) < ydim)
        ynbl++;

      xcnt = ipx;
      ycnt = ipy;

      GA_Sync();

      delta_t = GA_Wtime();

      /* 
         if (test == 0)
         { 
         if (full_size > 16) delta_t -= (.00000002*full_size);
         }
         */

      while (ycnt < ynbl) {
        int num_blocks;
        int offset;
        int elemsize;
        int ld;
        int *a_buf = NULL;
        int *b_buf = NULL;
        int *c_buf = NULL;
        int test_buf;
        int size_a;
        int size_b; 
        int size_c;

        loC[0] = xcnt * xbl;
        loC[1] = ycnt * ybl;

        hiC[0] = (xcnt + 1) * xbl - 1;
        hiC[1] = (ycnt + 1) * ybl - 1;

        if (hiC[0] >= xdim) hiC[0] = xdim - 1;

        if (hiC[0] >= ydim) hiC[0] = ydim - 1;

        // Calculating number of blocks for inner dimension
        num_blocks = dimsize/(hiC[0]-loC[0]+1);
        if (num_blocks*(hiC[0]-loC[0]+1) < dimsize) num_blocks++;

        //set up buffers
        offset = 0;
        //elemsize = GA_sizeofM(atype);
        elemsize = sizeof(int);

        c_buf = (void*)malloc((hiC[0]-loC[0]+1)*(hiC[1]-loC[1]+1)*elemsize);
        a_buf = (void*)malloc((hiC[0]-loC[0]+1)*(hiC[1]-loC[1]+1)*elemsize);
        b_buf = (void*)malloc((hiC[0]-loC[0]+1)*(hiC[1]-loC[1]+1)*elemsize);

        test_buf = (hiC[0]-loC[0]+1)*(hiC[1]-loC[1]+1)*elemsize;

        size_c = (hiC[0]-loC[0]+1)*(hiC[1]-loC[1]+1);

        ldC = hiC[1]-loC[1]+1;

        // calculate starting block index
        istart = (loC[0]-clo[0])/(hiC[0]-loC[0]+1);

        // loop over block pairs
        for (nb=0; nb<num_blocks; nb++) {

          ind = istart + nb;
          ind = ind%num_blocks;

          nlo = alo[1]+ind*(hiC[0]-loC[0]+1);
          loA[0] = loC[0];
          hiA[0] = hiC[0];
          loA[1] = nlo;
          hiA[1] = loA[1]+(hiC[0]-loC[0]);
          if (hiA[1] > ahi[1]) hiA[1] = ahi[1];
          ld = hiA[0]-loA[0]+1;
          size_a = (hiA[0]-loA[0]+1)*(hiA[1]-loA[1]+1);

          //fprintf(stderr, "rank[%d] GETBUFA loA[0]:%d,loA[1]:%d,hiA[0]:%d,hiA[1]:%d, ld:%d\n",rank, loA[0],loA[1],hiA[0],hiA[1], ld);

          NGA_Get(g_a,loA,hiA,a_buf,&ld);       

          loB[1] = loC[1];
          hiB[1] = hiC[1];
          nlo = blo[0]+ind*(hiC[0]-loC[0]+1);
          loB[0] = nlo;
          hiB[0] = loB[0]+(hiC[0]-loC[0]);
          if (hiB[0] > bhi[0]) hiB[0] = bhi[0];
          ld = hiB[0]-loB[0]+1;
          size_b = (hiB[0]-loB[0]+1)*(hiB[1]-loB[1]+1);        

          //fprintf(stderr, "rank[%d] GETBUFB loA[0]:%d,loA[1]:%d,hiA[0]:%d,hiA[1]:%d, ld:%d\n",rank, loA[0],loA[1],hiA[0],hiA[1], ld);

          NGA_Get(g_b,loB,hiB,b_buf,&ld);

          //clean-up
          xcdim = hiC[0] - loC[0] + 1;
          ycdim = hiC[1] - loC[1] + 1;

          for (i = 0; i < xcdim; i++) {
            for (j = 0; j < ycdim; j++) {
              c_buf[i*ycdim+j] = 0.0;
            }
          }

          //transpose B to reduce page faults

          kdim = hiA[1] - loA[1] + 1;

          for (i = 0; i < xcdim; i++) {
            for (j = 0; j < ycdim; j++) {
              for (k = 0; k < kdim; k++) {
                c_buf[i*ycdim+j] += a_buf[i*kdim+k]*b_buf[k*ycdim+j];
              }
            }
          }

          NGA_Acc(g_c, loC, hiC, c_buf, &ldC, &one);

        }

        // multiplication is done, free buffers 
        free(a_buf);
        free(b_buf);
        free(c_buf);


        xcnt += pdx;

        if (xcnt >= xnbl) {
          xcnt = ipx;
          ycnt += pdy;
        }
      }

      delta_t = GA_Wtime()-delta_t;      

      if (dimsize == 0 & rank == 0) printf("\n"); 
      if (test == 0 && rank == 0) printf("READ  - DIMSIZE: %5d  Time (us): %7.4f\n",dimsize,delta_t*1.0e6);
      if (test == 1 && rank == 0) printf("CACHE - DIMSIZE: %5d  Time (us): %7.4f\n",dimsize,delta_t*1.0e6);

      //GA_Sync();

      //if (dimsize == 4) GA_Print(g_c);

      GA_Sync();

      NGA_Unset_property(g_a);
      NGA_Unset_property(g_b);


      /* check multipy for correctness */
      abuf = (int*)malloc(sizeof(int)*dimsize*dimsize);
      bbuf = (int*)malloc(sizeof(int)*dimsize*dimsize);
      cbuf = (int*)malloc(sizeof(int)*dimsize*dimsize);
      ctest = (int*)malloc(sizeof(int)*dimsize*dimsize);
      /* get data from matrices a, b, c */
      NGA_Get(g_a,alo,ahi,abuf,&dimsize);
      NGA_Get(g_b,blo,bhi,bbuf,&dimsize);
      NGA_Get(g_c,clo,chi,ctest,&dimsize);
      for (i=0; i<dimsize; i++) {
        for (j=0; j<dimsize; j++) {
          cbuf[i*dimsize+j] = 0;
          for (k=0; k<dimsize; k++) {
            cbuf[i*dimsize+j] += abuf[i*dimsize+k]*bbuf[k*dimsize+j];
          }
          if (cbuf[i*dimsize+j] != ctest[i*dimsize+j] && rank == 0) {
            printf("mismatch for pair [%d,%d] expected: %d actual: %d\n",i,j,
                cbuf[i*dimsize+j],ctest[i*dimsize+j]);
          }
        }
      }
      free(abuf);
      free(bbuf);
      free(cbuf);
      free(ctest);

      GA_Destroy(g_a);
      GA_Destroy(g_b);
      GA_Destroy(g_c);

      free(local_A);
      free(array_A);
      free(local_B);     
      free(array_B);
    }
  }

  GA_Terminate();

  MPI_Finalize();

  //fprintf(stderr,"rank[%d] complete\n",rank);   
}
