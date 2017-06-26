#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include "mpi.h"
#include "ga.h"
#if defined(_OPENMP)
#include "omp.h"
#endif

#define MEDIUM_MESSAGE_SIZE 8192
#define ITER_SMALL 100
#define ITER_LARGE 10

#define WARMUP 2

#define DEFAULT_DIM 1024
#define MAX_MESSAGE_SIZE DEFAULT_DIM*DEFAULT_DIM

#define MAX_FACTOR 256
/**
 * Factor p processors into 2D processor grid of dimensions px, py
 */
void grid_factor(int p, int *idx, int *idy) {
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
 *    find two factors of p of approximately the
 *    same size
 */
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

/* Convenience function to check that something is true on all processors */
int trueEverywhere(int flag)
{
  int tflag, nprocs;
  if (tflag) tflag = 1;
  else tflag = 0;
  nprocs = GA_Nnodes();
  GA_Igop(&tflag,1,"+");
  if (nprocs == tflag) return 1;
  return 0;
}

/* Function to print out timing statistics */
void printTimes(double *time, int *ntime, int *nelems, int size, int nthread)
{
  int me = GA_Nodeid();
  int nproc = GA_Nnodes();
  double l_time;
  int l_ntime;
  int l_nelems;
  int one = 1;
  double bandwdth;
  double optime;
  int i;

  l_time = 0.0;
  l_ntime = 0;
  l_nelems = 0;
  for (i=0; i<nthread; i++) {
    l_time += time[i];
    l_ntime += ntime[i];
    l_nelems += nelems[i];
  }
  GA_Dgop(&l_time,one,"+");
  GA_Igop(&l_ntime,one,"+");
  GA_Igop(&l_nelems,one,"+");
  l_nelems *= sizeof(int);
  bandwdth = ((double)l_nelems)/l_time;
  bandwdth /= 1.0e6;
  optime = (l_time)/((double)l_ntime);
  if (me==0) {
    printf("         %7d       %12.6f       %12.6f   %d\n",
      size,optime,bandwdth,nthread);
  }
}

int main(int argc, char * argv[])
{
#if defined(_OPENMP)
    int x = DEFAULT_DIM;
    int y = DEFAULT_DIM;
    int block_x, block_y;
    int g_array, g_count;
    int me, nproc;
    int px, py, ipx, ipy;
    int glo[2], ghi[2], gld[2];
    int tx, ty;
    int i,j,icnt;
    int return_code = 0;
    int dims[2];
    int ndimx = 2;
    int thread_count = 4;
    int ok;
    int zero = 0, one = 1;
    double rone = 1.0;
    int *ptr;
    int next, nextx, nexty;
    char *env_threads;
    int provided;
    double *time, *ritime;
    int *ntime, *nelems, *rinc;
    int msg_size, ithread, iter;
    int buf_size = MAX_MESSAGE_SIZE;
    int ncount;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    GA_Initialize();

    nproc = GA_Nnodes();
    me = GA_Nodeid();
    if (provided < MPI_THREAD_MULTIPLE && me==0) {
      printf("\nMPI_THREAD_MULTIPLE not provided\n");
    }

    /* Find processor grid dimensions and processor grid coordinates */
    grid_factor(nproc, &px, &py);
    ipy = me%py;
    ipx = (me-ipy)/py;
    if (me==0) {
      printf("\nTest running of %d processors\n",nproc);
      printf("\n  Array dimension is %d X %d\n",x,y);
      printf("\n  Processor grid is %d X %d\n\n",px,py);
    }

    dims[0] = x;
    dims[1] = y;
  
    /* Create GA and set all elements to zero */
    g_array = NGA_Create(C_INT, 2, dims, "source", NULL);
    g_count = NGA_Create(C_INT, 1, &one, "counter", NULL);
    GA_Zero(g_array);

    if(env_threads = getenv("OMP_NUM_THREADS"))
        thread_count = atoi(env_threads);
    else
        omp_set_num_threads(thread_count);

    if (thread_count > 8) thread_count = 8;

    if (me==0) {
            printf("\n[%d]Testing %d threads.\n", me, thread_count);

            printf("\n[%d]Testing write1 from 0.\n", me);
    }
    time = (double*)malloc(thread_count*sizeof(double));
    ritime = (double*)malloc(thread_count*sizeof(double));
    ntime = (int*)malloc(thread_count*sizeof(int));
    nelems = (int*)malloc(thread_count*sizeof(int));
    rinc = (int*)malloc(thread_count*sizeof(int));
    for (i=0; i<thread_count; i++) {
      time[i] = 0.0;
      ritime[i] = 0.0;
      ntime[i] = 0;
      nelems[i] = 0;
      rinc[i] = 0;
    }

    if (me==0) {
       printf("\nPerformance of GA_Put\n");
       printf("msg size (bytes)    avg time (sec)    avg b/w (MB/sec) N threads\n");
    }

    for (msg_size = 1; msg_size <= buf_size; msg_size *= 2) {
      for (block_x=1; block_x <= DEFAULT_DIM; block_x *= 2) {
        for (ithread = 1; ithread<= thread_count; ithread++) {
          for (i=0; i<ithread; i++) {
            time[i] = 0.0;
            ritime[i] = 0.0;
            ntime[i] = 0;
            nelems[i] = 0;
            rinc[i] = 0;
          }
          block_y = msg_size/block_x;
          /* Fill global array with data by having each thread write
           * blocks to it */
          #pragma omp parallel num_threads(ithread)
          {
            /* declare variables local to each thread */
            int lo[2], hi[2], tlo[2], thi[2];
            int ld[2];
            int k, m, n;
            int xinc, yinc;
            int itx, ity;
            int offset;
            int *buf;
            int lld;
            long task, inc; 
            int id;
            double delta_t;
            int bsize;
            id = omp_get_thread_num();
            inc = 1;
            delta_t = GA_Wtime();
            task = NGA_Read_inc(g_count, &zero, inc);
            delta_t = GA_Wtime()-delta_t;
            ritime[id] += delta_t;
            rinc[id] += one;
            buf = (int*)malloc(block_x*block_y*sizeof(int));
            while (task < tx*ty) {
              ity = task%ty;
              itx = (task-ity)/ty;
              tlo[0] = itx*block_x;
              tlo[1] = ity*block_y;
              /*
                 printf("j: %d k: %d tlo[0]: %d tlo[1]: %d xinc: %d yinc: %d\n",
                 j,k,tlo[0],tlo[1],xinc,yinc);
               */
              thi[0] = tlo[0] + block_x - 1;
              if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
              thi[1] = tlo[1] + block_y - 1;
              if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
              lld = thi[1]-tlo[1]+1;
              bsize = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);

              /* Fill a portion of local buffer with correct values */
              for (m=tlo[0]; m<=thi[0]; m++) {
                for (n=tlo[1]; n<=thi[1]; n++) {
                  offset = (m-tlo[0])*lld + (n-tlo[1]);
                  buf[offset] = m*dims[1]+n;
                }
              }
              delta_t = GA_Wtime();
              NGA_Put(g_array, tlo, thi, buf, &lld);
              delta_t = GA_Wtime()-delta_t;
              time[id] += delta_t;
              ntime[id] += one;
              nelems[id] += bsize;
              delta_t = GA_Wtime();
              task = NGA_Read_inc(g_count, &zero, inc);
              delta_t = GA_Wtime()-delta_t;
              ritime[id] += delta_t;
              rinc[id] += one;
            }
            free(buf);
          }
          printTimes(time,ntime,nelems,block_x*block_y,ithread);
          /* Sync all processors at end of initialization loop */
          NGA_Sync(); 
        }
      }
    }

    /* Each process determines if it is holding the correct data */
    NGA_Distribution(g_array,me,glo,ghi);
    NGA_Access(g_array,glo,ghi,&ptr,gld);
    ok = 1;
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        if (ptr[icnt] != i*dims[1]+j) {
          ok = 0;
          printf("p[%d] (write1) mismatch at point [%d,%d] actual: %d expected: %d\n",
              me,i,j,ptr[icnt],i*dims[1]+j);
        }
        icnt++;
      }
    }
    NGA_Release(g_array,glo,ghi);
    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nwrite1 test OK\n");
    } else if (me == 0 && !ok) {
      printf("\nwrite1 test failed\n");
    }

    /* Move data from remote processor to local buffer */
    if (me==0) {
      printf("\n[%d]Testing read1 from 0.\n", me);
    }

#if 0
    /* Threads grab data from global array and copy them into a local
     * buffer and verify that data is correct. */
    GA_Zero(g_count);
    ok = 1;
    GA_Zero(g_time);
    GA_Zero(g_ntime);
    GA_Zero(g_elems);
    #pragma omp parallel num_threads(thread_count)
    {
      /* declare variables local to each thread */
      int lo[2], hi[2], tlo[2], thi[2];
      int ld[2];
      int k, m, n;
      int itx, ity;
      int offset;
      int *buf;
      int lld;
      int id;
      long task, inc; 
      double delta_t;
      int bsize;
      inc = 1;
      id = omp_get_thread_num();
      buf = (int*)malloc(block_x*block_y*sizeof(int));
      delta_t = GA_Wtime();
      task = NGA_Read_inc(g_count, &zero, inc);
      delta_t = GA_Wtime()-delta_t;
      NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
      NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      while (task < tx*ty) {
        ity = task%ty;
        itx = (task-ity)/ty;
        tlo[0] = itx*block_x;
        tlo[1] = ity*block_y;
        thi[0] = tlo[0] + block_x - 1;
        if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
        thi[1] = tlo[1] + block_y - 1;
        if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
        lld = thi[1]-tlo[1]+1;
        bsize = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
        delta_t = GA_Wtime();
        NGA_Get(g_array, tlo, thi, buf, &lld);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_time,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_ntime,&me,&me,&one,&one,&one);
        NGA_Acc(g_elems,&me,&me,&bsize,&one,&one);

        /* check that values in buffer are correct */
        for (m=tlo[0]; m<=thi[0]; m++) {
          for (n=tlo[1]; n<=thi[1]; n++) {
            offset = (m-tlo[0])*lld + (n-tlo[1]);
            if (buf[offset] != m*dims[1]+n) {
              ok = 0;
              printf("Read mismatch for [%d,%d] expected: %d actual: %d\n",
                  m,n,m*dims[1]+n,lld);
            }
          }
        }
        delta_t = GA_Wtime();
        task = NGA_Read_inc(g_count, &zero, inc);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      }
      free(buf);
    }
    /* Sync all processors at end of initialization loop */
    NGA_Sync(); 

    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nread1 test OK\n");
    } else if (me==0 && !ok) {
      printf("\nread1 test failed\n");
    }
    printTimes(g_time, g_ntime, g_elems, "Get performance");

    GA_Zero(g_count);

    /* Accumulate data to global array */
    if (me==0) {
      printf("\n[%d]Testing acc1 from 0.\n", me);
    }

    GA_Zero(g_time);
    GA_Zero(g_ntime);
    GA_Zero(g_elems);
    #pragma omp parallel num_threads(thread_count)
    {
      /* declare variables local to each thread */
      int lo[2], hi[2], tlo[2], thi[2];
      int ld[2];
      int k, m, n;
      int xinc, yinc;
      int itx, ity;
      int offset;
      int *buf;
      int lld;
      long task, inc; 
      int id;
      double delta_t;
      int bsize;
      id = omp_get_thread_num();
      inc = 1;
      delta_t = GA_Wtime();
      task = NGA_Read_inc(g_count, &zero, inc);
      delta_t = GA_Wtime()-delta_t;
      NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
      NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      buf = (int*)malloc(block_x*block_y*sizeof(int));
      while (task < tx*ty) {
        ity = task%ty;
        itx = (task-ity)/ty;
        tlo[0] = itx*block_x;
        tlo[1] = ity*block_y;
        thi[0] = tlo[0] + block_x - 1;
        if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
        thi[1] = tlo[1] + block_y - 1;
        if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
        lld = thi[1]-tlo[1]+1;
        bsize = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);

        /* Accumulate values to a portion of global array */
        for (m=tlo[0]; m<=thi[0]; m++) {
          for (n=tlo[1]; n<=thi[1]; n++) {
            offset = (m-tlo[0])*lld + (n-tlo[1]);
            buf[offset] = m*dims[1]+n;
          }
        }
        delta_t = GA_Wtime();
        NGA_Acc(g_array, tlo, thi, buf, &lld, &one);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_time,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_ntime,&me,&me,&one,&one,&one);
        NGA_Acc(g_elems,&me,&me,&bsize,&one,&one);
        delta_t = GA_Wtime();
        task = NGA_Read_inc(g_count, &zero, inc);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      }
      free(buf);
    }
    /* Sync all processors at end of initialization loop */
    NGA_Sync(); 

    /* Check local buffer for correct values */
    NGA_Distribution(g_array,me,glo,ghi);
    ok = 1;
    NGA_Access(g_array,glo,ghi,&ptr,gld);
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        if (ptr[icnt] != 2*(i*dims[1]+j)) {
          ok = 0;
          printf("p[%d] (Acc1) mismatch at point [%d,%d] actual: %d expected: %d\n",
              me,i,j,ptr[icnt],i*dims[1]+j);
        }
        icnt++;
      }
    }
    NGA_Release(g_array,glo,ghi);
    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nacc1 test OK\n");
    } else if (me==0 && !ok) {
      printf("\nacc1 test failed\n");
    }
    printTimes(g_time, g_ntime, g_elems, "Accumulate performance");
    
    /* Sync all processors*/
    NGA_Sync(); 
    /* Testing random work pattern */
    if (me==0) {
      printf("\n[%d]Testing ran1 from 0.\n", me);
    }

    /* Reinitialize source array */
    NGA_Distribution(g_array,me,glo,ghi);
    NGA_Access(g_array,glo,ghi,&ptr,gld);
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        ptr[icnt] = i*dims[1]+j;
        icnt++;
      }
    }
    NGA_Release(g_array,glo,ghi);

    /* Mimic random work. */
    GA_Zero(g_dest);
    GA_Zero(g_count);

    #pragma omp parallel num_threads(thread_count)
    {
      /* declare variables local to each thread */
      int tlo[2], thi[2];
      int ld[2];
      int k, m, n;
      int xinc, yinc;
      int itx, ity;
      int offset, icnt;
      int *buf, *buft;
      int lld;
      long task, inc; 
      int id;
      int tmp;
      double delta_t;
      id = omp_get_thread_num();
      inc = 1;
      delta_t = GA_Wtime();
      task = NGA_Read_inc(g_count, &zero, inc);
      delta_t = GA_Wtime()-delta_t;
      NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
      NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      buf = (int*)malloc(block_x*block_y*sizeof(int));
      buft = (int*)malloc(block_x*block_y*sizeof(int));
      /* Read and transpose data */
      while (task < 2*tx*ty) {
        k = task;
        if (k>=tx*ty) k -= tx*ty;
        ity = k%ty;
        itx = (k-ity)/ty;
        tlo[0] = itx*block_x;
        tlo[1] = ity*block_y;
        thi[0] = tlo[0] + block_x - 1;
        if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
        thi[1] = tlo[1] + block_y - 1;
        if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
        ld[0] = thi[0]-tlo[0]+1;
        ld[1] = thi[1]-tlo[1]+1;
        lld = thi[1]-tlo[1]+1;

        /* Get data from g_array */
        NGA_Get(g_array, tlo, thi, buf, &lld);

        /* Evaluate transpose of local bloack */
        icnt = 0;
        for (m=0; m<ld[0]; m++) {
          for (n=0; n<ld[1]; n++) {
            offset = n*ld[0]+m;
            buft[offset] = buf[icnt]; 
            icnt++;
          }
        }
        
        /* Find transposed block location */
        tmp = ity;
        ity = itx;
        itx = tmp;
        tlo[0] = itx*block_y;
        tlo[1] = ity*block_x;
        thi[0] = tlo[0] + block_y - 1;
        if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
        thi[1] = tlo[1] + block_x - 1;
        if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
        lld = thi[1]-tlo[1]+1;
        NGA_Acc(g_dest, tlo, thi, buft, &lld, &one);
        delta_t = GA_Wtime();
        task = NGA_Read_inc(g_count, &zero, inc);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      }
      free(buf);
      free(buft);
    }
    GA_Sync();

    /* Check local buffer for correct values */
    NGA_Distribution(g_dest,me,glo,ghi);
    ok = 1;
    NGA_Access(g_dest,glo,ghi,&ptr,gld);
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        if (ptr[icnt] != 2*(j*dims[0]+i)) {
          ok = 0;
          printf("p[%d] (Ran1) mismatch at point [%d,%d] actual: %d expected: %d\n",
              me,i,j,ptr[icnt],2*(j*dims[0]+i));
        }
        icnt++;
      }
    }
    NGA_Release(g_dest,glo,ghi);
    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nran1 test OK\n");
    } else if (me==0 && !ok) {
      printf("\nran1 test failed\n");
    }
    GA_Zero(g_array);
    GA_Zero(g_dest);
    /* Reinitialize source array */
    NGA_Distribution(g_array,me,glo,ghi);
    NGA_Access(g_array,glo,ghi,&ptr,gld);
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        ptr[icnt] = i*dims[1]+j;
        icnt++;
      }
    }
    NGA_Release(g_array,glo,ghi);
    if (me==0) {
      printf("\n[%d]Testing non-blocking-write1 from 0.\n", me);
    }

    /* Test nonblocking operations */
    /* Create array visible to all threads that can hold non-blocking
     * handles */
    GA_Zero(g_count);
    ok = 1;
    GA_Zero(g_array); 
    /* Fill global array with data by having each thread write
     * blocks to it */
    GA_Zero(g_time);
    GA_Zero(g_ntime);
    GA_Zero(g_elems);
    #pragma omp parallel num_threads(thread_count)
    {
      /* declare variables local to each thread */
      int lo[2], hi[2], tlo[2], thi[2];
      int ld[2];
      int k, m, n;
      int xinc, yinc;
      int itx, ity;
      int offset;
      int *buf;
      int lld;
      long task, inc; 
      int id;
      double delta_t;
      int bsize;
      int icnt = 0;
      ga_nbhdl_t* nb_hdl = (ga_nbhdl_t*) malloc(tx*ty*sizeof(ga_nbhdl_t));
      id = omp_get_thread_num();
      inc = 1;
      delta_t = GA_Wtime();
      task = NGA_Read_inc(g_count, &zero, inc);
      delta_t = GA_Wtime()-delta_t;
      NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
      NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      buf = (int*)malloc(block_x*block_y*sizeof(int));
      while (task < tx*ty) {
        ity = task%ty;
        itx = (task-ity)/ty;
        tlo[0] = itx*block_x;
        tlo[1] = ity*block_y;
        /*
        printf("j: %d k: %d tlo[0]: %d tlo[1]: %d xinc: %d yinc: %d\n",
        j,k,tlo[0],tlo[1],xinc,yinc);
        */
        thi[0] = tlo[0] + block_x - 1;
        if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
        thi[1] = tlo[1] + block_y - 1;
        if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
        lld = thi[1]-tlo[1]+1;
        bsize = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);

        /* Fill a portion of local buffer with correct values */
        for (m=tlo[0]; m<=thi[0]; m++) {
          for (n=tlo[1]; n<=thi[1]; n++) {
            offset = (m-tlo[0])*lld + (n-tlo[1]);
            buf[offset] = m*dims[1]+n;
          }
        }
        delta_t = GA_Wtime();
        NGA_NbPut(g_array, tlo, thi, buf, &lld, &nb_hdl[icnt]);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_time,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_ntime,&me,&me,&one,&one,&one);
        NGA_Acc(g_elems,&me,&me,&bsize,&one,&one);
        icnt++;
        /*
        NGA_NbWait(putid);
        */
        delta_t = GA_Wtime();
        task = NGA_Read_inc(g_count, &zero, inc);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      }
      free(buf);
      free(nb_hdl);
      /* Call wait on all outstanding tasks. Don't bother to reinitialize
         global counter */
      for (k=0; k<icnt; k++) {
        delta_t = GA_Wtime();
        NGA_NbWait(&nb_hdl[k]);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_time,&me,&me,&delta_t,&one,&rone);
        delta_t = GA_Wtime();
        task = NGA_Read_inc(g_count, &zero, inc);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      }
    }
    /* Sync all processors at end of initialization loop */
    NGA_Sync(); 

    /* Each process determines if it is holding the correct data */
    NGA_Distribution(g_array,me,glo,ghi);
    NGA_Access(g_array,glo,ghi,&ptr,gld);
    ok = 1;
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        if (ptr[icnt] != i*dims[1]+j) {
          ok = 0;
          printf("p[%d] (write1) mismatch at point [%d,%d] actual: %d expected: %d\n",
              me,i,j,ptr[icnt],i*dims[1]+j);
        }
        icnt++;
      }
    }
    NGA_Release(g_array,glo,ghi);
    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nnon-blocking-write1 test OK\n");
    } else if (me==0 && !ok) {
      printf("\nnon-blocking-write1 test failed\n");
    }
    printTimes(g_time, g_ntime, g_elems, "Non-blocking put performance");

    /* Move data from remote processor to local buffer */
   if (me==0) {
      printf("\n[%d]Testing non-blocking-read1 from 0.\n", me);
    }

    /* Threads grab data from global array and copy them into a local
     * buffer and verify that data is correct. */
    GA_Zero(g_count);
    ok = 1;
    GA_Zero(g_time);
    GA_Zero(g_ntime);
    GA_Zero(g_elems);
    #pragma omp parallel num_threads(thread_count)
    {
      /* declare variables local to each thread */
      int lo[2], hi[2], tlo[2], thi[2];
      int ld[2];
      int k, m, n;
      int itx, ity;
      int offset;
      int *buf;
      int lld;
      int id;
      long task, inc; 
      double delta_t;
      int bsize;
      inc = 1;
      ga_nbhdl_t* getid=(ga_nbhdl_t*)malloc(sizeof(ga_nbhdl_t));
      id = omp_get_thread_num();
      buf = (int*)malloc(block_x*block_y*sizeof(int));
      delta_t = GA_Wtime();
      task = NGA_Read_inc(g_count, &zero, inc);
      delta_t = GA_Wtime()-delta_t;
      NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
      NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      while (task < tx*ty) {
        ity = task%ty;
        itx = (task-ity)/ty;
        tlo[0] = itx*block_x;
        tlo[1] = ity*block_y;
        thi[0] = tlo[0] + block_x - 1;
        if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
        thi[1] = tlo[1] + block_y - 1;
        if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
        lld = thi[1]-tlo[1]+1;
        bsize = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);
        delta_t = GA_Wtime();
        NGA_NbGet(g_array, tlo, thi, buf, &lld,getid);
        NGA_NbWait(getid);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_time,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_ntime,&me,&me,&one,&one,&one);
        NGA_Acc(g_elems,&me,&me,&bsize,&one,&one);

        /* check that values in buffer are correct */
        for (m=tlo[0]; m<=thi[0]; m++) {
          for (n=tlo[1]; n<=thi[1]; n++) {
            offset = (m-tlo[0])*lld + (n-tlo[1]);
            if (buf[offset] != m*dims[1]+n) {
              ok = 0;
              printf("Read mismatch for [%d,%d] expected: %d actual: %d\n",
                  m,n,m*dims[1]+n,lld);
            }
          }
        }
        delta_t = GA_Wtime();
        task = NGA_Read_inc(g_count, &zero, inc);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      }
      free(buf);
      free(getid);
    }
    /* Sync all processors at end of initialization loop */
    NGA_Sync(); 
    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nnon-blocking-read1 test OK\n");
    } else if (me==0 && !ok) {
      printf("\nnon-blocking-read1 test failed\n");
    }
    printTimes(g_time, g_ntime, g_elems, "Non-blocking get performance");

    GA_Zero(g_count);
    if (me==0) {
      printf("\n[%d]Testing non-blocking-acc1 from 0.\n", me);
    }

    GA_Zero(g_time);
    GA_Zero(g_ntime);
    GA_Zero(g_elems);
    #pragma omp parallel num_threads(thread_count)
    {
      /* declare variables local to each thread */
      int lo[2], hi[2], tlo[2], thi[2];
      int ld[2];
      int k, m, n;
      int xinc, yinc;
      int itx, ity;
      int offset;
      int *buf;
      int lld;
      long task, inc; 
      int id;
      double delta_t;
      int bsize;
      ga_nbhdl_t* accid = (ga_nbhdl_t*) malloc(sizeof(ga_nbhdl_t*));
      id = omp_get_thread_num();
      inc = 1;
      delta_t = GA_Wtime();
      task = NGA_Read_inc(g_count, &zero, inc);
      delta_t = GA_Wtime()-delta_t;
      NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
      NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      buf = (int*)malloc(block_x*block_y*sizeof(int));
      while (task < tx*ty) {
        ity = task%ty;
        itx = (task-ity)/ty;
        tlo[0] = itx*block_x;
        tlo[1] = ity*block_y;
        thi[0] = tlo[0] + block_x - 1;
        if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
        thi[1] = tlo[1] + block_y - 1;
        if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
        lld = thi[1]-tlo[1]+1;
        bsize = (thi[0]-tlo[0]+1)*(thi[1]-tlo[1]+1);

        /* Accumulate values to a portion of global array */
        for (m=tlo[0]; m<=thi[0]; m++) {
          for (n=tlo[1]; n<=thi[1]; n++) {
            offset = (m-tlo[0])*lld + (n-tlo[1]);
            buf[offset] = m*dims[1]+n;
          }
        }
        delta_t = GA_Wtime();
        NGA_NbAcc(g_array, tlo, thi, buf, &lld, &one,accid);
        NGA_NbWait(accid);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_time,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_ntime,&me,&me,&one,&one,&one);
        NGA_Acc(g_elems,&me,&me,&bsize,&one,&one);
        delta_t = GA_Wtime();
        task = NGA_Read_inc(g_count, &zero, inc);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      }
      free(buf);
    }
    /* Sync all processors at end of initialization loop */
    NGA_Sync(); 

    /* Check local buffer for correct values */
    NGA_Distribution(g_array,me,glo,ghi);
    ok = 1;
    NGA_Access(g_array,glo,ghi,&ptr,gld);
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        if (ptr[icnt] != 2*(i*dims[1]+j)) {
          ok = 0;
          printf("p[%d] (Acc1) mismatch at point [%d,%d] actual: %d expected: %d\n",
              me,i,j,ptr[icnt],i*dims[1]+j);
        }
        icnt++;
      }
    }
    NGA_Release(g_array,glo,ghi);
    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nnon-blocking-acc1 test OK\n");
    } else if (me==0 && !ok) {
      printf("\nnon-blocking-acc1 test failed\n");
    }
    printTimes(g_time, g_ntime, g_elems, "Non-blocking accumulate performance");
    
    /* Sync all processors*/
    NGA_Sync(); 


    /* Testing random work pattern */
    if (me==0) {
      printf("\n[%d]Testing non-blocking-ran1 from 0.\n", me);
    }
    /* Reinitialize source array */
    NGA_Distribution(g_array,me,glo,ghi);
    NGA_Access(g_array,glo,ghi,&ptr,gld);
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        ptr[icnt] = i*dims[1]+j;
        icnt++;
      }
    }
    NGA_Release(g_array,glo,ghi);



    /* Mimic random work. */
    GA_Zero(g_dest);
    GA_Zero(g_count);

    #pragma omp parallel num_threads(thread_count)
    {
      /* declare variables local to each thread */
      int tlo[2], thi[2];
      int ld[2];
      int k, m, n;
      int xinc, yinc;
      int itx, ity;
      int offset, icnt;
      int *buf, *buft;
      int lld;
      long task, inc; 
      int id;
      int tmp;
      double delta_t;
      id = omp_get_thread_num();
      inc = 1;
      delta_t = GA_Wtime();
      task = NGA_Read_inc(g_count, &zero, inc);
      delta_t = GA_Wtime()-delta_t;
      NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
      NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      ga_nbhdl_t* gethdl=(ga_nbhdl_t*)malloc(sizeof(ga_nbhdl_t));
      ga_nbhdl_t* acchdl=(ga_nbhdl_t*)malloc(sizeof(ga_nbhdl_t));
      buf = (int*)malloc(block_x*block_y*sizeof(int));
      buft = (int*)malloc(block_x*block_y*sizeof(int));
      /* Read and transpose data */
      while (task < 2*tx*ty) {
        k = task;
        if (k>=tx*ty) k -= tx*ty;
        ity = k%ty;
        itx = (k-ity)/ty;
        tlo[0] = itx*block_x;
        tlo[1] = ity*block_y;
        thi[0] = tlo[0] + block_x - 1;
        if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
        thi[1] = tlo[1] + block_y - 1;
        if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
        ld[0] = thi[0]-tlo[0]+1;
        ld[1] = thi[1]-tlo[1]+1;
        lld = thi[1]-tlo[1]+1;

        /* Get data from g_array */
        NGA_NbGet(g_array, tlo, thi, buf, &lld, gethdl);
        NGA_NbWait(gethdl);
        /* Evaluate transpose of local bloack */
        icnt = 0;
        for (m=0; m<ld[0]; m++) {
          for (n=0; n<ld[1]; n++) {
            offset = n*ld[0]+m;
            buft[offset] = buf[icnt]; 
            icnt++;
          }
        }
        
        /* Find transposed block location */
        tmp = ity;
        ity = itx;
        itx = tmp;
        tlo[0] = itx*block_y;
        tlo[1] = ity*block_x;
        thi[0] = tlo[0] + block_y - 1;
        if (thi[0] >= dims[0]) thi[0] = dims[0]-1;
        thi[1] = tlo[1] + block_x - 1;
        if (thi[1] >= dims[1]) thi[1] = dims[1]-1;
        lld = thi[1]-tlo[1]+1;
        NGA_NbAcc(g_dest, tlo, thi, buft, &lld, &one,acchdl);
        NGA_NbWait(acchdl);
        delta_t = GA_Wtime();
        task = NGA_Read_inc(g_count, &zero, inc);
        delta_t = GA_Wtime()-delta_t;
        NGA_Acc(g_ritime,&me,&me,&delta_t,&one,&rone);
        NGA_Acc(g_rinc,&me,&me,&one,&one,&one);
      }
      free(buf);
      free(buft);
      free(gethdl);
      free(acchdl);
    }
    GA_Sync();

    /* Check local buffer for correct values */
    NGA_Distribution(g_dest,me,glo,ghi);
    ok = 1;
    NGA_Access(g_dest,glo,ghi,&ptr,gld);
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        if (ptr[icnt] != 2*(j*dims[0]+i)) {
          ok = 0;
          printf("p[%d] (Ran1) mismatch at point [%d,%d] actual: %d expected: %d\n",
              me,i,j,ptr[icnt],2*(j*dims[0]+i));
        }
        icnt++;
      }
    }
    NGA_Release(g_dest,glo,ghi);
    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nnon-blocking-ran1 test OK\n");
    } else if (me==0  && !ok) {
      printf("\nnon-blocking-ran1 test failed\n");
    }

    printTimes(g_ritime, g_rinc, g_rinc, "Read-increment performance");

#endif
    free(time);
    free(ritime);
    free(ntime);
    free(nelems);
    free(rinc);

    GA_Terminate();
    MPI_Finalize();

    return return_code;
#else
    printf("OPENMP Disabled\n");
    return 1;
#endif
}

