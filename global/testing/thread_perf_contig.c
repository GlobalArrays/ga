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

#define DEFAULT_DIM 1024*1024
#define MAX_MESSAGE_SIZE DEFAULT_DIM*DEFAULT_DIM

#define MAX_FACTOR 256
/* Convenience function to check that something is true on all processors */
int trueEverywhere(int flag)
{
  int tflag, nprocs;
  if (flag) tflag = 1;
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
  optime = 1.0e6*(l_time)/((double)l_ntime);
  if (me==0) {
    printf("         %7d      %12.6f        %12.6f       %3d\n",
      size,optime,bandwdth,nthread);
  }
}

/* Function to print out timing statistics for read-increment */
void printRITimes(double *time, int *ntime, int nthread)
{
  int me = GA_Nodeid();
  int nproc = GA_Nnodes();
  double l_time;
  int l_ntime;
  int one = 1;
  double optime;
  int i;

  l_time = 0.0;
  l_ntime = 0;
  for (i=0; i<nthread; i++) {
    l_time += time[i];
    l_ntime += ntime[i];
  }
  GA_Dgop(&l_time,one,"+");
  GA_Igop(&l_ntime,one,"+");
  optime = 1.0e6*(l_time)/((double)l_ntime);
  if (me==0) {
    printf("\nStatistics for Read-Increment\n");
    printf("\nTotal operations     Time per op (us)\n");
    printf("     %10d     %16.6f\n",l_ntime,optime);
  }
}

int main(int argc, char * argv[])
{
#if defined(_OPENMP)
    int dim = DEFAULT_DIM;
    int block;
    int g_array, g_count;
    int me, nproc;
    int glo, ghi, gld;
    int tn;
    int i,j,icnt;
    int return_code = 0;
    int thread_count = 4;
    int zero = 0, one = 1;
    double rone = 1.0;
    int ok;
    int *ptr;
    char *env_threads;
    int provided;
    double *time, *ritime;
    int *ntime, *nelems, *rinc, *arr_ok;
    int msg_size, ithread, iter;
    int buf_size = DEFAULT_DIM;
    int ncount;
    int ulim;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    GA_Initialize();

    nproc = GA_Nnodes();
    me = GA_Nodeid();
    if (provided < MPI_THREAD_MULTIPLE && me==0) {
      printf("\nMPI_THREAD_MULTIPLE not provided\n");
    }

    if (me==0) {
      printf("\nTest running of %d processors\n",nproc);
      printf("\n  Array dimension is %d",dim);
    }

    /* Create GA and set all elements to zero */
    g_array = NGA_Create(C_INT, 1, &dim, "source", NULL);
    g_count = NGA_Create(C_INT, 1, &one, "counter", NULL);
    GA_Zero(g_array);

    if(env_threads = getenv("OMP_NUM_THREADS"))
        thread_count = atoi(env_threads);
    else
        omp_set_num_threads(thread_count);

    if (thread_count > 8) thread_count = 8;

    if (me==0) {
            printf("\n[%d]Testing %d threads.\n", me, thread_count);
    }
    time = (double*)malloc(thread_count*sizeof(double));
    ritime = (double*)malloc(thread_count*sizeof(double));
    ntime = (int*)malloc(thread_count*sizeof(int));
    nelems = (int*)malloc(thread_count*sizeof(int));
    rinc = (int*)malloc(thread_count*sizeof(int));
    arr_ok = (int*)malloc(thread_count*sizeof(int));
    for (i=0; i<thread_count; i++) {
      time[i] = 0.0;
      ritime[i] = 0.0;
      ntime[i] = 0;
      nelems[i] = 0;
      rinc[i] = 0;
      arr_ok[i] = 0;
    }

    if (me==0) {
       printf("\nPerformance of GA_Put\n");
       printf("\nmsg size (bytes)     avg time (us)    avg b/w (MB/sec) N threads\n");
    }

    ok = 1;
    for (ithread = 1; ithread<= thread_count; ithread++) {
      for (msg_size = 1; msg_size <= buf_size; msg_size *= 2) {
        for (i=0; i<ithread; i++) {
          time[i] = 0.0;
          ntime[i] = 0;
          nelems[i] = 0;
        }
        tn = dim/msg_size;
        if (tn*msg_size < dim) tn++;
        /* Fill global array with data by having each thread write
         * blocks to it */
        GA_Zero(g_count);
#pragma omp parallel num_threads(ithread)
        {
          /* declare variables local to each thread */
          int tlo, thi;
          int k, m, n;
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
          buf = (int*)malloc(msg_size*sizeof(int));
          while (task < tn) {
            tlo = task*msg_size;
            thi = tlo + msg_size - 1;
            if (thi >= dim) thi = dim-1;
            lld = thi-tlo+1;
            bsize = thi-tlo+1;

            /* Fill a portion of local buffer with correct values */
            for (m=tlo; m<=thi; m++) {
              offset = m-tlo;
              buf[offset] = m;
            }

            delta_t = GA_Wtime();
            NGA_Put(g_array, &tlo, &thi, buf, &lld);
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
        printTimes(time,ntime,nelems,msg_size,ithread);
        /* Sync all processors at end of initialization loop */
        NGA_Sync(); 
        /* Each process determines if it is holding the correct data */
        NGA_Distribution(g_array,me,&glo,&ghi);
        NGA_Access(g_array,&glo,&ghi,&ptr,&gld);
        icnt = 0;
        for (i=glo; i<=ghi; i++) {
          if (ptr[icnt] != i) {
            ok = 0;
            printf("p[%d] (Put) mismatch at point [%d] actual: %d expected: %d\n",
                me,i,ptr[icnt],i);
          }
          icnt++;
        }
        NGA_Release(g_array,&glo,&ghi);
      }
    }

    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nPut test OK\n");
    } else if (me == 0 && !ok) {
      printf("\nPut test failed\n");
    }

    if (me==0) {
       printf("\nPerformance of GA_Get\n");
       printf("\nmsg size (bytes)     avg time (us)    avg b/w (MB/sec) N threads\n");
    }

    ok = 1;
    for (ithread = 1; ithread<= thread_count; ithread++) {
      for (msg_size = 1; msg_size <= buf_size; msg_size *= 2) {
        for (i=0; i<ithread; i++) {
          time[i] = 0.0;
          ntime[i] = 0;
          nelems[i] = 0;
          arr_ok[i] = 0;
        }
        tn = dim/msg_size;
        if (tn*msg_size < dim) tn++;
        /* Fill global array with data by having each thread write
         * blocks to it */
        GA_Zero(g_count);
#pragma omp parallel num_threads(ithread)
        {
          /* declare variables local to each thread */
          int tlo, thi;
          int k, m, n;
          int offset;
          int *buf;
          int lld;
          long task, inc; 
          int id;
          double delta_t;
          int bsize;
          id = omp_get_thread_num();
          inc = 1;
          arr_ok[id] = 1;
          delta_t = GA_Wtime();
          task = NGA_Read_inc(g_count, &zero, inc);
          delta_t = GA_Wtime()-delta_t;
          ritime[id] += delta_t;
          rinc[id] += one;
          buf = (int*)malloc(msg_size*sizeof(int));
          while (task < tn) {
            tlo = task*msg_size;
            thi = tlo+msg_size-1;
            if (thi >= dim) thi = dim-1;
            lld = thi-tlo+1;
            bsize = thi-tlo+1;

            delta_t = GA_Wtime();
            NGA_Get(g_array, &tlo, &thi, buf, &lld);
            delta_t = GA_Wtime()-delta_t;
            time[id] += delta_t;
            ntime[id] += one;
            nelems[id] += bsize;

            /* check that values in buffer are correct */
            for (m=tlo; m<=thi; m++) {
              offset = m-tlo;
              if (buf[offset] != m) {
                arr_ok[id] = 0;
                printf("p[%d] Read mismatch for [%d] expected: %d actual: %d\n",
                    me,m,m,buf[offset]);
              }
            }

            delta_t = GA_Wtime();
            task = NGA_Read_inc(g_count, &zero, inc);
            delta_t = GA_Wtime()-delta_t;
            ritime[id] += delta_t;
            rinc[id] += one;
          }
          free(buf);
        }
        for (i=0; i<ithread; i++) if (arr_ok[i] == 0) ok = 0;
        printTimes(time,ntime,nelems,msg_size,ithread);
        /* Sync all processors at end of initialization loop */
        NGA_Sync(); 
      }
    }
    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nGet test OK\n");
    } else if (me == 0 && !ok) {
      printf("\nGet test failed\n");
    }


    if (me==0) {
       printf("\nPerformance of GA_Acc\n");
       printf("\nmsg size (bytes)     avg time (us)    avg b/w (MB/sec) N threads\n");
    }

    ok = 1;
    for (ithread = 1; ithread<= thread_count; ithread++) {
      for (msg_size = 1; msg_size <= buf_size; msg_size *= 2) {
        for (i=0; i<ithread; i++) {
          time[i] = 0.0;
          ntime[i] = 0;
          nelems[i] = 0;
        }
        tn = dim/msg_size;
        if (tn*msg_size < dim) tn++;
        /* Fill global array with data by having each thread write
         * blocks to it */
        GA_Zero(g_count);
        GA_Zero(g_array);
#pragma omp parallel num_threads(ithread)
        {
          /* declare variables local to each thread */
          int tlo, thi;
          int k, m, n;
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
          buf = (int*)malloc(msg_size*sizeof(int));
          while (task < 2*tn) {
            k = task;
            if (task >= tn) k = k - tn;
            tlo = k*msg_size;
            thi = tlo+msg_size-1;
            if (thi >= dim) thi = dim-1;
            lld = thi-tlo+1;
            bsize = thi-tlo+1;

            /* Fill a portion of local buffer with correct values */
            for (m=tlo; m<=thi; m++) {
              offset = m-tlo;
              buf[offset] = m;
            }

            delta_t = GA_Wtime();
            NGA_Acc(g_array, &tlo, &thi, buf, &lld, &one);
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
        printTimes(time,ntime,nelems,msg_size,ithread);
        /* Sync all processors at end of initialization loop */
        NGA_Sync(); 
        /* Each process determines if it is holding the correct data */
        NGA_Distribution(g_array,me,&glo,&ghi);
        NGA_Access(g_array,&glo,&ghi,&ptr,&gld);
        icnt = 0;
        for (i=glo; i<=ghi; i++) {
          if (ptr[icnt] != 2*i) {
            ok = 0;
            printf("p[%d] (Acc) mismatch at point [%d] actual: %d expected: %d\n",
                me,i,ptr[icnt],2*i);
          }
          icnt++;
        }
        NGA_Release(g_array,&glo,&ghi);
      }
    }
    ok = trueEverywhere(ok);
    if (me==0 && ok) {
      printf("\nAcc test OK\n");
    } else if (me == 0 && !ok) {
      printf("\nAcc test failed\n");
    }

    printRITimes(ritime, rinc, thread_count);

    free(time);
    free(ritime);
    free(ntime);
    free(nelems);
    free(rinc);
    free(arr_ok);

    GA_Terminate();
    MPI_Finalize();

    return return_code;
#else
    printf("OPENMP Disabled\n");
    return 1;
#endif
}

