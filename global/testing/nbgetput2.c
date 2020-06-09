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

#include <unistd.h>

// #define NN 8192            /* dimension of matrices */
#define NN 32768            /* dimension of matrices */
#define WINDOWSIZE 2
#define min(a,b) (((a)<(b))?(a):(b))

#define TIMER GA_Wtime
int main( int argc, char **argv ) {
  int g_a;
  int g_b;
  int g_c;
  int lo[2], hi[2];
  int dims[2];
  int me, nproc;
  int  ld, isize, jsize;
  int type=MT_C_INT;
  int nelems, ok;
  int *buf_a, *ptr_a;
  int *buf_b, *ptr_b;
  int *buf_c, *ptr_c;
  ga_nbhdl_t *nbhdl_a;
  ga_nbhdl_t *nbhdl_b;
  ga_nbhdl_t *nbhdl_c;

  int heap=3000000, stack=2000000;

  MP_INIT(argc,argv);

  GA_INIT(argc,argv);                            /* initialize GA */
  me=GA_Nodeid(); 
  nproc=GA_Nnodes();
  if(me==0) {
    if(GA_Uses_fapi())GA_Error("Program runs with C array API only",1);
    printf("\nUsing %d processes\n",nproc);
    fflush(stdout);
  }

  heap /= nproc;
  stack /= nproc;
  if(! MA_init(MT_F_DBL, stack, heap)) 
    GA_Error("MA_init failed",stack+heap);  /* initialize memory allocator*/ 
  int N;
for (N = 8; N < NN; N*=2) {

  /* Create a regular matrix. */
  if(me==0)printf("\nCreating matrix A of size %d x %d\n",N,N);
  if(me==0)printf("\nCreating matrix B of size %d x %d\n",N,N);
  if(me==0)printf("\nCreating matrix C of size %d x %d\n",N,N);
  dims[0] = N;
  dims[1] = N;
  
  int n;
  g_a = NGA_Create(type, 2, dims, "A", NULL);
  if(!g_a) GA_Error("create failed: A",n); 

  g_b = NGA_Create(type, 2, dims, "B", NULL);
  if(!g_b) GA_Error("create failed: B",n); 

  g_c = NGA_Create(type, 2, dims, "C", NULL);
  if(!g_c) GA_Error("create failed: C",n); 

  /* Fill matrix from process 0 using non-blocking puts */
  nelems = N*N;
  GA_Sync();
  /* Copy matrix to process 0 using non-blocking gets */
  if (me == 0) {

    buf_a = (int*)malloc(nelems*sizeof(int));
    buf_b = (int*)malloc(nelems*sizeof(int));
    buf_c = (int*)malloc(nelems*sizeof(int));

    nbhdl_a = (ga_nbhdl_t*)malloc(nproc*sizeof(ga_nbhdl_t));
    nbhdl_b = (ga_nbhdl_t*)malloc(nproc*sizeof(ga_nbhdl_t));
    nbhdl_c = (ga_nbhdl_t*)malloc(nproc*sizeof(ga_nbhdl_t));

    //TODO: set this value
    //int alpha=1;

    double nbput_timings = 0;
    double nbget_timings = 0;
    double nbacc_timings = 0;
    double wait_timings = 0;
    double start_time, end_time;

    ptr_c = buf_c;
    int i;
    for (i=0; i<nproc; i+=WINDOWSIZE) {
        ld = N;
        int j;
        for(j=i; j <min(nproc, i+WINDOWSIZE); j++)
        {
            if(j==me) continue;

            NGA_Distribution(g_a, j, lo, hi);
            ptr_a = buf_a + lo[1] + N*lo[0];
            start_time = TIMER();
            NGA_NbGet(g_a, lo, hi, ptr_a, &ld, &nbhdl_a[j]);
            end_time = TIMER();
            nbget_timings += end_time - start_time;

            NGA_Distribution(g_b, j, lo, hi);
            ptr_b = buf_b + lo[1] + N*lo[0];
            start_time = TIMER();
            NGA_NbGet(g_b, lo, hi, ptr_b, &ld, &nbhdl_b[j]);
            end_time = TIMER();
            nbget_timings += end_time - start_time;


            NGA_Distribution(g_c, j, lo, hi);
            isize = (hi[0]-lo[0]+1);
            jsize = (hi[1]-lo[1]+1);
            ld = jsize;

            start_time = TIMER();
            NGA_NbPut(g_c, lo, hi, ptr_c, &ld, &nbhdl_c[j]);  
            end_time = TIMER();
            nbput_timings += end_time - start_time;
            ptr_c += isize*jsize;
        }

        for(j=i; j <min(nproc, i+WINDOWSIZE); j++)
        {
            if(j==me) continue;

            start_time = TIMER();
            NGA_NbWait(&nbhdl_a[j]);
            end_time = TIMER();
            wait_timings += end_time - start_time;

            start_time = TIMER();
            NGA_NbWait(&nbhdl_b[j]);
            end_time = TIMER();
            wait_timings += end_time - start_time;
            // if(j==me) continue;

            start_time = TIMER();
            NGA_NbWait(&nbhdl_c[j]);
            end_time = TIMER();
            wait_timings += end_time - start_time;
        }
    }

    printf("\n\n");
    printf("NbGet timings: %lf sec\n", nbget_timings);
    printf("NbPut timings: %lf sec\n", nbput_timings);
    printf("NbAcc timings: %lf sec\n", nbacc_timings);
    printf("NbWait timings: %lf sec\n", wait_timings);
    printf("================================\n");
    printf("NbTotal time: %lf sec\n", 
       nbget_timings + nbput_timings + nbacc_timings + wait_timings);
    printf("================================\n");
    printf("\n\n");

    double put_timings = 0;
    double get_timings = 0;
    double acc_timings = 0;

    ptr_c = buf_c;

    for (i=0; i<nproc; i+=WINDOWSIZE) {
        ld = N;
        int j;
        for(j=i; j <min(nproc, i+WINDOWSIZE); j++)
        {
            if(j==me) continue;

            NGA_Distribution(g_a, j, lo, hi);
            ptr_a = buf_a + lo[1] + N*lo[0];
            start_time = TIMER();
            NGA_Get(g_a, lo, hi, ptr_a, &ld);
            end_time = TIMER();
            get_timings += end_time - start_time;

            NGA_Distribution(g_b, j, lo, hi);
            ptr_b = buf_b + lo[1] + N*lo[0];
            start_time = TIMER();
            NGA_Get(g_b, lo, hi, ptr_b, &ld);
            end_time = TIMER();
            get_timings += end_time - start_time;

        // }

        // for(int j=i; j <min(nproc, i+WINDOWSIZE); j++)
        // {
        //     if(j==me) continue;
            NGA_Distribution(g_c, j, lo, hi);
            isize = (hi[0]-lo[0]+1);
            jsize = (hi[1]-lo[1]+1);
            ld = jsize;

            start_time = TIMER();
            NGA_Put(g_c, lo, hi, ptr_c, &ld);  
            end_time = TIMER();
            put_timings += end_time - start_time;

            ptr_c += isize*jsize;
        }
    }
    printf("\n\n");
    printf("Get timings: %lf sec\n", get_timings);
    printf("Put timings: %lf sec\n", put_timings);
    printf("Acc timings: %lf sec\n", acc_timings);
    printf("================================\n");
    printf("Total time: %lf sec\n", 
       get_timings + put_timings + acc_timings);
    printf("================================\n");
    printf("\n\n");


    free(buf_a);
    free(buf_b);
    free(buf_c);
    free(nbhdl_a);
    free(nbhdl_b);
    free(nbhdl_c);
  }
  GA_Sync();

  GA_Destroy(g_a);
  GA_Destroy(g_b);
  GA_Destroy(g_c);
}
  if(me==0)printf("\nSuccess\n");
  GA_Terminate();

  MP_FINALIZE();

 return 0;
}
