#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif

#include "ga.h"
#include "macdecls.h"
#include "mp3.h"

#define N 100            /* dimension of array */
#define NCNT 1000000     /* number of random numbers generated on each process */


int main(int argc, char **argv)
{
int heap=20000, stack=20000;
int me, nproc;
int i, idx;
double x, dx, norm;
double min, max;
int *bins;
char op[2];

    MP_INIT(argc,argv);

    GA_INIT(argc,argv);                            /* initialize GA */
    me=GA_Nodeid(); 
    nproc=GA_Nnodes();
    if(me==0) {
       if(GA_Uses_fapi())GA_Error("Program runs with C array API only",1);
       printf("Using %ld processes\n",(long)nproc);
       fflush(stdout);
    }

    heap /= nproc;
    stack /= nproc;
    if(! MA_init(MT_F_DBL, stack, heap)) 
       GA_Error("MA_init failed",stack+heap);  /* initialize memory allocator*/ 
    NGA_Rand(332457+me);

    bins = (int*)malloc(N*sizeof(int));
    for (i=0; i<N; i++) bins[i] = 0;
    dx = 1.0/((double)N);
    for (i=0; i<NCNT; i++) {
      x = NGA_Rand(0);
      idx = (int)(x/dx); 
      if (idx >= N) GA_Error("Index out of bounds",idx);
      bins[idx]++;
    }
    strcpy(op,"+");
    GA_Igop(bins,N,op);
    norm = ((double)(NCNT*nproc));
    if (me == 0) {
      min = ((double)N);
      max = 0.0;
      for (i=0; i<N; i++) {
        x = ((double)bins[i])/norm;
        if (x<min) min = x;
        if (x>max) max = x;
        printf("%f %f\n",(((double)i)+0.5)*dx,x);
      }
      printf("Min: %f Max: %f\n",min,max);
    }
    free(bins);

    GA_Terminate();
    MP_FINALIZE();

    return 0;
}

