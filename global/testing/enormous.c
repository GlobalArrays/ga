#include <stdio.h>
#include <stdlib.h>

#include "mp3.h"
#include "ga.h"
#include "macdecls.h"

/* This test is designed to test GAs ability to move very large chunks of
 * data in a single call. It requires about 18 GBytes of memory per process
 * to run successfully */

/* create nproc by N matrix (int32 precision) */
#define N 2147483647 /* int32 max */
#define NN 65536

int main(int argc, char **argv)
{
    int me, nproc, g_a, ndim, type=MT_C_INT, *buf;
    int64_t i, j, iproc, dims[2], lo[2], hi[2], ld[2];
    int ok;
    int *ptr;

    MP_INIT(argc,argv);
    GA_Initialize_ltd(-1);

    me=GA_Nodeid();
    nproc=GA_Nnodes();

    ndim = 1;
    dims[0] = N*((long)nproc);

    if(me==0) printf("Using %ld processes\n",(long)nproc);
    if(me==0) printf("memory = %ld bytes\n",((long)nproc)*((long)N)*4);

    if (me == 0) printf("Testing large 1D arrays\n");
    g_a = NGA_Create64(type, ndim, dims, "A", NULL);

    GA_Zero(g_a);

    GA_Print_distribution(g_a);

    buf = (int*)(malloc(N*sizeof(int)));
    if(me == 0) {
      for (iproc=0; iproc<nproc; iproc++) {
        lo[0] = iproc*N;
        hi[0] = (iproc+1)*N-1;
        for (i=0; i<N; i++) buf[i]=(int)(i+iproc*N);
        NGA_Put64(g_a, lo, hi, buf, NULL);
      }
    }

    GA_Sync();
    NGA_Distribution64(g_a, me, lo, hi);
    NGA_Access64(g_a, lo, hi, &ptr, ld);
    ok = 1;
    for (i=lo[0]; i<=hi[0]; i++) {
      if (ptr[i-lo[0]] != (int)i) {
        ok = 0;
      }
    }
    if (me == 0 && ok) {
      printf("All values are correct in 1D\n");
    } else if (!ok) {
      printf("Errors on process %d in 1D\n",me);
    }

    GA_Destroy(g_a);
    free(buf);

    ndim = 2;
    dims[0] = NN*((long)nproc);
    dims[1] = NN;

    ld[0] = NN;

    if (me == 0) printf("Testing large 2D arrays\n");
    g_a = NGA_Create64(type, ndim, dims, "B", NULL);

    GA_Zero(g_a);

    GA_Print_distribution(g_a);

    buf = (int*)(malloc(((long)NN)*((long)NN)*sizeof(int)));

    if(me == 0) {
      for (iproc=0; iproc<nproc; iproc++) {
        lo[0] = iproc*NN;
        hi[0] = (iproc+1)*NN-1;
        lo[1] = 0;
        hi[1] = NN-1;
        for (i=0; i<NN; i++) {
          for (j=0; j<NN; j++) {
            buf[i*NN+j]=(int)((i+iproc*NN)*NN+j);
          }
        }
        NGA_Put64(g_a, lo, hi, buf, ld);
      }
    }

    GA_Sync();
    NGA_Distribution64(g_a, me, lo, hi);
    NGA_Access64(g_a, lo, hi, &ptr, ld);
    ok = 1;
    for (i=lo[0]; i<=hi[0]; i++) {
      for (j=lo[1]; j<=hi[1]; j++) {
        if (ptr[(i-lo[0])*ld[0]+(j-lo[1])] != (int)(i*NN+j)) {
          ok = 0;
        }
      }
    }
    if (me == 0 && ok) {
      printf("All values are correct in 2D\n");
    } else if (!ok) {
      printf("Errors on process %d in 2D\n",me);
    }

    GA_Destroy(g_a);
    free(buf);

    GA_Terminate();
    MP_FINALIZE();

    return 0;
}
