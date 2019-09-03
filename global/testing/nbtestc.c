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

#define N 128            /* dimension of matrices */

#define TEST 1

int main( int argc, char **argv ) {
  int g_a;
  int g_cnt, zero;
  long lone;
  int one;
  int lo[2], hi[2];
  int dims[2];
  int me, nproc;
  int i, j, n, ld, isize, jsize, icnt, nleft;
  int type=MT_C_INT;
  int nelems, ok;
  int *buf, *ptr;
  ga_nbhdl_t *nbhdl;

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

  one = 1;
  lone = 1;
  zero = 0;
  /* Create a regular matrix. */
  if(me==0)printf("\nCreating matrix A of size %d x %d\n",N,N);
  dims[0] = N;
  dims[1] = N;
  g_a = NGA_Create(type, 2, dims, "A", NULL);
  if(!g_a) GA_Error("create failed: A",n); 
  g_cnt = NGA_Create(type,one,&one,"COUNT",NULL);
  GA_Zero(g_cnt);

#if 1
  /* Fill matrix from process 0 using non-blocking puts */
  nelems = N*N;
  if (me == 0) {
    buf = (int*)malloc(nelems*sizeof(int));
    ptr = buf;
    nbhdl = (ga_nbhdl_t*)malloc(nproc*sizeof(ga_nbhdl_t));
    printf("p[%d] Calling NbPut\n",me);
      NGA_Read_inc(g_cnt,&zero,lone); /* 0 */
    for (n=0; n<nproc; n++) {
      NGA_Distribution(g_a, n, lo, hi);
      isize = (hi[0]-lo[0]+1);
      jsize = (hi[1]-lo[1]+1);
      ld = jsize;
      for (i=0; i<isize; i++) {
        for (j=0; j<jsize; j++) {
          ptr[j+ld*i] = j+lo[1] + N*(i+lo[0]);
        }
      }
      NGA_NbPut(g_a, lo, hi, ptr, &ld, &nbhdl[n]);
      /* Buffer may still be in use (until test returns false). Need to
       * a new buffer for next put */
      ptr = ptr + isize*jsize;
    }
    printf("p[%d] Completed NbPut, calling NbTest\n",me);
    /* Test handles until completion */
    nleft = nproc;
      NGA_Read_inc(g_cnt,&zero,lone); /* 1 */
    while (nleft > 0) {
      icnt = 0;
      for (n=0; n<nleft; n++) {
#if TEST
        if (NGA_NbTest(&nbhdl[n])) {
          nbhdl[icnt] = nbhdl[n];
          icnt++;
        }
#else
        NGA_NbWait(&nbhdl[n]);
#endif
      }
      nleft = icnt;
    }
    printf("p[%d] Completed NbTest\n",me);
  }
  GA_Sync();

  /* Check to see if g_a is filled with correct values */
  NGA_Distribution(g_a, me, lo, hi);
  NGA_Access(g_a, lo, hi, &ptr, &ld);
  isize = (hi[0]-lo[0]+1);
  jsize = (hi[1]-lo[1]+1);
  ld = jsize;
  ok = 1;
  for (i=0; i<isize; i++) {
    for (j=0; j<jsize; j++) {
      if (ptr[j+ld*i] != j+lo[1] + N*(i+lo[0])) {
        ok = 0;
        printf("p[%d] Mismatch for non-blocking put. Expected: %d Actual: %d\n",
            me,j+lo[1] + N*(i+lo[0]),ptr[j+ld*isize]);
      }
    }
  }
  NGA_Release(g_a, lo, hi);
  GA_Igop(&ok,1,"*");
  if (me == 0) {
    if (ok) {
      printf("\nNon-blocking put test is OK\n");
    } else {
      printf("\nNbTest fails for non-blocking puts\n");
    }
  }
  /* Test non-blocking put to more than 1 processor */
  GA_Sync();
  GA_Zero(g_a);
  if (me == 0) {
    lo[0] = 0;
    hi[0] = N-1;
    lo[1] = 0;
    hi[1] = N-1;
    isize = (hi[0]-lo[0]+1);
    jsize = (hi[1]-lo[1]+1);
    ld = jsize;
    for (i=0; i<isize; i++) {
      for (j=0; j<jsize; j++) {
        buf[j+ld*i] = j + N*i;
      }
    }
      NGA_Read_inc(g_cnt,&zero,lone); /* 2 */
    NGA_NbPut(g_a, lo, hi, buf, &ld, &nbhdl[0]);
    /* Test handle until completion */
      NGA_Read_inc(g_cnt,&zero,lone); /* 3 */
#if TEST
    jsize = 0;
    while (NGA_NbTest(&nbhdl[0])) {
      jsize++;
    }
#else
    NGA_NbWait(&nbhdl[0]);
#endif
    free(buf);
  }
  GA_Sync();
  /* Check to see if g_a is filled with correct values */
  NGA_Distribution(g_a, me, lo, hi);
  NGA_Access(g_a, lo, hi, &ptr, &ld);
  isize = (hi[0]-lo[0]+1);
  jsize = (hi[1]-lo[1]+1);
  ld = jsize;
  ok = 1;
  for (i=0; i<isize; i++) {
    for (j=0; j<jsize; j++) {
      if (ptr[j+ld*i] != j+lo[1] + N*(i+lo[0])) {
        ok = 0;
        printf("p[%d] Mismatch for multi-process non-blocking put."
            " Expected: %d Actual: %d\n",
            me,j+lo[1] + N*(i+lo[0]),ptr[j+ld*isize]);
      }
    }
  }
  NGA_Release(g_a, lo, hi);
  GA_Igop(&ok,1,"*");
  if (me == 0) {
    if (ok) {
      printf("\nNon-blocking put test with multiple processes is OK\n");
    } else {
      printf("\nNbTest fails for non-blocking put to multiple processes\n");
    }
  }
#else
  NGA_Distribution(g_a, me, lo, hi);
  NGA_Access(g_a, lo, hi, &ptr, &ld);
  for (i=0; i<hi[0]-lo[0]+1; i++) {
    for (j=0; j<hi[1]-lo[1]+1; j++) {
      ptr[j+ld*i] = j+lo[1] + (i+lo[0])*N;
    }
  }
  NGA_Release(g_a,lo,hi);
  GA_Sync();
#endif

  /* Copy matrix to process 0 using non-blocking gets */
  if (me == 0) {
    buf = (int*)malloc(N*N*sizeof(int));
    ld = N;
      NGA_Read_inc(g_cnt,&zero,lone); /* 4 */
    for (n=0; n<nproc; n++) {
      NGA_Distribution(g_a, n, lo, hi);
      printf("p[%d] n: %d lo[0]: %d hi[0]: %d lo[1]: %d hi[1]: %d\n",
          me,n,lo[0],hi[0],lo[1],hi[1]);
      /* figure out offset in local buffer */
      ptr = buf + lo[1] + N*lo[0];
      NGA_NbGet(g_a, lo, hi, ptr, &ld, &nbhdl[n]);
    }
    /* Test handles until completion */
    nleft = nproc;
      NGA_Read_inc(g_cnt,&zero,lone); /* 5 */
    while (nleft > 0) {
      icnt = 0;
      for (n=0; n<nleft; n++) {
#if TEST
        if (NGA_NbTest(&nbhdl[n])) {
          nbhdl[icnt] = nbhdl[n];
          icnt++;
          printf("p[%d] ICNT: %d\n",me,icnt);
        }
#else
        NGA_NbWait(&nbhdl[n]);
#endif
      }
      nleft = icnt;
    }
    /* Check to see if local buffer is filled with correct values */
    ok = 1;
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        if (buf[j+N*i] != j+N*i) {
          printf("p[%d] Mismatch for non-blocking get. Expected: %d Actual: %d\n",
            me,j + N*i,buf[j+N*i]);
          ok = 0;
        }
      }
    }
    if (ok) {
      printf("\nNon-blocking get test is OK\n");
    } else {
      printf("\nNbTest fails for non-blocking gets\n");
    }
    free(buf);
    free(nbhdl);
  }
  GA_Sync();

  GA_Destroy(g_a);
  if(me==0)printf("\nSuccess\n");
  GA_Terminate();

  MP_FINALIZE();

 return 0;
}
