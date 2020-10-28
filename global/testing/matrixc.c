#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif

#include "ga.h"
#include "macdecls.h"
#include "mp3.h"

#define NDIM 8             /* dimension of matrices */

#define USEX_BLOCK_CYCLIC

/* Test the functions in matrix.c on double precision arrays*/
void do_work()
{
  int ONE=1, TWO=2;   /* useful constants */
  double RONE = 1.0;
  int g_a, g_v;
  int me=GA_Nodeid(), nproc=GA_Nnodes();
  int i, j, m, n, ndim, ld, ok;
  int dims[2];
  int lo[2], hi[2];
  int count;
  double a[NDIM*NDIM];
  double b[NDIM*NDIM];
  double va[NDIM];
  double vb[NDIM];
#ifdef USE_BLOCK_CYCLIC
  int block_size[2];
#endif

  /* Test vectors */
  ndim = ONE;
  dims[0] = NDIM;
  dims[1] = NDIM;
  lo[0] = 0;
  lo[1] = 0;
  hi[0] = NDIM-1;
  hi[1] = NDIM-1;
  ld = NDIM;

  /* Create vector */
  g_v = NGA_Create_handle();
  NGA_Set_data(g_v, ndim, dims, C_DBL);
#ifdef USE_BLOCK_CYCLIC
  block_size[0] = 17;
  block_size[1] = 17;
  NGA_Set_block_cyclic(g_v, block_size);
#endif
  NGA_Allocate(g_v);

  /* Create matrix */
  ndim = TWO;
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_DBL);
#ifdef USE_BLOCK_CYCLIC
  NGA_Set_block_cyclic(g_a, block_size);
#endif
  NGA_Allocate(g_a);
  /* Initialize data in local buffer */
  for (i=0; i<NDIM; i++) {
    for (j=0; j<NDIM; j++) {
      a[j + i*NDIM] = (double)(j+i*NDIM);
    }
  }
  if (me==0) {
    NGA_Put(g_a,lo,hi,a,&ld);
  }
  GA_Sync();
  
  /* Test GA_Get_diag */
  if (me == 0) {
    printf("\nTesting GA_Get_diag\n");
  }
  GA_Get_diag(g_a, g_v);
  GA_Sync();
  NGA_Get(g_v,lo,hi,vb,&ONE);
  ok = 1; 
  count = 0;
  for (i=0; i<NDIM; i++) {
    if (vb[i] != (double)count) {
      printf("Mismatch for get_diag i: %d expected: %f actual: %f\n",
          i,(double)count,vb[i]);
      ok = 0;
    }
    count += NDIM+1;
  }
  GA_Igop(&ok,ONE,"*");
  
  if (me == 0) {
    if (ok) {
      printf("\n  GA_Get_diag OK\n");
    } else {
      printf("\n  GA_Get_diag not OK\n");
    }
  }

  /* Test GA_Add_diagonal */
  if (me == 0) {
    printf("\nTesting GA_Add_diagonal\n");
  }
  GA_Add_diagonal(g_a, g_v);
  GA_Sync();
  GA_Get_diag(g_a, g_v);
  GA_Sync();
  NGA_Get(g_v,lo,hi,vb,&ONE);
  ok = 1; 
  count = 0;
  for (i=0; i<NDIM; i++) {
    if (vb[i] != 2.0*((double)count)) {
      printf("Mismatch for add_diagonal i: %d expected: %f actual: %f\n",
          i,2.0*((double)count),vb[i]);
      ok = 0;
    }
    count += NDIM+1;
  }
  GA_Igop(&ok,ONE,"*");
  
  if (me == 0) {
    if (ok) {
      printf("\n  GA_Add_diagonal OK\n");
    } else {
      printf("\n  GA_Add_diagonal not OK\n");
    }
  }

  /* Test GA_Set_diagonal */
  if (me == 0) {
    printf("\nTesting GA_Set_diagonal\n");
  }
  count = 0;
  for (i=0; i<NDIM; i++) {
    va[i] = (double)count;
    count += NDIM+1;
  }
  if (me==0) {
    NGA_Put(g_v,lo,hi,va,&ONE);
  }
  GA_Sync();
  GA_Set_diagonal(g_a, g_v);
  GA_Sync();
  GA_Get_diag(g_a, g_v);
  GA_Sync();
  NGA_Get(g_v,lo,hi,vb,&ONE);
  ok = 1; 
  count = 0;
  for (i=0; i<NDIM; i++) {
    if (vb[i] != (double)count) {
      printf("Mismatch for set_diagonal i: %d expected: %f actual: %f\n",
          i,(double)count,vb[i]);
      ok = 0;
    }
    count += NDIM+1;
  }
  GA_Igop(&ok,ONE,"*");
  
  if (me == 0) {
    if (ok) {
      printf("\n  GA_Set_diagonal OK\n");
    } else {
      printf("\n  GA_Set_diagonal not OK\n");
    }
  }

  /* Test GA_Shift_diagonal */
  if (me == 0) {
    printf("\nTesting GA_Shift_diagonal\n");
  }
  GA_Shift_diagonal(g_a, &RONE);
  GA_Sync();
  GA_Get_diag(g_a, g_v);
  GA_Sync();
  NGA_Get(g_v,lo,hi,vb,&ONE);
  ok = 1; 
  count = 0;
  for (i=0; i<NDIM; i++) {
    if (vb[i] != (double)(count+1)) {
      printf("Mismatch for shift_diagonal i: %d expected: %f actual: %f\n",
          i,(double)(count+1),vb[i]);
      ok = 0;
    }
    count += NDIM+1;
  }
  GA_Igop(&ok,ONE,"*");
  
  if (me == 0) {
    if (ok) {
      printf("\n  GA_Shift_diagonal OK\n");
    } else {
      printf("\n  GA_Shift_diagonal not OK\n");
    }
  }

  /* Test GA_Zero_diagonal */
  if (me == 0) {
    printf("\nTesting GA_Zero_diagonal\n");
  }
  GA_Zero_diagonal(g_a);
  GA_Sync();
  GA_Get_diag(g_a, g_v);
  GA_Sync();
  NGA_Get(g_v,lo,hi,vb,&ONE);
  ok = 1; 
  count = 0;
  for (i=0; i<NDIM; i++) {
    if (vb[i] != 0.0) {
      printf("Mismatch for zero_diagonal i: %d expected: %f actual: %f\n",
          i,0.0,vb[i]);
      ok = 0;
    }
    count += NDIM+1;
  }
  GA_Igop(&ok,ONE,"*");
  
  if (me == 0) {
    if (ok) {
      printf("\n  GA_Zero_diagonal OK\n");
    } else {
      printf("\n  GA_Zero_diagonal not OK\n");
    }
  }

  /* Test GA_Scale_rows */
  if (me == 0) {
    printf("\nTesting GA_Scale_rows\n");
  }
  /* Reset values in g_v and g_a */
  if (me == 0) {
    NGA_Put(g_v,lo,hi,va,&ONE);
  }
  GA_Sync();
  GA_Set_diagonal(g_a, g_v);
  GA_Sync();

  GA_Scale_rows(g_a,g_v);
  GA_Sync();
  NGA_Get(g_a,lo,hi,b,&ld);
  GA_Sync();
  ok = 1; 
  for (i=0; i<NDIM; i++) {
    for (j=0; j<NDIM; j++) {
      if (b[i*NDIM+j] != a[i*NDIM+j]*va[i]) {
        printf("Mismatch for scale_rows i: %d j: %d expected: %f actual: %f\n",
            i,j,a[i*NDIM+j]*va[i],b[i*NDIM+j]);
        ok = 0;
      }
    }
  }
  GA_Igop(&ok,ONE,"*");
  
  if (me == 0) {
    if (ok) {
      printf("\n  GA_Scale_rows OK\n");
    } else {
      printf("\n  GA_Scale_rows not OK\n");
    }
  }

  /* Test GA_Scale_rows */
  if (me == 0) {
    printf("\nTesting GA_Scale_cols\n");
  }
  /* Reset values in g_v and g_a */
  if (me == 0) {
    NGA_Put(g_a,lo,hi,a,&ld);
    NGA_Put(g_v,lo,hi,va,&ONE);
  }
  GA_Sync();

  GA_Scale_cols(g_a,g_v);
  GA_Sync();
  NGA_Get(g_a,lo,hi,b,&ld);
  GA_Sync();
  ok = 1; 
  for (i=0; i<NDIM; i++) {
    for (j=0; j<NDIM; j++) {
      if (b[i*NDIM+j] != a[i*NDIM+j]*va[j]) {
        printf("Mismatch for scale_cols i: %d j: %d expected: %f actual: %f\n",
            i,j,a[i*NDIM+j]*va[j],b[i*NDIM+j]);
        ok = 0;
      }
    }
  }
  GA_Igop(&ok,ONE,"*");
  
  if (me == 0) {
    if (ok) {
      printf("\n  GA_Scale_cols OK\n");
    } else {
      printf("\n  GA_Scale_cols not OK\n");
    }
  }

  NGA_Destroy(g_a);
  NGA_Destroy(g_v);
}

int main(int argc, char **argv)
{
int heap=20000, stack=20000;
int me, nproc;

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
    
    do_work();

    if(me==0)printf("Terminating ..\n");
    GA_Terminate();

    MP_FINALIZE();

    return 0;
}

