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

#define NDIM 5             /* dimension of matrices */
#define LDIM NDIM*NDIM       /* dimension of vectors */


/* Test the norm1 and norm_infinity functions on 1 and 2 dimensional global
 * arrays */
void do_work()
{
  int ONE=1, TWO=2;   /* useful constants */
  int g_a;
  int me=GA_Nodeid(), nproc=GA_Nnodes();
  int i, j, m, n, ndim, dim1, dim2, ld;
  int dims[2];
  int lo[2], hi[2];
  double rnorm_infinity, rnorm1, norm;
  int *iptr;
  long *lptr;
  float *fptr;
  double *dptr;
  DoubleComplex *zptr;
  SingleComplex *cptr;
  double delta;

  /* Test vectors */
  ndim = ONE;
  dims[0] = LDIM;
  rnorm1 = (double)(LDIM*(LDIM-1)/2);
  rnorm_infinity = (double)(LDIM-1);

  /* Test integers */
  if (me==0) printf("Testing integer vector\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_INT);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  NGA_Access(g_a, lo, hi, &iptr, &ld);
  for (i=0; i<dim1; i++) {
    n = -(i+lo[0]);
    iptr[i] = n;
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm_infinity) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails\n");
    }
  }
  GA_Norm1(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm1) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails\n");
    }
  }
  NGA_Destroy(g_a);
  
  /* Test long integers */
  if (me==0) printf("Testing long integer vector\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_LONG);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  NGA_Access(g_a, lo, hi, &lptr, &ld);
  for (i=0; i<dim1; i++) {
    n = -(i+lo[0]);
    lptr[i] = (long)n;
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm_infinity) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails\n");
    }
  }
  GA_Norm1(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm1) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails\n");
    }
  }
  NGA_Destroy(g_a);
  
  /* Test floats */
  if (me==0) printf("Testing float vector\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_FLOAT);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  NGA_Access(g_a, lo, hi, &fptr, &ld);
  for (i=0; i<dim1; i++) {
    n = -(i+lo[0]);
    fptr[i] = (float)n;
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  delta = fabs(norm-rnorm_infinity)/rnorm_infinity;
  if (me == 0) {
    if (delta<1.0e-6) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails expected: %f actual: %f\n",
          rnorm_infinity,norm);
    }
  }
  GA_Norm1(g_a, &norm);
  delta = fabs(norm-rnorm1)/rnorm1;
  /* strong roundoff error for this test (I think) */
  if (me == 0) {
    if (delta<1.0e-4) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails expected: %f actual: %f\n",rnorm1,norm);
    }
  }
  NGA_Destroy(g_a);
  
  /* Test doubles */
  if (me==0) printf("Testing double vector\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_DBL);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  NGA_Access(g_a, lo, hi, &dptr, &ld);
  for (i=0; i<dim1; i++) {
    n = -(i+lo[0]);
    dptr[i] = (double)n;
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  delta = fabs(norm-rnorm_infinity)/rnorm_infinity;
  if (me == 0) {
    if (delta<1.0e-12) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails expected: %f actual: %f\n",
          rnorm_infinity,norm);
    }
  }
  GA_Norm1(g_a, &norm);
  delta = fabs(norm-rnorm1)/rnorm1;
  if (me == 0) {
    if (delta<1.0e-12) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails expected: %f actual: %f\n",rnorm1,norm);
    }
  }
  NGA_Destroy(g_a);
  
  /* Test single complex */
  if (me==0) printf("Testing single complex vector\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_SCPL);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  NGA_Access(g_a, lo, hi, &cptr, &ld);
  for (i=0; i<dim1; i++) {
    n = -(i+lo[0]);
    cptr[i].real = (float)n;
    cptr[i].imag = 0.0;
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  delta = fabs(norm-rnorm_infinity)/rnorm_infinity;
  if (me == 0) {
    if (delta<1.0e-6) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails expected: %f actual: %f\n",
          rnorm_infinity,norm);
    }
  }
  GA_Norm1(g_a, &norm);
  delta = fabs(norm-rnorm1)/rnorm1;
  if (me == 0) {
    if (delta<1.0e-5) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails expected: %f actual: %f\n",rnorm1,norm);
    }
  }
  NGA_Destroy(g_a);

  /* Test double complex */
  if (me==0) printf("Testing double complex vector\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_DCPL);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  NGA_Access(g_a, lo, hi, &zptr, &ld);
  for (i=0; i<dim1; i++) {
    n = -(i+lo[0]);
    zptr[i].real = (double)n;
    zptr[i].imag = 0.0;
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm_infinity) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails expected: %f actual: %f\n",
          rnorm_infinity,norm);
    }
  }
  GA_Norm1(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm1) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails expected: %f actual: %f\n",rnorm1,norm);
    }
  }
  NGA_Destroy(g_a);

  /* Test matrices */
  ndim = TWO;
  dims[0] = NDIM;
  dims[1] = NDIM;
  rnorm1 = (double)(LDIM*(LDIM-1)/2);
  rnorm_infinity = (double)(LDIM-1);

  /* Test integers */
  if (me==0) printf("Testing integer matrix\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_INT);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  dim2 = hi[1]-lo[1]+1;
  NGA_Access(g_a, lo, hi, &iptr, &ld);
  for (j=0; j<dim2; j++) {
    for (i=0; i<dim1; i++) {
      m = i + dim1*j;
      n = -((i+lo[0])+(j+lo[1])*dims[0]);
      iptr[m] = n;
    }
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm_infinity) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails expected: %f actual: %f\n",
          rnorm_infinity,norm);
    }
  }
  GA_Norm1(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm1) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails expected: %f actual: %f\n",rnorm1,norm);
    }
  }
  NGA_Destroy(g_a);
  
  /* Test long integers */
  if (me==0) printf("Testing long integer matrix\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_LONG);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  dim2 = hi[1]-lo[1]+1;
  NGA_Access(g_a, lo, hi, &lptr, &ld);
  for (j=0; j<dim2; j++) {
    for (i=0; i<dim1; i++) {
      m = i + dim1*j;
      n = -((i+lo[0])+(j+lo[1])*dims[0]);
      lptr[m] = (long)n;
    }
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm_infinity) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails\n");
    }
  }
  GA_Norm1(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm1) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails\n");
    }
  }
  NGA_Destroy(g_a);
  
  /* Test floats */
  if (me==0) printf("Testing float matrix\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_FLOAT);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  dim2 = hi[1]-lo[1]+1;
  NGA_Access(g_a, lo, hi, &fptr, &ld);
  for (j=0; j<dim2; j++) {
    for (i=0; i<dim1; i++) {
      m = i + dim1*j;
      n = -((i+lo[0])+(j+lo[1])*dims[0]);
      fptr[m] = (float)n;
    }
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  delta = fabs(norm-rnorm_infinity)/rnorm_infinity;
  if (me == 0) {
    if (delta<1.0e-6) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails expected: %f actual: %f\n",
          rnorm_infinity,norm);
    }
  }
  GA_Norm1(g_a, &norm);
  delta = fabs(norm-rnorm1)/rnorm1;
  /* strong roundoff error for this test (I think) */
  if (me == 0) {
    if (delta<1.0e-4) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails expected: %f actual: %f\n",rnorm1,norm);
    }
  }
  NGA_Destroy(g_a);
  
  /* Test doubles */
  if (me==0) printf("Testing double vector\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_DBL);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  dim2 = hi[1]-lo[1]+1;
  NGA_Access(g_a, lo, hi, &dptr, &ld);
  for (j=0; j<dim2; j++) {
    for (i=0; i<dim1; i++) {
      m = i + dim1*j;
      n = -((i+lo[0])+(j+lo[1])*dims[0]);
      dptr[m] = (double)n;
    }
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  delta = fabs(norm-rnorm_infinity)/rnorm_infinity;
  if (me == 0) {
    if (delta<1.0e-12) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails expected: %f actual: %f\n",
          rnorm_infinity,norm);
    }
  }
  GA_Norm1(g_a, &norm);
  delta = fabs(norm-rnorm1)/rnorm1;
  if (me == 0) {
    if (delta<1.0e-12) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails expected: %f actual: %f\n",rnorm1,norm);
    }
  }
  NGA_Destroy(g_a);
  
  /* Test single complex */
  if (me==0) printf("Testing single complex vector\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_SCPL);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  dim2 = hi[1]-lo[1]+1;
  NGA_Access(g_a, lo, hi, &cptr, &ld);
  for (j=0; j<dim2; j++) {
    for (i=0; i<dim1; i++) {
      m = i + dim1*j;
      n = -((i+lo[0])+(j+lo[1])*dims[0]);
      cptr[m].real = (float)n;
      cptr[m].imag = 0.0;
    }
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  delta = fabs(norm-rnorm_infinity)/rnorm_infinity;
  if (me == 0) {
    if (delta<1.0e-6) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails expected: %f actual: %f\n",
          rnorm_infinity,norm);
    }
  }
  GA_Norm1(g_a, &norm);
  delta = fabs(norm-rnorm1)/rnorm1;
  if (me == 0) {
    if (delta<1.0e-5) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails expected: %f actual: %f\n",rnorm1,norm);
    }
  }
  NGA_Destroy(g_a);

  /* Test double complex */
  if (me==0) printf("Testing double complex vector\n");
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a, ndim, dims, C_DCPL);
  NGA_Allocate(g_a);
  NGA_Distribution(g_a, me, lo, hi);
  dim1 = hi[0]-lo[0]+1;
  dim2 = hi[1]-lo[1]+1;
  NGA_Access(g_a, lo, hi, &zptr, &ld);
  for (j=0; j<dim2; j++) {
    for (i=0; i<dim1; i++) {
      m = i + dim1*j;
      n = -((i+lo[0])+(j+lo[1])*dims[0]);
      zptr[m].real = (double)n;
      zptr[m].imag = 0.0;
    }
  }
  NGA_Release(g_a, lo, hi);
  GA_Norm_infinity(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm_infinity) {
      printf("\n   Norm infinity ok\n");
    } else {
      printf("\n   Norm infinity fails expected: %f actual: %f\n",
          rnorm_infinity,norm);
    }
  }
  GA_Norm1(g_a, &norm);
  if (me == 0) {
    if (norm == rnorm1) {
      printf("\n   Norm1  ok\n");
    } else {
      printf("\n   Norm1 fails expected: %f actual: %f\n",rnorm1,norm);
    }
  }
  NGA_Destroy(g_a);
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

    if (me == 0) printf("All tests successful\n");
    GA_Terminate();

    MP_FINALIZE();

    return 0;
}

