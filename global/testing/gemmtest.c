#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

/* utilities for GA test programs */
#include "testutil.h"

#define N 29          /* first dimension  */

#define GA_DATA_TYPE MT_C_FLOAT
#define GA_ABS(a) (((a) >= 0) ? (a) : (-(a)))
#define TOLERANCE 0.000001


DoublePrecision gTime=0.0, gStart;

void
test(int data_type) {
  int me=GA_Nodeid();
  int g_a, g_b, g_c;
  int ndim = 2;
  int dims[2]={N,N};
  int lo[2]={0,0};
  int hi[2]={N-1,N-1};
  int i,j,l,k,m,n, ld;

  double alpha_dbl = 1.0, beta_dbl = 0.0;
  double dzero = 0.0;
  double ddiff;

  float alpha_flt = 1.0, beta_flt = 0.0;
  float fzero = 0.0;
  float fdiff;

  DoubleComplex alpha_dcpl = {1.0, 0.0} , beta_dcpl = {0.0, 0.0}; 
  DoubleComplex zzero = {0.0,0.0};
  DoubleComplex zdiff;

  SingleComplex alpha_scpl = {1.0, 0.0} , beta_scpl = {0.0, 0.0}; 
  SingleComplex czero = {0.0,0.0};
  SingleComplex cdiff;

  void *alpha=NULL, *beta=NULL;
  void *abuf=NULL, *bbuf=NULL, *cbuf=NULL, *c_ptr=NULL;

  switch (data_type) {
  case C_FLOAT:
    alpha  = (void *)&alpha_flt;
    beta   = (void *)&beta_flt;
    abuf = (void*)malloc(N*N*sizeof(float));
    bbuf = (void*)malloc(N*N*sizeof(float));
    cbuf = (void*)malloc(N*N*sizeof(float));
    if(me==0) printf("Single Precision: Testing GA_Sgemm,NGA_Matmul_patch for %d-Dimension", ndim);
    break;      
  case C_DBL:
    alpha  = (void *)&alpha_dbl;
    beta   = (void *)&beta_dbl;
    abuf = (void*)malloc(N*N*sizeof(double));
    bbuf = (void*)malloc(N*N*sizeof(double));
    cbuf = (void*)malloc(N*N*sizeof(double));
    if(me==0) printf("Double Precision: Testing GA_Dgemm,NGA_Matmul_patch for %d-Dimension", ndim); 
    break;    
  case C_DCPL:
    alpha  = (void *)&alpha_dcpl;
    beta   = (void *)&beta_dcpl;
    abuf = (void*)malloc(N*N*sizeof(DoubleComplex));
    bbuf = (void*)malloc(N*N*sizeof(DoubleComplex));
    cbuf = (void*)malloc(N*N*sizeof(DoubleComplex));
    if(me==0) printf("Double Complex:   Testing GA_Zgemm,NGA_Matmul_patch for %d-Dimension", ndim);
    break;
  case C_SCPL:
    alpha  = (void *)&alpha_scpl;
    beta   = (void *)&beta_scpl;
    abuf = (void*)malloc(N*N*sizeof(SingleComplex));
    bbuf = (void*)malloc(N*N*sizeof(SingleComplex));
    cbuf = (void*)malloc(N*N*sizeof(SingleComplex));
    if(me==0) printf("Single Complex:   Testing GA_Cgemm,NGA_Matmul_patch for %d-Dimension", ndim);
    break;
  default:
    GA_Error("wrong data type", data_type);
  }

  if (me==0) printf("\nCreate A, B, C\n");
  g_a = NGA_Create(data_type, ndim, dims, "array A", NULL);
  g_b = GA_Duplicate(g_a, "array B");  
  g_c = GA_Duplicate(g_a, "array C");
  if(!g_a || !g_b || !g_c) GA_Error("Create failed: a, b or c",1);

  ld = N;
  if (me==0) printf("\nInitialize A\n");
  /* Set up matrix A */
  if (me == 0) {
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        switch (data_type) {
          case C_FLOAT:
            ((float*)abuf)[i*N+j] = (float)(i*N+j);
            break;
          case C_DBL:
            ((double*)abuf)[i*N+j] = (double)(i*N+j);
            break;
          case C_DCPL:
            ((DoubleComplex*)abuf)[i*N+j].real = (double)(i*N+j);
            ((DoubleComplex*)abuf)[i*N+j].imag = 1.0;
            break;
          case C_SCPL:
            ((SingleComplex*)abuf)[i*N+j].real = (float)(i*N+j);
            ((SingleComplex*)abuf)[i*N+j].imag = 1.0;
            break;
          default:
            GA_Error("wrong data type", data_type);
        }
      }
    }
    NGA_Put(g_a,lo,hi,abuf,&ld);
  }
  GA_Sync();

  if (me==0) printf("\nInitialize B\n");
  /* Set up matrix B */
  if (me == 0) {
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        switch (data_type) {
          case C_FLOAT:
            ((float*)bbuf)[i*N+j] = (float)(j*N+i);
            break;
          case C_DBL:
            ((double*)bbuf)[i*N+j] = (double)(j*N+i);
            break;
          case C_DCPL:
            ((DoubleComplex*)bbuf)[i*N+j].real = (double)(j*N+i);
            ((DoubleComplex*)bbuf)[i*N+j].imag = 1.0;
            break;
          case C_SCPL:
            ((SingleComplex*)bbuf)[i*N+j].real = (float)(j*N+i);
            ((SingleComplex*)bbuf)[i*N+j].imag = 1.0;
            break;
          default:
            GA_Error("wrong data type", data_type);
        }
      }
    }
    NGA_Put(g_b,lo,hi,bbuf,&ld);
  }
  GA_Sync();

  if (me==0) printf("\nPerform matrix multiply\n");
  switch (data_type) {
    case C_FLOAT:
      GA_Sgemm('N','N',N,N,N,alpha_flt,g_a,g_b,beta_flt,g_c);
      break;
    case C_DBL:
      GA_Dgemm('N','N',N,N,N,alpha_dbl,g_a,g_b,beta_dbl,g_c);
      break;
    case C_SCPL:
      GA_Cgemm('N','N',N,N,N,alpha_scpl,g_a,g_b,beta_scpl,g_c);
      break;
    case C_DCPL:
      GA_Zgemm('N','N',N,N,N,alpha_dcpl,g_a,g_b,beta_dcpl,g_c);
      break;
    default:
      GA_Error("wrong data type", data_type);
  }
  GA_Sync();
  if (me==0) printf("\nCheck answer\n");

  /* Check answer */
  NGA_Get(g_a,lo,hi,abuf,&ld);
  NGA_Get(g_b,lo,hi,bbuf,&ld);
  for (i=0; i<N; i++) {
    for (j=0; j<N; j++) {
      switch (data_type) {
        case C_FLOAT:
          ((float*)cbuf)[i*N+j] = fzero;
          break;
        case C_DBL:
          ((double*)cbuf)[i*N+j] = dzero;
          break;
        case C_DCPL:
          ((DoubleComplex*)cbuf)[i*N+j] = zzero;
          break;
        case C_SCPL:
          ((SingleComplex*)cbuf)[i*N+j] = czero;
          break;
        default:
          GA_Error("wrong data type", data_type);
      }
      for (k=0; k<N; k++) {
        switch (data_type) {
          case C_FLOAT:
            ((float*)cbuf)[i*N+j] += ((float*)abuf)[i*N+k]
              *((float*)bbuf)[k*N+j];
            break;
          case C_DBL:
            ((double*)cbuf)[i*N+j] += ((double*)abuf)[i*N+k]
              *((double*)bbuf)[k*N+j];
            break;
          case C_DCPL:
            ((DoubleComplex*)cbuf)[i*N+j].real +=
              (((DoubleComplex*)abuf)[i*N+k].real
               *((DoubleComplex*)bbuf)[k*N+j].real
               -(((DoubleComplex*)abuf)[i*N+k].imag
                 *((DoubleComplex*)bbuf)[k*N+j].imag));
            ((DoubleComplex*)cbuf)[i*N+j].imag +=
              (((DoubleComplex*)abuf)[i*N+k].real
               *((DoubleComplex*)bbuf)[k*N+j].imag
               +(((DoubleComplex*)abuf)[i*N+k].imag
                 *((DoubleComplex*)bbuf)[k*N+j].real));
            break;
          case C_SCPL:
            ((SingleComplex*)cbuf)[i*N+j].real +=
              (((SingleComplex*)abuf)[i*N+k].real
               *((SingleComplex*)bbuf)[k*N+j].real
               -(((SingleComplex*)abuf)[i*N+k].imag
                 *((SingleComplex*)bbuf)[k*N+j].imag));
            ((SingleComplex*)cbuf)[i*N+j].imag +=
              (((SingleComplex*)abuf)[i*N+k].real
               *((SingleComplex*)bbuf)[k*N+j].imag
               +(((SingleComplex*)abuf)[i*N+k].imag
                 *((SingleComplex*)bbuf)[k*N+j].real));
            break;
          default:
            GA_Error("wrong data type", data_type);
        }
      }
    }
  }
  GA_Sync();
  /*
  GA_Print(g_a);
  if (me == 0) printf("\n\n\n\n");
  GA_Print(g_c);
  if (me == 0) printf("\n\n\n\n");
  GA_Print(g_b); 
  */
  if (me == 0) {
    NGA_Get(g_c,lo,hi,abuf,&ld);
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        switch (data_type) {
          case C_FLOAT:
            fdiff = ((float*)abuf)[i*N+j]-((float*)cbuf)[i*N+j];
            if (fabs(fdiff) > TOLERANCE) {
              printf("p[%d] [%d,%d] Actual: %f Expected: %f\n",me,i,j,
                  ((float*)abuf)[i*N+j],((float*)cbuf)[i*N+j]);
            }
            break;
          case C_DBL:
            ddiff = ((double*)abuf)[i*N+j]-((double*)cbuf)[i*N+j];
            if (fabs(ddiff) > TOLERANCE) {
              printf("p[%d] [%d,%d] Actual: %f Expected: %f\n",me,i,j,
                  ((double*)abuf)[i*N+j],((double*)cbuf)[i*N+j]);
            }
            break;
          case C_DCPL:
            zdiff.real = ((DoubleComplex*)abuf)[i*N+j].real
              -((DoubleComplex*)cbuf)[i*N+j].real;
            zdiff.imag = ((DoubleComplex*)abuf)[i*N+j].imag
              -((DoubleComplex*)cbuf)[i*N+j].imag;
            if (fabs(zdiff.real) > TOLERANCE) {
              printf("p[%d] [%d,%d] Actual: (%f,%f) Expected: (%f,%f)\n",me,i,j,
                  ((DoubleComplex*)abuf)[i*N+j].real,
                  ((DoubleComplex*)abuf)[i*N+j].imag,
                  ((DoubleComplex*)cbuf)[i*N+j].real,
                  ((DoubleComplex*)cbuf)[i*N+j].imag);
            }
            break;
          case C_SCPL:
            cdiff.real = ((SingleComplex*)abuf)[i*N+j].real
              -((SingleComplex*)cbuf)[i*N+j].real;
            cdiff.imag = ((SingleComplex*)abuf)[i*N+j].imag
              -((SingleComplex*)cbuf)[i*N+j].imag;
            if (fabs(cdiff.real) > TOLERANCE) {
              printf("p[%d] [%d,%d] Actual: (%f,%f) Expected: (%f,%f)\n",me,i,j,
                  ((SingleComplex*)abuf)[i*N+j].real,
                  ((SingleComplex*)abuf)[i*N+j].imag,
                  ((SingleComplex*)cbuf)[i*N+j].real,
                  ((SingleComplex*)cbuf)[i*N+j].imag);
            }
            break;
          default:
            GA_Error("wrong data type", data_type);
        }
      }
    }
  }
  GA_Sync();

  /* copy cbuf back to g_a */
  if (me == 0) {
    NGA_Put(g_a,lo,hi,cbuf,&ld);
  }
  GA_Sync();

  /* subtract C from A and put the results in B */
  beta_flt = -1.0;
  beta_dbl = -1.0;
  beta_scpl.real = -1.0;
  beta_dcpl.real = -1.0;
  GA_Zero(g_b);
  GA_Add(alpha,g_a,beta,g_c,g_b);
  /* evaluate the norm of the difference between the two matrices */
  switch (data_type) {
    case C_FLOAT:
      fdiff = GA_Fdot(g_b, g_b);
      if(fabs(fdiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(fdiff), TOLERANCE);
        GA_Error("GA_Sgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Sgemm OK\n\n");
      }
      break;
    case C_DBL:
      ddiff = GA_Ddot(g_b, g_b);
      if(fabs(ddiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(ddiff), TOLERANCE);
        GA_Error("GA_Dgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Dgemm OK\n\n");
      }
      break;
    case C_DCPL:
      zdiff = GA_Zdot(g_b, g_b);
      if(fabs(zdiff.real) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(zdiff.real), TOLERANCE);
        GA_Error("GA_Zgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Zgemm OK\n\n");
      }
      break;
    case C_SCPL:
      cdiff = GA_Cdot(g_b, g_b);
      if(fabs(cdiff.real) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(cdiff.real), TOLERANCE);
        GA_Error("GA_Cgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Cgemm OK\n\n");
      }
      break;
    default:
      GA_Error("wrong data type", data_type);
  }

  free(abuf);
  free(bbuf);
  free(cbuf);


  GA_Destroy(g_a);
  GA_Destroy(g_b);
  GA_Destroy(g_c);
}

void
do_work() {
  int i;
  int me = GA_Nodeid();

  test(C_FLOAT);
  test(C_DBL);
  test(C_DCPL);
/*  test(C_SCPL); */
  if(me == 0) printf("\n\n");
}
     

int 
main(int argc, char **argv) {

Integer heap=9000000, stack=9000000;
int me, nproc;
DoublePrecision time;

    MP_INIT(argc,argv);

    GA_INIT(argc,argv);                           /* initialize GA */

    nproc = GA_Nnodes();
    me = GA_Nodeid();

    if(me==0) printf("Using %d processes\n\n",nproc);

    if (me==0) printf ("Matrix size is %d X %d\n",N,N);

    if(!MA_init((Integer)MT_F_DBL, stack/nproc, heap/nproc))
       GA_Error("MA_init failed bytes= %d",stack+heap);   

#ifdef PERMUTE
      {
        int i, *list = (int*)malloc(nproc*sizeof(int));
        if(!list)GA_Error("malloc failed",nproc);

        for(i=0; i<nproc;i++)list[i]=nproc-1-i;

        GA_Register_proclist(list, nproc);
        free(list);
      }
#endif

    if(GA_Uses_fapi())GA_Error("Program runs with C API only",1);

    time = MP_TIMER();
    do_work();
    /*    printf("%d: Total Time = %lf\n", me, MP_TIMER()-time);
      printf("%d: GEMM Total Time = %lf\n", me, gTime);
    */

    if(me==0)printf("\nSuccess\n\n");
    GA_Terminate();

    MP_FINALIZE();

    return 0;
}

