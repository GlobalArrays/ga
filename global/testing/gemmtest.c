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

#define N 8          /* first dimension  */

#define NB 2          /* block dimension */

#define GA_DATA_TYPE MT_C_FLOAT
#define GA_ABS(a) (((a) >= 0) ? (a) : (-(a)))
#define TOLERANCE 0.0001

#define USE_REGULAR
/*
#define USE_SIMPLE_CYCLIC
#define USE_SCALAPACK
#define USE_TILED
*/

DoublePrecision gTime=0.0, gStart;

#define MAX_FACTOR 512
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

void
test(int data_type) {
  int me=GA_Nodeid();
  int nproc = GA_Nnodes();
  int g_a, g_b, g_c;
  int ndim = 2;
  int dims[2]={N,N};
  int lo[2]={0,0};
  int hi[2]={N-1,N-1};
  int block_size[2]={NB,NB-1};
  int proc_grid[2];
  int i,j,l,k,m,n, ld;

  double alpha_dbl = 1.0, beta_dbl = 0.0;
  double dzero = 0.0;
  double ddiff;

  float alpha_flt = 1.0, beta_flt = 0.0;
  float fzero = 0.0;
  float fdiff;
  float ftmp;
  double dtmp;
  SingleComplex ctmp;
  DoubleComplex ztmp;

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
#ifdef USE_REGULAR
  g_a = NGA_Create(data_type, ndim, dims, "array A", NULL);
#endif
#ifdef USE_SIMPLE_CYCLIC
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a,ndim,dims,data_type);
  NGA_Set_array_name(g_a,"array A");
  NGA_Set_block_cyclic(g_a,block_size);
  if (!GA_Allocate(g_a)) {
    GA_Error("Failed: create: g_a",40);
  }
#endif
#ifdef USE_SCALAPACK
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a,ndim,dims,data_type);
  NGA_Set_array_name(g_a,"array A");
  grid_factor(nproc,&i,&j);
  proc_grid[0] = i;
  proc_grid[1] = j;
  NGA_Set_block_cyclic_proc_grid(g_a,block_size,proc_grid);
  if (!GA_Allocate(g_a)) {
    GA_Error("Failed: create: g_a",40);
  }
#endif
#ifdef USE_TILED
  g_a = NGA_Create_handle();
  NGA_Set_data(g_a,ndim,dims,data_type);
  NGA_Set_array_name(g_a,"array A");
  grid_factor(nproc,&i,&j);
  proc_grid[0] = i;
  proc_grid[1] = j;
  NGA_Set_tiled_proc_grid(g_a,block_size,proc_grid);
  if (!GA_Allocate(g_a)) {
    GA_Error("Failed: create: g_a",40);
  }
#endif
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
      NGA_Matmul_patch('N','N',&alpha_flt,&beta_flt,g_a,lo,hi,
        g_b,lo,hi,g_c,lo,hi);
      break;
    case C_DBL:
      NGA_Matmul_patch('N','N',&alpha_dbl,&beta_dbl,g_a,lo,hi,
        g_b,lo,hi,g_c,lo,hi);
      break;
    case C_SCPL:
      NGA_Matmul_patch('N','N',&alpha_scpl,&beta_scpl,g_a,lo,hi,
        g_b,lo,hi,g_c,lo,hi);
      break;
    case C_DCPL:
      NGA_Matmul_patch('N','N',&alpha_dcpl,&beta_dcpl,g_a,lo,hi,
        g_b,lo,hi,g_c,lo,hi);
      break;
    default:
      GA_Error("wrong data type", data_type);
  }
  GA_Sync();
#if 0
  if (me==0) printf("\nCheck answer\n");
  /*
  GA_Print(g_a);
  if (me == 0) printf("\n\n\n\n");
  GA_Print(g_b);
  if (me == 0) printf("\n\n\n\n");
  GA_Print(g_c); 
  */

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
  if (me == 0) {
    NGA_Get(g_c,lo,hi,abuf,&ld);
    for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
        switch (data_type) {
          case C_FLOAT:
            fdiff = ((float*)abuf)[i*N+j]-((float*)cbuf)[i*N+j];
            if (((float*)abuf)[i*N+j] != 0.0) {
              fdiff /= ((float*)abuf)[i*N+j];
            }
            if (fabs(fdiff) > TOLERANCE) {
              printf("p[%d] [%d,%d] Actual: %f Expected: %f\n",me,i,j,
                  ((float*)abuf)[i*N+j],((float*)cbuf)[i*N+j]);
            }
            break;
          case C_DBL:
            ddiff = ((double*)abuf)[i*N+j]-((double*)cbuf)[i*N+j];
            if (((double*)abuf)[i*N+j] != 0.0) {
              ddiff /= ((double*)abuf)[i*N+j];
            }
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
            if (((DoubleComplex*)abuf)[i*N+j].real != 0.0 ||
                ((DoubleComplex*)abuf)[i*N+j].imag != 0.0) {
              ztmp = ((DoubleComplex*)abuf)[i*N+j];
              ddiff = sqrt((zdiff.real*zdiff.real+zdiff.imag*zdiff.imag)
                  /(ztmp.real*ztmp.real+ztmp.imag*ztmp.imag));
            } else {
              ddiff = sqrt(zdiff.real*zdiff.real+zdiff.imag*zdiff.imag);
            }
            if (fabs(ddiff) > TOLERANCE) {
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
            if (((SingleComplex*)abuf)[i*N+j].real != 0.0 ||
                ((SingleComplex*)abuf)[i*N+j].imag != 0.0) {
              ctmp = ((SingleComplex*)abuf)[i*N+j];
              fdiff = sqrt((cdiff.real*cdiff.real+cdiff.imag*cdiff.imag)
                  /(ctmp.real*ctmp.real+ctmp.imag*ctmp.imag));
            } else {
              fdiff = sqrt(cdiff.real*cdiff.real+cdiff.imag*cdiff.imag);
            }
            if (fabs(fdiff) > TOLERANCE) {
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

  /* Get norm of g_a */
  switch (data_type) {
    case C_FLOAT:
      ftmp = GA_Fdot(g_a,g_a);
      break;
    case C_DBL:
      dtmp = GA_Ddot(g_a,g_a);
      break;
    case C_DCPL:
      ztmp = GA_Zdot(g_a,g_a);
      break;
    case C_SCPL:
      ctmp = GA_Cdot(g_a,g_a);
      break;
    default:
      GA_Error("wrong data type", data_type);
  }
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
      if (ftmp != 0.0) {
        fdiff /= ftmp;
      }
      if(fabs(fdiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(fdiff), TOLERANCE);
        GA_Error("GA_Sgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Sgemm OK\n\n");
      }
      break;
    case C_DBL:
      ddiff = GA_Ddot(g_b, g_b);
      if (dtmp != 0.0) {
        ddiff /= dtmp;
      }
      if(fabs(ddiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(ddiff), TOLERANCE);
        GA_Error("GA_Dgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Dgemm OK\n\n");
      }
      break;
    case C_DCPL:
      zdiff = GA_Zdot(g_b, g_b);
      if (ztmp.real != 0.0 || ztmp.imag != 0.0) {
        ddiff = sqrt((zdiff.real*zdiff.real+zdiff.imag*zdiff.imag)
            /(ztmp.real*ztmp.real+ztmp.imag*ztmp.imag));
      } else {
        ddiff = sqrt(zdiff.real*zdiff.real+zdiff.imag*zdiff.imag);
      }
      if(fabs(ddiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(zdiff.real), TOLERANCE);
        GA_Error("GA_Zgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Zgemm OK\n\n");
      }
      break;
    case C_SCPL:
      cdiff = GA_Cdot(g_b, g_b);
      if (ctmp.real != 0.0 || ctmp.imag != 0.0) {
        fdiff = sqrt((cdiff.real*cdiff.real+cdiff.imag*cdiff.imag)
            /(ctmp.real*ctmp.real+ctmp.imag*ctmp.imag));
      } else {
        fdiff = sqrt(cdiff.real*cdiff.real+cdiff.imag*cdiff.imag);
      }
      if(fabs(fdiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(cdiff.real), TOLERANCE);
        GA_Error("GA_Cgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Cgemm OK\n\n");
      }
      break;
    default:
      GA_Error("wrong data type", data_type);
  }
#endif

  free(abuf);
  free(bbuf);
  free(cbuf);

  switch (data_type) {
  case C_FLOAT:
    abuf = (void*)malloc(N*N*sizeof(float)/4);
    bbuf = (void*)malloc(N*N*sizeof(float)/4);
    cbuf = (void*)malloc(N*N*sizeof(float)/4);
    break;      
  case C_DBL:
    abuf = (void*)malloc(N*N*sizeof(double)/4);
    bbuf = (void*)malloc(N*N*sizeof(double)/4);
    cbuf = (void*)malloc(N*N*sizeof(double)/4);
    break;    
  case C_DCPL:
    abuf = (void*)malloc(N*N*sizeof(DoubleComplex)/4);
    bbuf = (void*)malloc(N*N*sizeof(DoubleComplex)/4);
    cbuf = (void*)malloc(N*N*sizeof(DoubleComplex)/4);
    break;
  case C_SCPL:
    abuf = (void*)malloc(N*N*sizeof(SingleComplex)/4);
    bbuf = (void*)malloc(N*N*sizeof(SingleComplex)/4);
    cbuf = (void*)malloc(N*N*sizeof(SingleComplex)/4);
    break;
  default:
    GA_Error("wrong data type", data_type);
  }

  /* Test multiply on a fraction of matrix. Start by reinitializing
   * A and B */
  GA_Zero(g_a);
  GA_Zero(g_b);
  GA_Zero(g_c);

  if (me==0) printf("\nTest patch multiply\n");

  lo[0] = N/4;
  lo[1] = N/4;
  hi[0] = 3*N/4-1;
  hi[1] = 3*N/4-1;
  ld = N/2;

  /* Set up matrix A */
  if (me==0) printf("\nInitialize A\n");
  if (me == 0) {
    for (i=N/4; i<3*N/4; i++) {
      for (j=N/4; j<3*N/4; j++) {
        switch (data_type) {
          case C_FLOAT:
            ((float*)abuf)[(i-N/4)*N/2+(j-N/4)] = (float)(i*N+j);
            break;
          case C_DBL:
            ((double*)abuf)[(i-N/4)*N/2+(j-N/4)] = (double)(i*N+j);
            break;
          case C_DCPL:
            ((DoubleComplex*)abuf)[(i-N/4)*N/2+(j-N/4)].real = (double)(i*N+j);
            ((DoubleComplex*)abuf)[(i-N/4)*N/2+(j-N/4)].imag = 1.0;
            break;
          case C_SCPL:
            ((SingleComplex*)abuf)[(i-N/4)*N/2+(j-N/4)].real = (float)(i*N+j);
            ((SingleComplex*)abuf)[(i-N/4)*N/2+(j-N/4)].imag = 1.0;
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
    for (i=N/4; i<3*N/4; i++) {
      for (j=N/4; j<3*N/4; j++) {
        switch (data_type) {
          case C_FLOAT:
            ((float*)bbuf)[(i-N/4)*N/2+(j-N/4)] = (float)(j*N+i);
            break;
          case C_DBL:
            ((double*)bbuf)[(i-N/4)*N/2+(j-N/4)] = (double)(j*N+i);
            break;
          case C_DCPL:
            ((DoubleComplex*)bbuf)[(i-N/4)*N/2+(j-N/4)].real = (double)(j*N+i);
            ((DoubleComplex*)bbuf)[(i-N/4)*N/2+(j-N/4)].imag = 1.0;
            break;
          case C_SCPL:
            ((SingleComplex*)bbuf)[(i-N/4)*N/2+(j-N/4)].real = (float)(j*N+i);
            ((SingleComplex*)bbuf)[(i-N/4)*N/2+(j-N/4)].imag = 1.0;
            break;
          default:
            GA_Error("wrong data type", data_type);
        }
      }
    }
    NGA_Put(g_b,lo,hi,bbuf,&ld);
  }
  GA_Sync();

  beta_flt = 0.0;
  beta_dbl = 0.0;
  beta_scpl.real = 0.0;
  beta_dcpl.real = 0.0;
  if (me==0) printf("\nPerform matrix multiply on sub-blocks\n");
  switch (data_type) {
    case C_FLOAT:
      NGA_Matmul_patch('N','N',&alpha_flt,&beta_flt,g_a,lo,hi,
        g_b,lo,hi,g_c,lo,hi);
      break;
    case C_DBL:
      NGA_Matmul_patch('N','N',&alpha_dbl,&beta_dbl,g_a,lo,hi,
        g_b,lo,hi,g_c,lo,hi);
      break;
    case C_SCPL:
      NGA_Matmul_patch('N','N',&alpha_scpl,&beta_scpl,g_a,lo,hi,
        g_b,lo,hi,g_c,lo,hi);
      break;
    case C_DCPL:
      NGA_Matmul_patch('N','N',&alpha_dcpl,&beta_dcpl,g_a,lo,hi,
        g_b,lo,hi,g_c,lo,hi);
      break;
    default:
      GA_Error("wrong data type", data_type);
  }
  GA_Sync();
#if 0
  if (0) {
  /*
  if (data_type != C_SCPL && data_type != C_DCPL) {
  */

  if (me==0) printf("\nCheck answer\n");

  /* Multiply buffers by hand */
  if (me == 0) {
    for (i=0; i<N/2; i++) {
      for (j=0; j<N/2; j++) {
        switch (data_type) {
          case C_FLOAT:
            ((float*)cbuf)[i*N/2+j] = fzero;
            break;
          case C_DBL:
            ((double*)cbuf)[i*N/2+j] = dzero;
            break;
          case C_DCPL:
            ((DoubleComplex*)cbuf)[i*N/2+j] = zzero;
            break;
          case C_SCPL:
            ((SingleComplex*)cbuf)[i*N/2+j] = czero;
            break;
          default:
            GA_Error("wrong data type", data_type);
        }
        for (k=0; k<N/2; k++) {
          switch (data_type) {
            case C_FLOAT:
              ((float*)cbuf)[i*N/2+j] += ((float*)abuf)[i*N/2+k]
                *((float*)bbuf)[k*N/2+j];
              break;
            case C_DBL:
              ((double*)cbuf)[i*N/2+j] += ((double*)abuf)[i*N/2+k]
                *((double*)bbuf)[k*N/2+j];
              break;
            case C_DCPL:
              ((DoubleComplex*)cbuf)[i*N/2+j].real +=
                (((DoubleComplex*)abuf)[i*N/2+k].real
                 *((DoubleComplex*)bbuf)[k*N/2+j].real
                 -(((DoubleComplex*)abuf)[i*N/2+k].imag
                   *((DoubleComplex*)bbuf)[k*N/2+j].imag));
              ((DoubleComplex*)cbuf)[i*N/2+j].imag +=
                (((DoubleComplex*)abuf)[i*N/2+k].real
                 *((DoubleComplex*)bbuf)[k*N/2+j].imag
                 +(((DoubleComplex*)abuf)[i*N/2+k].imag
                   *((DoubleComplex*)bbuf)[k*N/2+j].real));
              break;
            case C_SCPL:
              ((SingleComplex*)cbuf)[i*N/2+j].real +=
                (((SingleComplex*)abuf)[i*N/2+k].real
                 *((SingleComplex*)bbuf)[k*N/2+j].real
                 -(((SingleComplex*)abuf)[i*N/2+k].imag
                   *((SingleComplex*)bbuf)[k*N/2+j].imag));
              ((SingleComplex*)cbuf)[i*N/2+j].imag +=
                (((SingleComplex*)abuf)[i*N/2+k].real
                 *((SingleComplex*)bbuf)[k*N/2+j].imag
                 +(((SingleComplex*)abuf)[i*N/2+k].imag
                   *((SingleComplex*)bbuf)[k*N/2+j].real));
              break;
            default:
              GA_Error("wrong data type", data_type);
          }
        }
      }
    }
    NGA_Put(g_a,lo,hi,cbuf,&ld);
  }
  if (me == 0) printf("\n\n\n\n");

  /* Get norm of g_a */
  switch (data_type) {
    case C_FLOAT:
      ftmp = NGA_Fdot_patch(g_a,'N',lo,hi,g_a,'N',lo,hi);
      break;
    case C_DBL:
      dtmp = NGA_Ddot_patch(g_a,'N',lo,hi,g_a,'N',lo,hi);
      break;
    case C_DCPL:
      ztmp = NGA_Zdot_patch(g_a,'N',lo,hi,g_a,'N',lo,hi);
      break;
    case C_SCPL:
      ctmp = NGA_Cdot_patch(g_a,'N',lo,hi,g_a,'N',lo,hi);
      break;
    default:
      GA_Error("wrong data type", data_type);
  }
  /* subtract C from A and put the results in B */
  beta_flt = -1.0;
  beta_dbl = -1.0;
  beta_scpl.real = -1.0;
  beta_dcpl.real = -1.0;
  NGA_Zero_patch(g_b,lo,hi);
  NGA_Add_patch(alpha,g_a,lo,hi,beta,g_c,lo,hi,g_b,lo,hi);
  /* evaluate the norm of the difference between the two matrices */
  switch (data_type) {
    case C_FLOAT:
      fdiff = NGA_Fdot_patch(g_b,'N',lo,hi,g_b,'N',lo,hi);
      if (ftmp != 0.0) {
        fdiff /= ftmp;
      }
      if(fabs(fdiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(fdiff), TOLERANCE);
        GA_Error("GA_Sgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Sgemm OK\n\n");
      }
      break;
    case C_DBL:
      ddiff = NGA_Ddot_patch(g_b,'N',lo,hi,g_b,'N',lo,hi);
      if (dtmp != 0.0) {
        ddiff /= dtmp;
      }
      if(fabs(ddiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(ddiff), TOLERANCE);
        GA_Error("GA_Dgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Dgemm OK\n\n");
      }
      break;
    case C_DCPL:
      zdiff = NGA_Zdot_patch(g_b,'N',lo,hi,g_b,'N',lo,hi);
      if (ztmp.real != 0.0 || ztmp.imag != 0.0) {
        ddiff = sqrt((zdiff.real*zdiff.real+zdiff.imag*zdiff.imag)
            /(ztmp.real*ztmp.real+ztmp.imag*ztmp.imag));
      } else {
        ddiff = sqrt(zdiff.real*zdiff.real+zdiff.imag*zdiff.imag);
      }
      if(fabs(ddiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(zdiff.real), TOLERANCE);
        GA_Error("GA_Zgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Zgemm OK\n\n");
      }
      break;
    case C_SCPL:
      cdiff = NGA_Cdot_patch(g_b,'N',lo,hi,g_b,'N',lo,hi);
      if (ctmp.real != 0.0 || ctmp.imag != 0.0) {
        fdiff = sqrt((cdiff.real*cdiff.real+cdiff.imag*cdiff.imag)
            /(ctmp.real*ctmp.real+ctmp.imag*ctmp.imag));
      } else {
        fdiff = sqrt(cdiff.real*cdiff.real+cdiff.imag*cdiff.imag);
      }
      if(fabs(fdiff) > TOLERANCE) {
        printf("\nabs(result) = %f > %f\n", fabsf(cdiff.real), TOLERANCE);
        GA_Error("GA_Cgemm Failed", 1);
      } else if (me == 0) {
        printf("\nGA_Cgemm OK\n\n");
      }
      break;
    default:
      GA_Error("wrong data type", data_type);
  }

  }
#endif
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
  test(C_SCPL);
  /*
  */
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

#ifdef USE_REGULAR
    if (me == 0) printf("\nUsing regular data distribution\n\n");
#endif
#ifdef USE_SIMPLE_CYCLIC
    if (me == 0) printf("\nUsing simple block-cyclic data distribution\n\n");
#endif
#ifdef USE_SCALAPACK
    if (me == 0) printf("\nUsing ScaLAPACK data distribution\n\n");
#endif
#ifdef USE_TILED
    if (me == 0) printf("\nUsing tiled data distribution\n\n");
#endif

    if(!MA_init((Integer)MT_F_DBL, stack/nproc, heap/nproc))
       GA_Error("MA_init failed bytes= %d",stack+heap);   

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

