#define BASE_NAME  "/scratch/da.try"
#define BASE_NAME1 "/scratch/da1.try"
#define BASE_NAME2 "/scratch/da2.try"
#  define FNAME   BASE_NAME
#  define FNAME1  BASE_NAME1
#  define FNAME2  BASE_NAME2

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "macdecls.h"
#include "ga.h"
#include "dra.h"
#include "sndrcv.h"
#include "srftoc.h"

#define NDIM 3
#define SIZE 300
#define NFAC 3
/*
#define NDIM 3
#define SIZE 1800

#define NDIM 2
#define SIZE 4000

#define NDIM 1
#define SIZE 16000000
*/

#define MAXDIM 7
#define TRUE (logical)1
#define FALSE (logical)0

#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876
float ran0(long *idum)
{
  long k;
  float ans;

  *idum ^= MASK;
  k=(*idum)/IQ;
  *idum = IA*(*idum-k*IQ)-IR*k;
  if (*idum < 0) *idum += IM;
  ans=AM*(*idum);
  *idum ^= MASK;
  return ans;
}


void fill_random(double *a, int isize)
{
  long *idum;
  long i, j;

  j = 38282;
  idum = &j;
  a[0] = (double)ran0(idum);
  for (i=0; i<(long)isize; i++) {
    a[i] = (double)(10000.0*ran0(idum));
  }
}


void test_io_dbl()
{
  int n,ndim = NDIM,nfac=NFAC;
  double err, tt0, tt1, mbytes;
  int g_a, g_b, g_c, g_d, d_a, d_b, d_c;
  int i, j, itmp, req, loop, nelem;
  int dlo[MAXDIM],dhi[MAXDIM],glo[MAXDIM],ghi[MAXDIM];
  int dims[MAXDIM],reqdims[MAXDIM],icoord[MAXDIM];
  int me, nproc, isize;
  double plus, minus;
  double *index;
  int ld[MAXDIM], chunk[MAXDIM];
  char filename[80];
  logical status;
 
  n = SIZE;

  loop  = 30;
  req = -1;
  nproc = GA_Nnodes();
  me    = GA_Nodeid();

  if (me == 0) {
    printf("Creating temporary global array %d",n);
    for (i=1; i<ndim; i++) {
      printf(" x %d",n);
    }
    printf("\n");
  }
  if (me == 0) fflush(stdout);
  GA_Sync();
  for (i=0; i<ndim; i++) {
    dims[i] = n;
    chunk[i] = 1;
  }

  g_a = NGA_Create(MT_DBL, ndim, dims, "a", chunk);
  if (!g_a) GA_Error("NGA_Create failed: a", 0);
  g_b = NGA_Create(MT_DBL, ndim, dims, "b", chunk);
  if (!g_b) GA_Error("NGA_Create failed: b", 0);
  if (me == 0) printf("done\n");
  if (me == 0) fflush(stdout);

/*     initialize g_a with random values
      ... use ga_access to avoid allocating local buffers for ga_put */

  GA_Sync();
  NGA_Distribution(g_a, me, glo, ghi);
  NGA_Access(g_a, glo, ghi, &index, ld);
  isize = 1;
  for (i=0; i<ndim; i++) isize *= (ghi[i]-glo[i]+1);
  fill_random(index, isize);
  GA_Sync();
  GA_Zero(g_b);


/*.......................................................................*/
  if (me == 0) {
    printf("Creating Disk array %d",n*nfac);
    for (i=1; i<ndim; i++) {
      printf(" x %d",n*nfac);
    }
    printf("\n");
  }
  if (me == 0) fflush(stdout);
  for (i=0; i<ndim; i++) {
    reqdims[i] = n;
    dims[i] = n*nfac;
  }
  strcpy(filename,FNAME);
  GA_Sync();
  if (NDRA_Create(MT_DBL, ndim, dims, "A", filename, DRA_RW,
      reqdims, &d_a) != 0) GA_Error("NDRA_Create failed(d_a): ",0);
  if (me == 0) printf("Blocking write\n");
  if (me == 0) fflush(stdout);
  nelem = 1;
  for (i=0; i<ndim; i++) nelem *= nfac;
  
  for (i=0; i<ndim; i++) {
    glo[i] = 0;
    ghi[i] = n-1;
  }
  tt1 = 0.0;
  for (i=0; i<nelem; i++) {
    /* calculate indices corresponding to element i */
    itmp = i;
    icoord[0] = itmp%nfac;
      j = 0;
      if (me == 0) if (icoord[j] >= nfac || icoord[j] < 0)
        printf("Invalid icoord[%d]: %d\n",j,icoord[j]);
    for (j=1; j<ndim; j++) {
      itmp = (itmp-icoord[j-1])/nfac;
      icoord[j] = itmp%nfac;
      if (me == 0) if (icoord[j] >= nfac || icoord[j] < 0)
        printf("Invalid icoord[%d]: %d\n",j,icoord[j]);
    }
    for (j=0; j<ndim; j++) {
      dlo[j] = n*icoord[j];
      dhi[j] = n*(icoord[j]+1)-1;
    }
    tt0 = tcgtime_();
    if (NDRA_Write_section(FALSE, g_a, glo, ghi,
                         d_a, dlo, dhi, &req) != 0)
      GA_Error("ndra_write_section failed:",0);

    if (DRA_Wait(req) != 0) GA_Error("DRA_Wait failed(d_a): ",req);
    tt1 += (tcgtime_() - tt0);
  }
  mbytes = 1.e-6*(double)(pow(nfac*n,ndim)*sizeof(double));
  if (me == 0) {
    printf("%11.2f MB  time = %11.2f rate = %11.3f MB/s\n",
        mbytes,tt1,mbytes/tt1);
  }

  tt0 = tcgtime_();
  if (DRA_Close(d_a) != 0) GA_Error("DRA_Close failed(d_a): ",d_a);
  tt1 += (tcgtime_() - tt0);
  if (me == 0) {
    printf("Time including DRA_Close\n");
    printf("%11.2f MB  time = %11.2f rate = %11.3f MB/s\n",
        mbytes,tt1,mbytes/tt1);
  }

  if (me == 0) printf("\n");
  if (me == 0) printf("disk array closed\n");
  if (me == 0) fflush(stdout);
/*.......................................................................*/


  if (me == 0) printf("\n");
  if (me == 0) printf("opening disk array\n");
  if (DRA_Open(filename, DRA_R, &d_a) != 0) GA_Error("DRA_Open failed",0);
  if (me == 0) printf("Blocking read\n");
  if (me == 0) fflush(stdout);

  tt1 = 0.0;
  for (i=0; i<nelem; i++) {
    /* calculate indices correspondint to element i */
    itmp = i;
    icoord[0] = itmp%nfac;
      j = 0;
      if (me == 0) if (icoord[j] >= nfac || icoord[j] < 0)
        printf("Invalid icoord[%d]: %d\n",j,icoord[j]);
    for (j=1; j<ndim; j++) {
      itmp = (itmp-icoord[j-1])/nfac;
      icoord[j] = itmp%nfac;
      if (me == 0) if (icoord[j] >= nfac || icoord[j] < 0)
        printf("Invalid icoord[%d]: %d\n",j,icoord[j]);
    }
    for (j=0; j<ndim; j++) {
      dlo[j] = n*icoord[j];
      dhi[j] = n*(icoord[j]+1)-1;
    }
    tt0 = tcgtime_();
    if (NDRA_Read_section(FALSE, g_b, glo, ghi, d_a, dlo, dhi, &req) != 0)
      GA_Error("NDRA_Read_section failed:",0);

    if (DRA_Wait(req) != 0) GA_Error("DRA_Wait failed: ",&req);
    tt1 += (tcgtime_() - tt0);
    plus = 1.0;
    minus = -1.0;
    GA_Add(&plus, g_a, &minus, g_b, g_b);
    err = GA_Ddot(g_b, g_b);
    if (err != 0) {
      if (me == 0) printf("BTW, we have error = %f\n",err);
    } else {
      if (me == 0) printf("OK\n");
    }
  }
  if (me == 0) {
    printf("%11.2f MB  time = %11.2f rate = %11.3f MB/s\n",
        mbytes,tt1,mbytes/tt1);
  }
  if (DRA_Delete(d_a) != 0) GA_Error("DRA_Delete failed",0);
/*.......................................................................*/
  GA_Destroy(g_a);
  GA_Destroy(g_b);
}

void main(argc, argv)
int argc;
char **argv;
{
  int status, me;
  int max_arrays = 10;
  double max_sz = 1e8, max_disk = 1e10, max_mem = 1e6;
#if   defined(IBM)|| defined(CRAY_T3E)
  int stack = 9000000, heap = 4000000;
#else
  int stack = 1200000, heap = 800000;
#endif

  PBEGIN_(argc, argv); 
  GA_Initialize();
  if (!GA_Uses_ma()) {
    stack = 100000;
    heap  = 100000;
  }

  if (MA_init(MT_F_DBL, stack, heap) ) {
    me    = GA_Nodeid();
    if (DRA_Init(max_arrays, max_sz, max_disk, max_mem) != 0)
       GA_Error("DRA_Init failed: ",0);
    if (me == 0) printf("\n");
    if (me == 0) printf("TESTING PERFORMANCE OF DISK ARRAYS\n");
    if (me == 0) printf("\n");
    test_io_dbl();
    status = DRA_Terminate();
    GA_Terminate();
  } else {
    printf("MA_init failed\n");
  }
  if(me == 0) printf("all done ...\n");
  PEND_();
}
