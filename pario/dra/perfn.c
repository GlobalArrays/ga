#define BASE_NAME  "da.try"
#define BASE_NAME1 "da1.try"
#  define FNAME   BASE_NAME
#  define FNAME1  BASE_NAME1

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "macdecls.h"
#include "ga.h"
#include "dra.h"
#include "sndrcv.h"
#include "srftoc.h"

#define NDIM 3
#define SIZE 20
#define NSIZE 8000
#define LSIZE 64000
#define MAXDIM 7
#define TRUE (logical)1
#define FALSE (logical)0

#define MULTFILES 0

#ifdef SOLARIS
#  if MULTFILES
#    define USEMULTFILES 1
#  endif
#else
#  define USEMULTFILES 1
#endif

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
  int n,m,ndim = NDIM;
  double err, tt0, tt1, mbytes;
  int g_a, g_b, d_a, d_b;
  int i, req, loop;
  int dlo[MAXDIM],dhi[MAXDIM],glo[MAXDIM],ghi[MAXDIM];
  int dims[MAXDIM],reqdims[MAXDIM];
  int me, nproc, isize;
  double plus, minus;
  double *index;
  int ld[MAXDIM], chunk[MAXDIM];
#if USEMULTFILES
  int ilen;
#endif
  char filename[80], filename1[80];
  logical status;
 
  n = pow(NSIZE,1.0/ndim)+0.5;
  m = pow(LSIZE,1.0/ndim)+0.5;

  loop  = 30;
  req = -1;
  nproc = GA_Nnodes();
  me    = GA_Nodeid();

  if (me == 0) {
    printf("Creating global arrays %d",n);
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

/*     initialize g_a, g_b with random values
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
    printf("Creating Disk array %d",n);
    for (i=1; i<ndim; i++) {
      printf(" x %d",n);
    }
    printf("\n");
  }
  if (me == 0) fflush(stdout);
  for (i=0; i<ndim; i++) {
    reqdims[i] = n;
  }
  strcpy(filename,FNAME);
#if USEMULTFILES
  ilen = strlen(filename);
  if (me < 10) {
    filename[ilen] = '0'+me;
    filename[ilen+1] = '\0';
  } else if (10 <= me && me < 100) {
    filename[ilen] = '0' + me/10;
    filename[ilen+1] = '0' + me%10;
    filename[ilen+2] = '\0';
  } else {
    filename[ilen] = '0' + me/100;
    i = me - me/100;
    filename[ilen+1] = '0' + i/10;
    filename[ilen+2] = '0' + i%10;
    filename[ilen+3] = '\0';
  }
#endif
  if (NDRA_Create(MT_DBL, ndim, dims, "A", filename, DRA_RW,
      reqdims, &d_a) != 0) GA_Error("NDRA_Create failed: ",0);
  if (me == 0) printf("alligned blocking write\n");
  fflush(stdout);
  tt0 = tcgtime_();
  if (NDRA_Write(g_a, d_a, &req) != 0) GA_Error("NDRA_Write failed:",0);
  if (DRA_Wait(req) != 0) GA_Error("DRA_Wait failed: ",req);
  tt1 = tcgtime_() - tt0;
  mbytes = 1.e-6 * (double)(pow(n,ndim));
  if (me == 0) {
    printf("%11.2f MB  time = %f11.2 rate = %f11.3 MB/s\n",
        mbytes,tt1,mbytes/tt1);
  }

  if (DRA_Close(d_a) != 0) GA_Error("DRA_Close failed: ",d_a);
  tt1 = tcgtime_() - tt0;
  if (me == 0) {
    printf("%11.2f MB  time = %f11.2 rate = %f11.3 MB/s\n",
        mbytes,tt1,mbytes/tt1);
  }

  if (me == 0) printf("\n");
  if (me == 0) printf("disk array closed\n");
  if (me == 0) fflush(stdout);
/*.......................................................................*/


  if (me == 0) {
    printf("Creating Disk array %d",m);
    for (i=1; i<ndim; i++) {
      printf(" x %d",m);
    }
    printf("\n");
  }
  for (i=0; i<ndim; i++) {
    dims[i] = m;
    reqdims[i] = n;
  }
  strcpy(filename1,FNAME1);
#if USEMULTFILES
  ilen = strlen(filename);
  if (me < 10) {
    filename[ilen] = '0'+me;
    filename[ilen+1] = '\0';
  } else if (10 <= me && me < 100) {
    filename[ilen] = '0' + me/10;
    filename[ilen+1] = '0' + me%10;
    filename[ilen+2] = '\0';
  } else {
    filename[ilen] = '0' + me/100;
    i = me - me/100;
    filename[ilen+1] = '0' + i/10;
    filename[ilen+2] = '0' + i%10;
    filename[ilen+3] = '\0';
  }
#endif
  if (NDRA_Create(MT_DBL, ndim, dims, "B", filename1, DRA_RW,
      reqdims, &d_b) != 0) GA_Error("NDRA_Create failed: ",0);

  if (me == 0) printf("non alligned blocking write\n");
  if (me == 0) fflush(stdout);

  for (i=0; i<ndim; i++) {
    glo[i] = 0;
    ghi[i] = n-1;
    dlo[i] = 1;
    dhi[i] = n;
  }
  tt0 = tcgtime_();
  if (NDRA_Write_section(FALSE, g_a, glo, ghi,
                         d_b, dlo, dhi, &req) != 0)
      GA_Error("ndra_write_section failed:",0);

  if (DRA_Wait(req) != 0) GA_Error("DRA_Wait failed: ",req);
  tt1 = tcgtime_() - tt0;
  mbytes = 1.e-6*(double)(pow(n,ndim));
  if (me == 0) {
    printf("%11.2f MB  time = %f11.2 rate = %f11.3 MB/s\n",
        mbytes,tt1,mbytes/tt1);
  }

  if (DRA_Close(d_b) != 0) GA_Error("DRA_Close failed: ",d_b);
  tt1 = tcgtime_() - tt0;
  if (me == 0) {
    printf("%11.2f MB  time = %f11.2 rate = %f11.3 MB/s\n",
        mbytes,tt1,mbytes/tt1);
  }

  if (me == 0) printf("\n");
  if (me == 0) printf("disk array closed\n");
  if (me == 0) fflush(stdout);
/*.......................................................................*/


  if (me == 0) printf("\n");
  if (me == 0) printf("opening disk array\n");
  if (DRA_Open(filename, DRA_R, &d_a) != 0) GA_Error("DRA_Open failed",0);
  if (me == 0) printf("alligned blocking read\n");
  if (me == 0) fflush(stdout);
  tt0 = tcgtime_();
  if (NDRA_Read(g_b, d_a, &req) != 0) GA_Error("NDRA_Read failed:",0);
  if (DRA_Wait(req) != 0) GA_Error("DRA_Wait failed: ",req);
  tt1 = tcgtime_() - tt0;
  if (me == 0) {
    printf("%11.2f MB  time = %f11.2 rate = %f11.3 MB/s\n",
        mbytes,tt1,mbytes/tt1);
  }
  plus = 1.0;
  minus = -1.0;
  GA_Add(&plus, g_a, &minus, g_b, g_b);
  err = GA_Ddot(g_b, g_b);
  if (err != 0) {
    if (me == 0) printf("BTW, we have error = %f\n",err);
  } else {
    if (me == 0) printf("OK\n");
  }
  if (DRA_Delete(d_a) != 0) GA_Error("DRA_Delete failed",0);
/*.......................................................................*/

  if (me == 0) printf("\n");
  if (me == 0) printf("opening disk array\n");
  if (DRA_Open(filename1, DRA_R, &d_b) != 0) GA_Error("DRA_Open failed",0);
  if (me == 0) printf("non alligned blocking read\n");
  if (me == 0) fflush(stdout);
  tt0 = tcgtime_();
  if (NDRA_Read_section(FALSE, g_b, glo, ghi, d_b, dlo, dhi, &req) != 0)
    GA_Error("NDRA_Read_section failed:",0);
  if (DRA_Wait(req) != 0) GA_Error("DRA_Wait failed: ",&req);
  tt1 = tcgtime_() - tt0;
  if (me == 0) {
    printf("%11.2f MB  time = %f11.2 rate = %f11.3 MB/s\n",
        mbytes,tt1,mbytes/tt1);
  }
  GA_Add(&plus, g_a, &minus, g_b, g_b);
  err = GA_Ddot(g_b, g_b);
  if (err != 0) {
    if (me == 0) printf("BTW, we have error = %f\n",err);
  } else {
    if (me == 0) printf("OK\n");
  }
  if (DRA_Delete(d_b) != 0) GA_Error("DRA_Delete failed",0);
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
