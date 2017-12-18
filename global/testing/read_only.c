#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"
#include "ga.h"
#include "macdecls.h"

#define DEFAULT_DIM 4096
#define MAX_LOOP 20

/* Convenience function to check that something is true on all processors */
int trueEverywhere(int flag)
{
  int tflag, nprocs;
  if (flag) tflag = 1;
  else tflag = 0;
  nprocs = GA_Nnodes();
  GA_Igop(&tflag,1,"+");
  if (nprocs == tflag) return 1;
  return 0;
}

/* Return box dims that have been increased by a factor of 2 */
int nextBlock(int *ix, int *iy, int dims[])
{
  if (*ix > *iy) {
    *iy *= 2;
  } else {
    *ix *= 2;
  }
  if ((*ix)>=dims[0] || (*iy) > dims[1]) return 0;
  return 1;
}

/* Function to print out timing statistics */
void printTimes(double g_time, int g_ntime, int g_elems, int size[])
{
  int me = GA_Nodeid();
  int nproc = GA_Nnodes();
  double time;
  int ntime;
  long nelems;
  long lone = 1;
  int one = 1;
  double bandwdth;
  double optime;
  int bytes = sizeof(int)*size[0]*size[1];

  NGA_Get(g_time,&me,&me,&time,&one);
  NGA_Get(g_ntime,&me,&me,&ntime,&one);
  NGA_Get(g_elems,&me,&me,&nelems,&one);
  GA_Dgop(&time,one,"+");
  GA_Igop(&ntime,one,"+");
  GA_Lgop(&nelems,one,"+");
  nelems *= sizeof(int);
  bandwdth = ((double)nelems)/time;
  bandwdth /= 1.0e6;
  if (me==0) {
    printf("%10d %10d %10d %10d %16.6e\n",bytes,size[0],size[1],ntime,bandwdth);
  }
}

/* Generate random block */
void getRandomBlock( int ndim, int dims[], int blocksize[], int lo[], int hi[])
{
  int toss;
  int d;
  for (d=0; d<ndim; d++) {
    toss = rand()%(dims[d]-blocksize[d]);
    lo[d] = toss;
    hi[d] = toss+blocksize[d]-1;
  }
}

int main(int argc, char * argv[])
{
  int x = DEFAULT_DIM;
  int y = DEFAULT_DIM;
  int g_src;
  int me, nproc;
  int lo[2], hi[2], ld[2];
  int size[2];
  int i,j,ix,iy,idx,iloc,icnt;
  int return_code = 0;
  int dims[2];
  int ndim = 2;
  int ok;
  int zero = 0, one = 1;
  double rone = 1.0;
  int *ptr;
  long nelems;
  long lone = 1;
  double delta_t;
  int g_time, g_ntime, g_elems;
  int nextOK;

  /* Use a different array size if specified in arguments */
  if(argc >= 3)
  {
    x = atoi(argv[1]);
    y = atoi(argv[2]);
  }

  MPI_Init(&argc, &argv);
  GA_Initialize();

  nproc = GA_Nnodes();
  me = GA_Nodeid();
  i = GA_Cluster_nprocs(GA_Cluster_nodeid());

  /* Find processor grid dimensions and processor grid coordinates */
  if (me==0) {
    printf("\nTest running on %d processors with %d processes per node\n",nproc,i);
    printf("\nArray dimension is %d X %d\n",x,y);
  }

  dims[0] = x;
  dims[1] = y;

  /* Create GA and set all elements to zero */
  g_src = NGA_Create(C_INT, 2, dims, "source", NULL);
  g_time = NGA_Create(C_DBL, 1, &nproc, "times", &one);
  g_ntime = NGA_Create(C_INT, 1, &nproc, "ntimes", &one);
  g_elems = NGA_Create(C_LONG, 1, &nproc, "nelems", &one);
  GA_Zero(g_src);

  /* Fill the global array with unique values */
  NGA_Distribution(g_src,me,lo,hi);
  NGA_Access(g_src,lo,hi,&ptr,ld);
  for (i=lo[0]; i<=hi[0]; i++) {
    for (j=lo[1]; j<=hi[1]; j++) {
      idx = i*y+j;
      iloc = (i-lo[0])*ld[0]+(j-lo[1]);
      ptr[iloc] = idx;
    }
  }
  NGA_Release(g_src,lo,hi);
  GA_Sync();

  /* Array is ready for reading. Set read-only property */
  NGA_Set_property(g_src,"read_only");

  if (me == 0) {
    printf("\nPerformance results for read-only array\n");
  }
  ok = 1;
  nextOK = 1;
  ix = 1;
  iy = 1;
  if (me == 0) {
    printf("\n     Bytes      I-Dim       J-Dim Transfers Bandwidth (MB/s)\n\n");
  }
  while (nextOK) {
    /* Initialize counters and statistics */
    icnt = 0;
    GA_Zero(g_time);
    GA_Zero(g_ntime);
    GA_Zero(g_elems);
    /* Read random blocks from array */
    size[0] = ix;
    size[1] = iy;
    while (icnt < MAX_LOOP) {
      /* Get random block and copy it to local array */
      getRandomBlock(ndim, dims, size, lo, hi);
      nelems = (long)((hi[0]-lo[0]+1)*(hi[1]-lo[1]+1)); 
      ptr = (int*)malloc(nelems*sizeof(int));
      ld[0] = (hi[1]-lo[1]+1);
      delta_t = GA_Wtime();
      NGA_Get(g_src,lo,hi,ptr,ld);
      delta_t = GA_Wtime()-delta_t;
      NGA_Acc(g_time,&me,&me,&delta_t,&one,&rone);
      NGA_Acc(g_ntime,&me,&me,&one,&one,&one);
      NGA_Acc(g_elems,&me,&me,&nelems,&one,&lone);
      /* Check that values are okay */
      for (i=lo[0]; i<=hi[0]; i++) {
        for (j=lo[1]; j<=hi[1]; j++) {
          idx = i*y+j;
          iloc = (i-lo[0])*ld[0]+(j-lo[1]);
          if (ptr[iloc] != idx) {
            ok = 0;
            printf("p[%d] expected: %d actual: %d\n",me,idx,ptr[iloc]);
          }
        }
      }
      free(ptr);
      icnt++;
    }
    printTimes(g_time, g_ntime, g_elems, size);
    nextOK = nextBlock(&ix, &iy, dims);
  }
  if (trueEverywhere(ok)) {
    if (me == 0) {
      printf("\nRead-only test succeeded\n");
    }
  } else if (me == 0) {
    printf("\nRead-only test FAILED\n");
  }

  /* Unset read-only property */
  NGA_Unset_property(g_src);

  if (me == 0) {
    printf("\nPerformance results for regular array\n");
  }
  ok = 1;
  nextOK = 1;
  ix = 1;
  iy = 1;
  if (me == 0) {
    printf("\n     Bytes      I-Dim       J-Dim Transfers Bandwidth (MB/s)\n\n");
  }
  while (nextOK) {
    /* Initialize counters and statistics */
    icnt = 0;
    GA_Zero(g_time);
    GA_Zero(g_ntime);
    GA_Zero(g_elems);
    /* Read random blocks from array */
    size[0] = ix;
    size[1] = iy;
    while (icnt < MAX_LOOP) {
      /* Get random block and copy it to local array */
      getRandomBlock(ndim, dims, size, lo, hi);
      nelems = (long)((hi[0]-lo[0]+1)*(hi[1]-lo[1]+1)); 
      ptr = (int*)malloc(nelems*sizeof(int));
      ld[0] = (hi[1]-lo[1]+1);
      delta_t = GA_Wtime();
      NGA_Get(g_src,lo,hi,ptr,ld);
      delta_t = GA_Wtime()-delta_t;
      NGA_Acc(g_time,&me,&me,&delta_t,&one,&rone);
      NGA_Acc(g_ntime,&me,&me,&one,&one,&one);
      NGA_Acc(g_elems,&me,&me,&nelems,&one,&lone);
      /* Check that values are okay */
      for (i=lo[0]; i<=hi[0]; i++) {
        for (j=lo[1]; j<=hi[1]; j++) {
          idx = i*y+j;
          iloc = (i-lo[0])*ld[0]+(j-lo[1]);
          if (ptr[iloc] != idx) {
            ok = 0;
            printf("p[%d] expected: %d actual: %d\n",me,idx,ptr[iloc]);
          }
        }
      }
      free(ptr);
      icnt++;
    }
    printTimes(g_time, g_ntime, g_elems, size);
    nextOK = nextBlock(&ix, &iy, dims);
  }
  if (trueEverywhere(ok)) {
    if (me == 0) {
      printf("\nRegular array test succeeded\n\n");
    }
  } else if (me == 0) {
    printf("\nRegular array test FAILED\n");
  }

  GA_Destroy(g_src);
  GA_Destroy(g_time);
  GA_Destroy(g_ntime);
  GA_Destroy(g_elems);

  GA_Terminate();
  MPI_Finalize();

  return return_code;
}

