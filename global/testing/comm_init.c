#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "mpi.h"
#include "ga.h"
#include "macdecls.h"

#define DIM 256

void comm_test(MPI_Comm comm)
{
  double *buf;
  int i, j, ii, jj, ld;
  int lo[2], hi[2];
  int dims[2];
  int two = 2;
  int g_a;
  int me, nprocs, prem;
  int nelems, ok;
  int rank;
  MPI_Comm_rank(comm,&rank);
  if (GA_Initialize_comm(comm)) {
    me = GA_Nodeid();
    nprocs = GA_Nnodes();
    prem = (me+1)%nprocs;
    if (me == 0) {
      printf("\nRunning comm_test on %d processors\n\n",nprocs);
    }

    g_a = NGA_Create_handle();
    dims[0] = DIM;
    dims[1] = DIM;
    NGA_Set_data(g_a,two,dims,C_DBL);
    NGA_Allocate(g_a);
    GA_Zero(g_a);

    /* Find out what section of global array I'm writing to */
    NGA_Distribution(g_a,prem,lo,hi);
    nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
    ld = hi[1]-lo[1]+1;
    buf = (double*)malloc(nelems*sizeof(double));
    for (i=lo[0]; i<=hi[0]; i++) {
      ii = i-lo[0];
      for (j=lo[1]; j<=hi[1]; j++) {
        jj = j-lo[1];
        buf[ii*ld+jj] = (double)(i*DIM+j);
      }
    }
    NGA_Put(g_a,lo,hi,buf,&ld);
    GA_Sync();
    free(buf);
    /* Copy data from a different section of global array and check values */
    prem = me-1;
    if (prem < 0) prem = nprocs-1;
    NGA_Distribution(g_a,prem,lo,hi);
    nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
    ld = hi[1]-lo[1]+1;
    buf = (double*)malloc(nelems*sizeof(double));
    for (i=0; i<nelems; i++) buf[i] = 0.0;
    NGA_Get(g_a,lo,hi,buf,&ld);

    ok = 1;
    for (i=lo[0]; i<=hi[0]; i++) {
      ii = i-lo[0];
      for (j=lo[1]; j<=hi[1]; j++) {
        jj = j-lo[1];
        if (buf[ii*ld+jj] != (double)(i*DIM+j)) {
          if (ok == 1) {
            printf("p[%d] Mismatch for (%d,%d) expected: %f actual: %f\n",
                me,i,j,buf[ii*ld+jj],(double)(i*DIM+j));
            ok = 0;
          }
        }
      }
    }
    free(buf);
    if (me == 0 && ok) {
      printf("Test passed\n");
    } else if (!ok) {
      printf("Test failed\n");
    }
    GA_Terminate();
  }
}

int main(int argc, char **argv)
{
  int rank, size;
  int color;
  MPI_Comm group;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    printf("MPI COMM WORLD has %d processors\n",size);
  }
  color = 1;
  if (rank < size/2) color = 0;
  MPI_Comm_split(MPI_COMM_WORLD,color,rank,&group);
  if (rank == 0) {
    printf("\nRun test on first group\n");
  }
  if (color == 0) {
    comm_test(group);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    printf("\nRun test on second group\n");
  }
  if (color == 1) {
    comm_test(group);
  }
  MPI_Finalize();
}
