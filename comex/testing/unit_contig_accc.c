#include "comex.h"
#include <stdio.h>

/* Test contiguous accumulate operation */

#define TEST_SIZE 1048576
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  comex_group_t group = COMEX_GROUP_WORLD;
  comex_init();
  {
    int rank, size, nghbr;
    void **ptrs;
    size_t bytes;
    long *buf;
    int i;
    int world_nghbr;
    long *ptr;
    int ok;
    long scale;
    int op;
    comex_request_t req;

    comex_group_rank(group,&rank);
    comex_group_size(group,&size);
    if (rank == 0) {
      printf("Number of processors: %d\n",size);
    }
    nghbr = (rank+1)%size;

    bytes = sizeof(long)*TEST_SIZE;
    ptrs = (void**)malloc(size*sizeof(void*));
    comex_malloc(ptrs,bytes,group);
    if (rank == 0) {
      printf("Distributed malloc completed\n");
    }
    comex_barrier(group);
    /* initialize data in allocation */
    ptr = (long*)ptrs[rank];
    for (i=0; i<TEST_SIZE; i++) {
      ptr[i] = rank*TEST_SIZE+i;
    }

    /* initialize buffer */
    buf = (long*)malloc(TEST_SIZE*sizeof(long));
    for (i=0; i<TEST_SIZE; i++) {
      buf[i] = nghbr*TEST_SIZE+i;
    }
    comex_group_translate_world(group,nghbr,&world_nghbr);
    printf("Process %d sending data to neighber %d\n",rank,world_nghbr);
    scale = 1;
    op =  COMEX_ACC_LNG;
    comex_acc(op,&scale,buf,ptrs[nghbr],bytes,world_nghbr,group);
    comex_fence_all(group);
    comex_barrier(group);
    ok = 1;
    for (i=0; i<TEST_SIZE; i++) {
      if (ptr[i] != (long)(2*(i+rank*TEST_SIZE)) && ok) {
        printf("p[%d] (acc) ptr[%d]: %ld expected: %ld\n",rank,
            i,ptr[i],(long)(2*(i+rank*TEST_SIZE)));
        ok = 0;
      }
    }
    if (!ok) {
      printf("Mismatch found on process %d\n",rank);
    } else if (rank == 0) {
      printf("Contiguous ACC operation is OK\n");
    }
    comex_barrier(group);
    printf("Process %d sending data to neighbor %d using"
        " non-blocking accumulate\n",rank,world_nghbr);
    comex_nbacc(op,&scale,buf,ptrs[nghbr],bytes,world_nghbr,group,&req);
    comex_wait(&req);
    comex_fence_all(group);
    comex_barrier(group);
    ok = 1;
    for (i=0; i<TEST_SIZE; i++) {
      if (ptr[i] != (long)(3*(i+rank*TEST_SIZE)) && ok) {
        printf("p[%d] (nbacc) ptr[%d]: %ld expected: %ld\n",rank,
            i,ptr[i],(long)(3*(i+rank*TEST_SIZE)));
        ok = 0;
      }
    }
    if (!ok) {
      printf("Mismatch found on process %d\n",rank);
    } else if (rank == 0) {
      printf("Contiguous Non-blocking ACC operation is OK\n");
    }

    comex_free(ptrs[rank],group);
    if (rank == 0) {
      printf("Allocation freed\n");
    }
    free(buf);
  }
  comex_finalize();
  MPI_Finalize();
  return 0;
}
