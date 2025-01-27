#include "comex.h"
#include <stdio.h>

/* Test contiguous put operation */

#define TEST_SIZE 1048576
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size, nghbr;
  comex_group_t group = COMEX_GROUP_WORLD;
  comex_init();
  {
    int nghbr;
    void **ptrs;
    size_t bytes;
    long *buf;
    int i;
    int world_nghbr;
    long *ptr;
    int ok;

    comex_group_rank(group,&rank);
    comex_group_size(group,&size);
    if (rank == 0) {
      printf("Number of processors: %d\n",size);
    }
    nghbr = (rank+1)%size;

    bytes = sizeof(long)*TEST_SIZE;
    ptrs = (void**)malloc(size*sizeof(void*));
    comex_malloc(ptrs,bytes,group);
    for (i=0; i<size; i++) {
      printf("p[%d] ptrs[%d]: %p\n",rank,i,ptrs[i]);
    }
    if (rank == 0) {
      printf("Distributed malloc completed\n");
    }
    comex_barrier(group);
    /* access pointers on all processors */
    buf = (long*)malloc(sizeof(long)*TEST_SIZE);
    for (i=0; i<TEST_SIZE; i++) {
      buf[i] = nghbr*TEST_SIZE+i;
    }
    comex_group_translate_world(group,nghbr,&world_nghbr);
    printf("Process %d sending data to neighbor %d\n",rank,
        world_nghbr);
    comex_put(buf,ptrs[nghbr],bytes,world_nghbr,group);
    comex_fence_all(group);
    comex_barrier(group);
    ptr = (long*)ptrs[rank];
    ok = 1;
    for (i=0; i<TEST_SIZE; i++) {
      if (ptr[i] != (long)(i+rank*TEST_SIZE)) {
        printf("p[%d] ptr[%d]: %ld expected: %d\n",rank,
            i,ptr[i],i+rank*TEST_SIZE);
        ok = 0;
        break;
      }
    }
    if (!ok) {
      printf("Mismatch found on process %d\n",rank);
    } else if (rank == 0) {
      printf("Contiguous PUT operation is OK %d\n",rank);
    }
    comex_free(ptrs[rank],group);
    if (rank == 0) {
      printf("Allocation free\n");
    }
  }
  comex_finalize();
  MPI_Finalize();
  return 0;
}
