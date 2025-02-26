#include "group.hpp"
#include "environment.hpp"
#include "alloc.hpp"
#include <iostream>

/* Test contiguous put operation */

#define TEST_SIZE 1048576
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  int rank, size, nghbr;
  {
    CMX::Group *world = env->getWorldGroup();
    rank = world->rank();
    size = world->size();
    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
    if (rank == 0) {
      std::cout <<"Number of processors: "<<size<<std::endl;
    }
    nghbr = (rank+1)%size;
    std::vector<void*> ptrs;

    CMX::Allocation alloc;
    long bytes = sizeof(long)*TEST_SIZE;
    alloc.malloc(bytes,world);
    if (rank == 0) {
      std::cout <<"Distributed malloc completed"<<std::endl;
    }
    world->barrier();
    /* access pointers on all processors */
    alloc.access(ptrs);
    int i;
    long *buf = new long[TEST_SIZE];
    for (i=0; i<TEST_SIZE; i++) {
      buf[i] = nghbr*TEST_SIZE+i;
    }
    int world_nghbr = world->getWorldRank(nghbr);
    std::cout<<"Process "<<rank<<" sending data to neighbor "
      <<world_nghbr<<std::endl;
    alloc.put(buf,ptrs[nghbr],bytes,world_nghbr);
    alloc.fenceAll();
    world->barrier();
    long* ptr = static_cast<long*>(alloc.access());
    bool ok = true;
    for (i=0; i<TEST_SIZE; i++) {
      if (ptr[i] != (long)(i+rank*TEST_SIZE) && ok) {
        printf("p[%d] ptr[%d]: %ld expected: %ld\n",wrank,
            i,ptr[i],(long)(i+rank*TEST_SIZE));
        ok = false;
      }
    }
    if (!ok) {
      std::cout<<"Mismatch found on process "<<rank<<std::endl;
    } else if (rank == 0) {
      std::cout<<"Contiguous PUT operation is OK"<<std::endl;
    }
    /* Set all values in allocation to zero */
    for (i=0; i<TEST_SIZE; i++) {
      ptr[i] = (long)0;
    }
    std::cout<<"Process "<<rank<<" sending data to neighbor "
      <<world_nghbr<< " using non-blocking put"<<std::endl;
    cmx_request req;
    alloc.nbput(buf,ptrs[nghbr],bytes,world_nghbr,&req);
    alloc.wait(&req);
    alloc.fenceAll();
    world->barrier();
    ok = true;
    for (i=0; i<TEST_SIZE; i++) {
      if (ptr[i] != (long)(i+rank*TEST_SIZE) && ok) {
        printf("p[%d] ptr[%d]: %ld expected: %ld\n",wrank,
            i,ptr[i],(long)(i+rank*TEST_SIZE));
        ok = false;
      }
    }
    if (!ok) {
      std::cout<<"Mismatch found on process "<<rank<<std::endl;
    } else if (rank == 0) {
      std::cout<<"Contiguous Non-blocking PUT operation is OK"<<std::endl;
    }

    alloc.free();
    if (rank == 0) {
      std::cout <<"Allocation freed"<<std::endl;
    }
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
