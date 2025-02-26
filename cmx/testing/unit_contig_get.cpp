#include "group.hpp"
#include "environment.hpp"
#include "alloc.hpp"
#include <iostream>

/* Test contiguous get operation */

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
    printf("Original value of wrank: %d\n",wrank);
    if (rank == 0) {
      std::cout <<"Number of processors: "<<size<<std::endl;
    }
    nghbr = (rank+1)%size;
    std::vector<void*> ptrs;

    CMX::Allocation alloc;
    int64_t bytes = sizeof(long)*TEST_SIZE;
    alloc.malloc(bytes,world);
    if (rank == 0) {
      std::cout <<"Distributed malloc completed"<<std::endl;
    }
    world->barrier();
    /* initialize data in allocation */
    int i;
    long* ptr = static_cast<long*>(alloc.access());
    for (i=0; i<TEST_SIZE; i++) {
      ptr[i] = rank*TEST_SIZE+i;
    }
    world->barrier();
    long *buf = new long[TEST_SIZE];
    /* initialize buffer to zero */
    for (i=0; i<TEST_SIZE; i++) {
      buf[i] = static_cast<long>(0);
    }
    /* access pointers on all processors */
    alloc.access(ptrs);
    int world_nghbr = world->getWorldRank(nghbr);
    std::cout<<"Process "<<rank<<" getting data from neighbor "
      <<world_nghbr<<std::endl;
    alloc.get(ptrs[nghbr],buf,bytes,world_nghbr);
    bool ok = true;
    for (i=0; i<TEST_SIZE; i++) {
      if (buf[i] != static_cast<long>(i+nghbr*TEST_SIZE) && ok) {
        printf("p[%d] ptr[%d]: %ld expected: %d\n",wrank,
            i,buf[i],i+rank*TEST_SIZE);
        ok = false;
      }
    }
    if (!ok) {
      std::cout<<"Mismatch found on process "<<rank<<std::endl;
    } else if (rank == 0) {
      std::cout<<"Contiguous GET operation is OK"<<std::endl;
    }
    /* initialize buffer to zero */
    for (i=0; i<TEST_SIZE; i++) {
      buf[i] = static_cast<long>(0);
    }
    std::cout<<"Process "<<rank<<" getting data from neighbor "
      <<world_nghbr<< " using non-blocking get"<<std::endl;
    cmx_request req;
    alloc.nbget(ptrs[nghbr],buf,bytes,world_nghbr,&req);
    alloc.wait(&req);
    ok = true;
    for (i=0; i<TEST_SIZE; i++) {
      if (buf[i] != static_cast<long>(i+nghbr*TEST_SIZE) && ok) {
        printf("p[%d] ptr[%d]: %ld expected: %d\n",wrank,
            i,buf[i],i+rank*TEST_SIZE);
        ok = false;
      }
    }
    if (!ok) {
      std::cout<<"Mismatch found on process "<<rank<<std::endl;
    } else if (rank == 0) {
      std::cout<<"Contiguous Non-blocking GET operation is OK"<<std::endl;
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
