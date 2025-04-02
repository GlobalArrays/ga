#include "cmx_group.hpp"
#include "cmx_environment.hpp"
#include "cmx_alloc.hpp"
#include <iostream>

/* Test fenceProc and fenceAll operation */

#define TEST_SIZE 65536
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  int rank, size;
  {
    CMX::Group *world = env->getWorldGroup();
    rank = world->rank();
    size = world->size();
    if (rank == 0) {
      std::cout <<"Number of processors: "<<size<<std::endl;
    }
    std::vector<void*> ptrs;
    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&wrank);

    CMX::Allocation alloc;
    long bytes = sizeof(long)*size*TEST_SIZE;
    alloc.malloc(bytes,world);
    if (rank == 0) {
      std::cout <<"Distributed malloc completed"<<std::endl;
    }
    world->barrier();
    /* access pointers on all processors */
    alloc.access(ptrs);
    int i, iproc;
    for (i=0; i<size*TEST_SIZE; i++) {
      reinterpret_cast<long*>(ptrs[rank])[i] = 0;
    }
    long *buf = new long[TEST_SIZE];
    bytes = sizeof(long)*TEST_SIZE;
    /* fill up system with data */
    for (iproc = 0; iproc<size; iproc++) {
      for (i=0; i<TEST_SIZE; i++) {
        buf[i] = iproc*size*TEST_SIZE+rank*TEST_SIZE+i;
      }
      void *tptr = static_cast<void*>(static_cast<long*>(ptrs[iproc])
          +rank*TEST_SIZE);
      alloc.put(buf,tptr,bytes,iproc);
    }
    for (iproc = 0; iproc<size; iproc++) {
      alloc.fenceProc(iproc);
    }
    world->barrier();
    long* ptr = static_cast<long*>(alloc.access());
    bool ok = true;
    for (i=0; i<size*TEST_SIZE; i++) {
      if (ptr[i] != (long)(i+rank*size*TEST_SIZE) && ok) {
        printf("p[%d] ptr[%d]: %ld expected: %ld\n",rank,
            i,ptr[i],(long)(i+rank*size*TEST_SIZE));
        ok = false;
      }
    }
    if (!ok) {
      std::cout<<"Mismatch found on process "<<rank<<std::endl;
    } else if (rank == 0) {
      std::cout<<"fenceProc operation is OK"<<std::endl;
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
