#include "cmx_group.hpp"
#include "cmx_environment.hpp"
#include "cmx_alloc.hpp"
#include <iostream>

/* Test atomic read-modify-write operation */

#define TEST_SIZE 1048576
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  {
    CMX::Group *world = env->getWorldGroup();
    int rank = world->rank();
    int size = world->size();
    int one = 1;
    if (rank == 0) {
      std::cout <<"Number of processors: "<<size<<std::endl;
    }
    std::vector<void*> ptrs;

    CMX::Allocation alloc;
    long bytes = 0;
    if (rank == 0) {
      bytes = sizeof(int);
    }
    alloc.malloc(bytes,world);
    if (rank == 0) {
      std::cout <<"Distributed malloc completed"<<std::endl;
    }
    world->barrier();
    /* access pointers on all processors */
    alloc.access(ptrs);
    /* allocate array to keep track of values */
    int *counts = new int[TEST_SIZE];
    int i;
    for (i=0; i<TEST_SIZE; i++) counts[i] = 0;
    i=0;
    int op = CMX_FETCH_AND_ADD;
    while (i<TEST_SIZE) {
      alloc.readModifyWrite(op,&i,ptrs[0],one,0);
      if (i<TEST_SIZE) {
        counts[i] = 1;
      }
    }
    MPI_Comm comm = world->MPIComm();
    int *fcounts = new int[TEST_SIZE];
    MPI_Allreduce(counts,fcounts,TEST_SIZE,MPI_INT,MPI_SUM,comm);
    bool ok = true;
    for (i=0; i<TEST_SIZE; i++) {
      if (fcounts[i] != 1) {
        ok = false;
      }
    }
    if (ok && rank == 0) {
      std::cout << "Test of read-modify-write fetch-and-add operation PASSES"
        << std::endl;
    } else if (!ok) {
      std::cout << "Test of read-modify-write fetch-and-add operation FAILS"
        << std::endl;
    }

    delete [] counts;
    delete [] fcounts;
    alloc.free();
    if (rank == 0) {
      std::cout <<"Allocation freed for fetch-and-add test"<<std::endl;
    }

    /* test swap operation */
    bytes = TEST_SIZE*sizeof(int);
    CMX::Allocation alloc_swp;
    alloc_swp.malloc(bytes,world);
    if (rank == 0) {
      std::cout <<"Distributed malloc completed"<<std::endl;
    }
    world->barrier();
    /* access pointers on all processors */
    alloc_swp.access(ptrs);
    /* allocate array to keep track of values */
    counts = new int[TEST_SIZE];
    int nghbr = (rank+1)%size;
    int *ptr = reinterpret_cast<int*>(ptrs[rank]);
    for (i=0; i<TEST_SIZE; i++) {
      counts[i] = 0;
      ptr[i] = i+TEST_SIZE*rank;
    }
    op = CMX_SWAP;
    for (i=0; i<TEST_SIZE; i++) {
      alloc_swp.readModifyWrite(op,&counts[i],
          reinterpret_cast<int*>(ptrs[nghbr])+i,one,nghbr);
    }
    alloc_swp.fenceAll();
    world->barrier();
    ok = true;
    for (i=0; i<TEST_SIZE; i++) {
      if (counts[i] != i+nghbr*TEST_SIZE) {
        ok = false;
      }
      if (ptr[i] != 0) {
        ok = false;
      }
    }
    if (ok && rank == 0) {
      std::cout << "Test of read-modify-write swap operation PASSES"
        << std::endl;
    } else if (!ok) {
      std::cout << "Test of read-modify-write swap operation FAILS"
        << std::endl;
    }

    delete [] counts;
    alloc_swp.free();
    if (rank == 0) {
      std::cout <<"Allocation freed for swap test"<<std::endl;
    }
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
