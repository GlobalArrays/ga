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
    CMX::Group *world = env->getWorldGroup(); rank = world->rank();
    size = world->size();
    int wrank;
    MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
    if (rank == 0) {
      std::cout <<"Number of processors: "<<size<<std::endl;
    }
    int total_len = size*TEST_SIZE;
    nghbr = (rank+1)%size;
    std::vector<void*> ptrs;
    /* create vector of starting location */
    std::vector<int> istart(size+1);
    int i;
    for (i=0; i<size; i++) istart[i] = i*TEST_SIZE;
    istart[size] = size*TEST_SIZE;
    /* figure out how large buffers need to be. Add 2*size extra for safety */
    int nsize = istart[rank+1]-istart[rank]+2*size;

    CMX::Allocation alloc;
    long bytes = sizeof(long)*(istart[rank+1]-istart[rank]);
    alloc.malloc(bytes,world);
    if (rank == 0) {
      std::cout <<"Distributed malloc completed"<<std::endl;
    }
    world->barrier();
    /* access pointers on all processors */
    alloc.access(ptrs);
    long *buf = new long[nsize];
    int iproc = 0;
    int icnt = 0;
    cmx_giov_t giov;
    giov.src = new void*[nsize];
    giov.dst = new void*[nsize];
    for (i=rank; i<=total_len+size; i += size) {
      if (i >= istart[iproc+1]) {
        giov.count = icnt;
        giov.bytes = sizeof(long);
        if (icnt > 0) alloc.putv(&giov, 1, iproc);
        iproc++;
        icnt = 0;
      }
      if (i >= total_len) break;
      buf[icnt] = static_cast<long>(i);
      giov.src[icnt] = reinterpret_cast<void*>(&buf[icnt]);
      giov.dst[icnt] = static_cast<char*>(ptrs[iproc])
                     + (i-istart[iproc])*sizeof(long);
      icnt++;
    }
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
      std::cout<<"Vector PUT operation is OK"<<std::endl;
    }
    /* Set all values in allocation to zero */
    for (i=0; i<TEST_SIZE; i++) {
      ptr[i] = (long)0;
    }
    cmx_request *req;
    req = new cmx_request[size];
    bool *rproc;
    rproc = new bool[size];
    for (i=0; i<size; i++) rproc[i] = false;
    icnt = 0;
    iproc = 0;
    for (i=rank; i<total_len+size; i += size) {
      if (i >= istart[iproc+1]) {
        giov.count = icnt;
        giov.bytes = sizeof(long);
        if (icnt > 0) alloc.nbputv(&giov, 1, iproc, &req[iproc]);
        if (icnt > 0) rproc[iproc] = true;
        iproc++;
        icnt = 0;
      }
      if (i >= total_len) break;
      buf[icnt] = static_cast<long>(i);
      giov.src[icnt] = reinterpret_cast<void*>(&buf[icnt]);
      giov.dst[icnt] = static_cast<char*>(ptrs[iproc])
                     + (i-istart[iproc])*sizeof(long);
      icnt++;
    }
    for (i=0; i<size; i++) {
      if (rproc[i]) alloc.wait(&req[i]);
    }
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
      std::cout<<"Vector Non-blocking PUT operation is OK"<<std::endl;
    }
    delete [] rproc;
    delete [] req;
    delete [] giov.src;
    delete [] giov.dst;
    delete [] buf;

    alloc.free();
    if (rank == 0) {
      std::cout <<"Allocation freed"<<std::endl;
    }
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
