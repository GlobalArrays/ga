#include "group.hpp"
#include "environment.hpp"
#include "alloc.hpp"
#include <iostream>

/* Test strided get operation */

#define DIM 32  // DIM must be multiple of 2
int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  CMX::Environment *env = CMX::Environment::instance(&argc,&argv);
  int rank, size, nghbr;
  int wrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
  {
    CMX::Group *world = env->getWorldGroup();
    rank = world->rank();
    size = world->size();
    if (rank == 0) {
      std::cout <<"Number of processors: "<<size<<std::endl;
    }
    nghbr = (rank+1)%size;
    std::vector<void*> ptrs;

    CMX::Allocation alloc;
    int64_t bytes = sizeof(long)*DIM*DIM*DIM;
    alloc.malloc(bytes,world);
    if (rank == 0) {
      std::cout <<"Distributed malloc completed"<<std::endl;
    }
    world->barrier();
    /* access pointers on all processors */
    alloc.access(ptrs);
    int i, j, k, iblk;
    long *buf = new long[DIM*DIM*DIM];
    for (i=0; i<DIM*DIM*DIM; i++) buf[i] = 0;
    long* ptr = static_cast<long*>(alloc.access());
    for (k=0; k<DIM; k++) {
      for (j=0; j<DIM; j++) {
        for (i=0; i<DIM; i++) {
          ptr[i+j*DIM+k*DIM*DIM] = rank*DIM*DIM*DIM+i+j*DIM+k*DIM*DIM;
        }
      }
    }
    world->barrier();
    int world_nghbr = world->getWorldRank(nghbr);
    printf("Process %d getting data from neighbor %d using"
        " strided get\n",rank,world_nghbr);
    for (iblk = 0; iblk < 8; iblk++) {
      int64_t src_stride[2], dst_stride[2], count[3];
      int stride_levels = 2;
      int ib, jb, kb, itmp;
      int64_t offset;
      long *src, *dst;
      itmp = iblk;
      ib = itmp%2;
      itmp = (itmp-ib)/2;
      jb = itmp%2;
      kb = (itmp-jb)/2;
      src_stride[0] = sizeof(long)*DIM;
      src_stride[1] = sizeof(long)*DIM*DIM;
      dst_stride[0] = sizeof(long)*DIM;
      dst_stride[1] = sizeof(long)*DIM*DIM;
      count[0] = sizeof(long)*DIM/2;
      count[1] = DIM/2;
      count[2] = DIM/2;
      offset = DIM*(kb*src_stride[1]+jb*src_stride[0]+sizeof(long)*ib)/2;
      src = reinterpret_cast<long*>(reinterpret_cast<char*>(ptrs[nghbr])+offset);
      offset = DIM*(kb*dst_stride[1]+jb*dst_stride[0]+sizeof(long)*ib)/2;
      dst = reinterpret_cast<long*>(reinterpret_cast<char*>(buf)+offset);
      alloc.gets(src,src_stride,dst,dst_stride,count,stride_levels,world_nghbr);
    }
    alloc.fenceAll();
    world->barrier();
    bool ok = true;
    for (k=0; k<DIM; k++) {
      for (j=0; j<DIM; j++) {
        for (i=0; i<DIM; i++) {
          if (buf[i+j*DIM+k*DIM*DIM] !=
              (long)(i+j*DIM+k*DIM*DIM+nghbr*DIM*DIM*DIM) && ok) {
            printf("p[%d] buf[%d][%d][%d]: %ld expected: %ld\n",rank,
                k,j,i,buf[i+j*DIM+k*DIM*DIM],
                (long)(i+j*DIM+k*DIM*DIM+nghbr*DIM*DIM*DIM));
            ok = false;
          }
        }
      }
    }
    if (!ok) {
      std::cout<<"Mismatch found on process "<<rank<<std::endl;
    } else if (rank == 0) {
      std::cout<<"Strided GET operation is OK"<<std::endl;
    }
    /* Set all values in buffer to zero */
    for (i=0; i<DIM*DIM*DIM; i++) {
      buf[i] = (long)0;
    }
    printf("Process %d getting data from neighbor %d using"
        " non-blocking strided get\n",rank,world_nghbr);
    cmx_request req[8];
    for (iblk = 0; iblk < 8; iblk++) {
      int64_t src_stride[2], dst_stride[2], count[3];
      int stride_levels = 2;
      int ib, jb, kb, itmp;
      int64_t offset;
      long *src, *dst;
      itmp = iblk;
      ib = itmp%2;
      itmp = (itmp-ib)/2;
      jb = itmp%2;
      kb = (itmp-jb)/2;
      src_stride[0] = sizeof(long)*DIM;
      src_stride[1] = sizeof(long)*DIM*DIM;
      dst_stride[0] = sizeof(long)*DIM;
      dst_stride[1] = sizeof(long)*DIM*DIM;
      count[0] = sizeof(long)*DIM/2;
      count[1] = DIM/2;
      count[2] = DIM/2;
      offset = DIM*(kb*src_stride[1]+jb*src_stride[0]+sizeof(long)*ib)/2;
      src = reinterpret_cast<long*>(reinterpret_cast<char*>(ptrs[nghbr])+offset);
      offset = DIM*(kb*dst_stride[1]+jb*dst_stride[0]+sizeof(long)*ib)/2;
      dst = reinterpret_cast<long*>(reinterpret_cast<char*>(buf)+offset);
      alloc.nbgets(src,src_stride,dst,dst_stride,count,stride_levels,
          world_nghbr,&req[iblk]);
    }
    for (iblk = 0; iblk < 8; iblk++) {
      alloc.wait(&req[iblk]);
    }
    alloc.fenceAll();
    world->barrier();
    ok = true;
    for (k=0; k<DIM; k++) {
      for (j=0; j<DIM; j++) {
        for (i=0; i<DIM; i++) {
          if (buf[i+j*DIM+k*DIM*DIM] !=
              (long)(i+j*DIM+k*DIM*DIM+nghbr*DIM*DIM*DIM) && ok) {
            printf("p[%d] buf[%d][%d][%d]: %ld expected: %ld\n",rank,
                k,j,i,buf[i+j*DIM+k*DIM*DIM],
                (long)(i+j*DIM+k*DIM*DIM+nghbr*DIM*DIM*DIM));
            ok = false;
          }
        }
      }
    }
    if (!ok) {
      std::cout<<"Mismatch found on process "<<rank<<std::endl;
    } else if (rank == 0) {
      std::cout<<"Strided Non-blocking GET operation with wait is OK"<<std::endl;
    }

    /* Set all values in buffer to zero */
    for (i=0; i<DIM*DIM*DIM; i++) {
      buf[i] = (long)0;
    }
    printf("Process %d getting data from neighbor %d using"
        " non-blocking strided get\n",rank,world_nghbr);
    for (iblk = 0; iblk < 8; iblk++) {
      int64_t src_stride[2], dst_stride[2], count[3];
      int stride_levels = 2;
      int ib, jb, kb, itmp;
      int64_t offset;
      long *src, *dst;
      itmp = iblk;
      ib = itmp%2;
      itmp = (itmp-ib)/2;
      jb = itmp%2;
      kb = (itmp-jb)/2;
      src_stride[0] = sizeof(long)*DIM;
      src_stride[1] = sizeof(long)*DIM*DIM;
      dst_stride[0] = sizeof(long)*DIM;
      dst_stride[1] = sizeof(long)*DIM*DIM;
      count[0] = sizeof(long)*DIM/2;
      count[1] = DIM/2;
      count[2] = DIM/2;
      offset = DIM*(kb*src_stride[1]+jb*src_stride[0]+sizeof(long)*ib)/2;
      src = reinterpret_cast<long*>(reinterpret_cast<char*>(ptrs[nghbr])+offset);
      offset = DIM*(kb*dst_stride[1]+jb*dst_stride[0]+sizeof(long)*ib)/2;
      dst = reinterpret_cast<long*>(reinterpret_cast<char*>(buf)+offset);
      alloc.nbgets(src,src_stride,dst,dst_stride,count,stride_levels,
          world_nghbr,&req[iblk]);
    }
    bool test_rslt[8];
    for (iblk = 0; iblk < 8; iblk++) test_rslt[iblk] = false;
    ok = true;
    while (ok) {
      bool done = true;
      for (iblk=0; iblk<8; iblk++) {
        if (!test_rslt[iblk]) test_rslt[iblk] = alloc.test(&req[iblk]);
        if (!test_rslt[iblk]) done = false;
      }
      if (done) ok = false;
    }
    alloc.fenceAll();
    world->barrier();
    ok = true;
    for (k=0; k<DIM; k++) {
      for (j=0; j<DIM; j++) {
        for (i=0; i<DIM; i++) {
          if (buf[i+j*DIM+k*DIM*DIM] !=
              (long)(i+j*DIM+k*DIM*DIM+nghbr*DIM*DIM*DIM) && ok) {
            printf("p[%d] buf[%d][%d][%d]: %ld expected: %ld\n",rank,
                k,j,i,buf[i+j*DIM+k*DIM*DIM],
                (long)(i+j*DIM+k*DIM*DIM+nghbr*DIM*DIM*DIM));
            ok = false;
          }
        }
      }
    }
    if (!ok) {
      std::cout<<"Mismatch found on process "<<rank<<std::endl;
    } else if (rank == 0) {
      std::cout<<"Strided Non-blocking GET operation with test is OK"<<std::endl;
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
