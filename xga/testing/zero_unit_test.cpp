#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#define DIM  2048
int main(int argc, char **argv)
{
  XGA::Environment *env = XGA::Environment::instance(&argc,&argv);
  XGA::Group *group = env->getWorldGroup();
  int rank = group->rank();
  int size = group->size();
  int wrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
  /* Create global array */
  int ndim = 2;
  int64_t dims[2];
  dims[0] = DIM;
  dims[1] = DIM;
  if (rank == 0) {
    printf("\nTesting ZERO on a  %d x %d matrix",dims[0],dims[1]);
    printf(" running on %d processors\n",size);
  }
  XGA::GlobalArray<double> ga(group, ndim, dims);
  ga.allocate();

  int64_t lo[2], hi[2], ld;
  ga.distribution(rank,lo,hi);
  void *vptr;
  ga.accessPtr(lo, hi, &vptr, &ld);
  double *dptr = static_cast<double*>(vptr);
  /* initialize global array with non-zero values */
  int64_t idim = hi[0]-lo[0]+1;
  int64_t jdim = hi[1]-lo[1]+1;
  int64_t i, j;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      dptr[j+jdim*i] = static_cast<double>(j+lo[1] + (i+lo[0])*dims[1]);
    }
  }
  ga.sync();
  ga.zero();
  int ok = 1;
  int chk;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (dptr[j+jdim*i] != 0.0) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: 0.0\n",
            wrank,i+lo[0],j+lo[1],dptr[j+jdim*i]);
        ok = 0;
      }
    }
  }

  MPI_Comm comm = group->MPIComm();
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Zero test PASSES\n");
  } else if (chk == 0) {
    printf("\n Zero test FAILS\n");
  }
  ga.clear();
  env->finalize();
  MPI_Finalize();
  return 0;
}
