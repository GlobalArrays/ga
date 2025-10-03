#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#define DIM  2048
template<typename idx_type, typename data_type>
void fill_test()
{
  XGA::Environment *env = XGA::Environment::instance();
  XGA::Group *group = env->getWorldGroup();
  int rank = group->rank();
  int size = group->size();
  int wrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
  /* Create global array */
  int ndim = 2;
  idx_type dims[2];
  dims[0] = DIM;
  dims[1] = DIM;
  XGA::GlobalArray<data_type> ga(group, ndim, dims);
  ga.allocate();

  idx_type lo[2], hi[2], ld;
  ga.distribution(rank,lo,hi);
  void *vptr;
  ga.accessPtr(lo, hi, &vptr, &ld);
  data_type *dptr = static_cast<data_type*>(vptr);
  /* initialize global array with zero values */
  idx_type idim = hi[0]-lo[0]+1;
  idx_type jdim = hi[1]-lo[1]+1;
  idx_type i, j;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      dptr[j+jdim*i] = static_cast<data_type>(0);
    }
  }
  ga.releasePtr(lo, hi);
  ga.sync();
  data_type two = static_cast<data_type>(2);
  ga.fill(two);
  int ok = 1;
  int chk;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (dptr[j+jdim*i] != two) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i+lo[0],j+lo[1],dptr[j+jdim*i],two);
        ok = 0;
      }
    }
  }

  MPI_Comm comm = group->MPIComm();
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Fill test PASSES\n");
  } else if (chk == 0) {
    printf("\n Fill test FAILS\n");
  }
}

int main(int argc, char **argv)
{
  XGA::Environment *env = XGA::Environment::instance(&argc,&argv);
  XGA::Group *group = env->getWorldGroup();
  int rank = group->rank();
  int size = group->size();
  if (rank == 0) {
    int64_t dims[2];
    dims[0] = DIM;
    dims[1] = DIM;
    printf("\nTesting FILL on a  %d x %d matrix",dims[0],dims[1]);
    printf(" running on %d processors\n",size);
  }
  if (rank == 0) {
    printf("\nTesting FILL for ints and int64_t indices\n");
  }
  fill_test<int64_t,int>();
  if (rank == 0) {
    printf("\nTesting FILL for longs and int64_t indices\n");
  }
  fill_test<int64_t,long>();
  if (rank == 0) {
    printf("\nTesting FILL for long longs and int64_t indices\n");
  }
  fill_test<int64_t,long long>();
  if (rank == 0) {
    printf("\nTesting FILL for floats and int64_t indices\n");
  }
  fill_test<int64_t,float>();
  if (rank == 0) {
    printf("\nTesting FILL for doubles and int64_t indices\n");
  }
  fill_test<int64_t,double>();
  if (rank == 0) {
    printf("\nTesting FILL for complex floats and int64_t indices\n");
  }
  fill_test<int64_t,std::complex<float> >();
  if (rank == 0) {
    printf("\nTesting FILL for complex doubles and int64_t indices\n");
  }
  fill_test<int64_t,std::complex<double> >();
  if (rank == 0) {
    printf("\nTesting FILL for ints and int indices\n");
  }
  fill_test<int,int>();
  if (rank == 0) {
    printf("\nTesting FILL for longs and int indices\n");
  }
  fill_test<int,long>();
  if (rank == 0) {
    printf("\nTesting FILL for long longs and int indices\n");
  }
  fill_test<int64_t,long long>();
  if (rank == 0) {
    printf("\nTesting FILL for floats and int indices\n");
  }
  fill_test<int,float>();
  if (rank == 0) {
    printf("\nTesting FILL for doubles and int indices\n");
  }
  fill_test<int,double>();
  if (rank == 0) {
    printf("\nTesting FILL for complex floats and int indices\n");
  }
  fill_test<int,std::complex<float> >();
  if (rank == 0) {
    printf("\nTesting FILL for complex doubles and int indices\n");
  }
  fill_test<int,std::complex<double> >();
  env->finalize();
  MPI_Finalize();
  return 0;
}
