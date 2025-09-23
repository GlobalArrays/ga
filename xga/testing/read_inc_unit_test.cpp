#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#define DIM  2048
template <typename idx_type, typename data_type>
void readinc_test()
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
  dims[1] = 2*DIM;
  XGA::GlobalArray<data_type> ga(group, ndim, dims);
  ga.allocate();
  /* Create counter */
  int one = 1;
  idx_type dim1 = 1;
  XGA::GlobalArray<data_type> counter(group, one, &dim1);
  counter.allocate();

  /* initialize global array and counter to zero*/
  idx_type lo[2], hi[2], ld;
  counter.distribution(rank,lo,hi);
  void *vptr;
  data_type *dptr;
  if (lo[0] == hi[0] && lo[0] == 0) {
    counter.accessPtr(lo, hi, &vptr, &ld);
    dptr = static_cast<data_type*>(vptr);
    dptr[0] = static_cast<data_type>(0);
  }
  ga.distribution(rank,lo,hi);
  ga.accessPtr(lo, hi, &vptr, &ld);
  dptr = static_cast<data_type*>(vptr);
  idx_type idim = hi[0]-lo[0]+1;
  idx_type jdim = hi[1]-lo[1]+1;
  int ok = 1;
  int chk;
  idx_type i, j;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      dptr[j+jdim*i] = static_cast<data_type>(0);
    }
  }

  idx_type nelems = dims[0]*dims[1];
  /* Fill in array using read increment function */
  data_type inc;
  data_type d_one = 1;
  idx_type zero = 0;
  data_type idx = counter.readInc(&zero, d_one);
  while (idx < nelems) {
    idx_type subscript[2];
    idx_type i, j;
    j = idx % dims[1];
    i = (idx - j)/dims[1];
    subscript[0] = i;
    subscript[1] = j;
    inc = static_cast<data_type>(idx);
    i = ga.readInc(subscript, inc);
    if (i != 0) {
      env->error("read inc error", i);
    }
    idx = counter.readInc(&zero, d_one);
  }
  ga.sync();
  ga.distribution(rank,lo,hi);
  ga.accessPtr(lo, hi, &vptr, &ld);
  dptr = static_cast<data_type*>(vptr);
  ok = 1;
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (dptr[j+jdim*i] != static_cast<data_type>(j+lo[1] + (i+lo[0])*dims[1])) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i+lo[0],j+lo[1], dptr[j+jdim*i],
            static_cast<data_type>(j+lo[1] + (i+lo[0])*dims[1]));
        ok = 0;
      }
    }
  }
  MPI_Comm comm = group->MPIComm();
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Read-increment test PASSES\n");
  } else if (chk == 0) {
    printf("\n Read-increment test FAILS\n");
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
    dims[1] = 2*DIM;
    printf("\nTesting READINC on a  %d x %d matrix",dims[0],dims[1]);
    printf(" running on %d processors\n",size);
  }
  if (rank == 0) {
    printf("\nTesting READINC for ints and int64_t indices\n");
  }
  readinc_test<int64_t,int>();
  if (rank == 0) {
    printf("\nTesting READINC for longs and int64_t indices\n");
  }
  readinc_test<int64_t,long>();
  if (rank == 0) {
    printf("\nTesting READINC for ints and int indices\n");
  }
  readinc_test<int,int>();
  if (rank == 0) {
    printf("\nTesting READINC for longs and int indices\n");
  }
  readinc_test<int,long>();
  env->finalize();
  MPI_Finalize();
}
