#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#define DIM 256
template<typename idx_type, typename data_type>
void scatter_acc_test()
{
  XGA::Environment *env = XGA::Environment::instance();
  XGA::Group *group = env->getWorldGroup();
  int rank = group->rank();
  int size = group->size();
  int wrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
  /* Create global array */
  int ndim = 3;
  idx_type dims[3];
  dims[0] = DIM;
  dims[1] = DIM;
  dims[2] = DIM;
  XGA::GlobalArray<data_type> ga(group, ndim, dims);
  ga.allocate();

  /* initialize global array */
  idx_type lo[3], hi[3], ld[2];
  ga.distribution(rank,lo,hi);
  /* Access local data in array */
  idx_type idim = hi[0]-lo[0]+1;
  idx_type jdim = hi[1]-lo[1]+1;
  idx_type kdim = hi[2]-lo[2]+1;
  void *vptr;
  ga.accessPtr(lo, hi, &vptr, ld);
  data_type *dptr = static_cast<data_type*>(vptr);
  idx_type i, j, k, n, idx, icnt;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      for (k=0; k<kdim; k++) {
        dptr[k+kdim*j+kdim*jdim*i] = static_cast<data_type>(k+lo[2] + (j+lo[1])*dims[2]
            + (i+lo[0])*dims[2]*dims[1]);
      }
    }
  }
  ga.sync();

  /* update global array using scatteracc */
  idx_type nelems = static_cast<idx_type>(static_cast<double>(dims[0]*dims[1]*dims[2])
    / static_cast<double>(size))+1;
  idx_type total = dims[0]*dims[1]*dims[2];
  data_type *values = new data_type[nelems];
  idx_type *subscripts = new idx_type[nelems*ndim];
  icnt = 0;
  for (n=rank; n<total; n+=size) {
    idx = n;
    k = idx%dims[2];
    idx = (idx-k)/dims[2];
    j = idx%dims[1];
    i = (idx-j)/dims[1];
    values[icnt] = static_cast<data_type>(n);
    subscripts[ndim*icnt] = i;
    subscripts[ndim*icnt+1] = j;
    subscripts[ndim*icnt+2] = k;
    icnt++;
  }
  ga.distribution(rank,lo,hi);
  data_type one = static_cast<data_type>(1);
  data_type two = static_cast<data_type>(2);
  ga.scatterAcc(values, subscripts, icnt, one);
  ga.sync();
  delete [] values;
  delete [] subscripts;
  /* Check values */
  nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1)*(hi[2]-lo[2]+1);
  /* Access local data in array */
  int ok = 1;
  int chk;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      for (k=0; k<kdim; k++) {
        if (dptr[k+kdim*j+kdim*jdim*i]
            != two*(static_cast<data_type>(k+lo[2] + (j+lo[1])*dims[2]
              + (i+lo[0])*dims[2]*dims[1]))) {
          printf("p[%d] Check fails for i: %ld j: %ld k: %ld"
              " actual: %f expected: %f\n",
              wrank,i+lo[0],j+lo[1],k+lo[2],dptr[k+kdim*j+i*kdim*jdim],
              two*(static_cast<data_type>(k+lo[2] + (j+lo[1])*dims[2]
                + (i+lo[0])*dims[2]*dims[1])));
          ok = 0;
        }
      }
    }
  }

  MPI_Comm comm = group->MPIComm();
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n scatter-accumulate test PASSES\n");
  } else if (chk == 0) {
    printf("\n scatter-accumulate test FAILS\n");
  }
}

int main(int argc, char **argv)
{
  XGA::Environment *env = XGA::Environment::instance(&argc,&argv);
  XGA::Group *group = env->getWorldGroup();
  int rank = group->rank();
  int size = group->size();
  if (rank == 0) {
    int64_t dims[3];
    dims[0] = DIM;
    dims[1] = DIM;
    dims[2] = DIM;
    printf("\nTesting SCATTERACC on a  %d x %d x %d array",dims[0],dims[1],dims[2]);
    printf(" running on %d processors\n",size);
  }
  if (rank == 0) {
    printf("\nTesting SCATTERACC for ints and int64_t indices\n");
  }
  scatter_acc_test<int64_t,int>();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for longs and int64_t indices\n");
  }
  scatter_acc_test<int64_t,long>();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for floats and int64_t indices\n");
  }
  scatter_acc_test<int64_t,float>();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for doubles and int64_t indices\n");
  }
  scatter_acc_test<int64_t,double>();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for complex floats and int64_t indices\n");
  }
  scatter_acc_test<int64_t,std::complex<float> >();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for complex doubles and int64_t indices\n");
  }
  scatter_acc_test<int64_t,std::complex<double> >();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for ints and int indices\n");
  }
  scatter_acc_test<int,int>();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for longs and int indices\n");
  }
  scatter_acc_test<int64_t,long>();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for floats and int indices\n");
  }
  scatter_acc_test<int,float>();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for doubles and int indices\n");
  }
  scatter_acc_test<int,double>();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for complex floats and int indices\n");
  }
  scatter_acc_test<int,std::complex<float> >();
  if (rank == 0) {
    printf("\nTesting SCATTERACC for complex doubles and int indices\n");
  }
  scatter_acc_test<int,std::complex<double> >();
  env->finalize();
  MPI_Finalize();
  return 0;
}
