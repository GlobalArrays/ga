#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#define DIM 256
int main(int argc, char **argv)
{
  XGA::Environment *env = XGA::Environment::instance(&argc,&argv);
  XGA::Group *group = env->getWorldGroup();
  int rank = group->rank();
  int size = group->size();
  int wrank;
  MPI_Comm_rank(MPI_COMM_WORLD,&wrank);
  /* Create global array */
  int ndim = 3;
  int64_t dims[3];
  dims[0] = DIM;
  dims[1] = DIM;
  dims[2] = DIM;
  if (rank == 0) {
    printf("\nTesting SCATTER on a  %d x %d x %d array",dims[0],dims[1],dims[2]);
    printf(" running on %d processors\n",size);
  }
  XGA::GlobalArray<double> ga(group, ndim, dims);
  ga.allocate();

  /* initialize global array using scatter */
  int64_t nelems = static_cast<int64_t>(static_cast<double>(dims[0]*dims[1]*dims[2])
    / static_cast<double>(size))+1;
  int64_t total = dims[0]*dims[1]*dims[2];
  double *values = new double[nelems];
  int64_t *subscripts = new int64_t[nelems*ndim];
  int64_t i, j, k, n, idx, icnt;
  icnt = 0;
  for (n=rank; n<total; n+=size) {
    idx = n;
    k = idx%dims[2];
    idx = (idx-k)/dims[2];
    j = idx%dims[1];
    i = (idx-j)/dims[1];
    values[icnt] = static_cast<double>(n);
    subscripts[ndim*icnt] = i;
    subscripts[ndim*icnt+1] = j;
    subscripts[ndim*icnt+2] = k;
    icnt++;
  }
  int64_t lo[3], hi[3], ld[2];
  ga.distribution(rank,lo,hi);
  ga.scatter(values, subscripts, icnt);
  ga.sync();
  delete [] values;
  delete [] subscripts;
  /* Check values */
  nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1)*(hi[2]-lo[2]+1);
  /* Access local data in array */
  int64_t idim = hi[0]-lo[0]+1;
  int64_t jdim = hi[1]-lo[1]+1;
  int64_t kdim = hi[2]-lo[2]+1;
  void *vptr;
  ga.accessPtr(lo, hi, &vptr, ld);
  double *dptr = static_cast<double*>(vptr);
  int ok = 1;
  int chk;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      for (k=0; k<kdim; k++) {
        if (dptr[k+kdim*j+kdim*jdim*i]
            != static_cast<double>(k+lo[2] + (j+lo[1])*dims[2]
              + (i+lo[0])*dims[2]*dims[1])) {
          printf("p[%d] Check fails for i: %ld j: %ld k: %ld"
              " actual: %f expected: %f\n",
              wrank,i+lo[0],j+lo[1],k+lo[2],dptr[k+kdim*j+i*kdim*jdim],
              static_cast<double>(k+lo[2] + (j+lo[1])*dims[2]
                + (i+lo[0])*dims[2]*dims[1]));
          ok = 0;
        }
      }
    }
  }

  MPI_Comm comm = group->MPIComm();
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n scatter test PASSES\n");
  } else if (chk == 0) {
    printf("\n scatter test FAILS\n");
  }
  ga.clear();
  env->finalize();
  MPI_Finalize();
  return 0;
}
