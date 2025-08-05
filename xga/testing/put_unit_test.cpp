#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#define DIM  5
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
  dims[1] = 2*DIM;
  if (rank == 0) {
    printf("Testing PUT on a  %d x %d matrix\n",dims[0],dims[1]);
  }
  XGA::GlobalArray<double> ga(group, ndim, dims);
  printf("p[%d] (main) Calling allocate\n",rank);
  ga.allocate();
  printf("p[%d] (main) Completed allocate\n",rank);

  /* initialize global array using put */
  int64_t lo[2], hi[2], ld;
  int nghbr = (rank+1)%size;
  ga.distribution(nghbr,lo,hi);
  printf("p[%d] (main) lo[0]: %d hi[0]: %d lo[1]: %d hi[1]: %d\n",
      wrank,lo[0],hi[0],lo[1],hi[1]);
  int64_t nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  double *buf = new double[nelems];
  /* initialize local buffer */
  int64_t idim = hi[0]-lo[0]+1;
  int64_t jdim = hi[1]-lo[1]+1;
  int64_t i, j;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      buf[j+jdim*i] = static_cast<double>(j+lo[1] + (i+lo[0])*dims[1]);
    }
  }
  printf("p[%d] (main) Calling put\n",rank);
  ga.put(lo,hi,buf,&jdim);
  printf("p[%d] (main) Completed put\n",rank);
  ga.distribution(rank,lo,hi);
  printf("p[%d] (main) Completed destribution lo[0]: %ld hi[0]: %ld"
      " lo[1]: %ld hi[1]: %ld\n",
      rank,lo[0],hi[0],lo[1],hi[1]);
  void *vptr;
  ga.accessPtr(lo, hi, &vptr, &ld);
  printf("p[%d] (main) Completed accessPtr\n",rank);
  double *dptr = static_cast<double*>(vptr);
  printf("p[%d] dptr: %p\n",rank,dptr);
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  int ok = 1;
  int chk;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      printf("p[%d] i: %ld j: %ld val: %f exp: %f\n",rank,i+lo[0],j+lo[1],
          dptr[j+jdim*i],static_cast<double>(j+lo[1] + (i+lo[0])*dims[1]));
      if (dptr[j+jdim*i] != static_cast<double>(j+lo[1] + (i+lo[0])*dims[1])) {
        ok = 0;
      }
    }
  }
  printf("p[%d] (main) Completed correctness check ok: %d\n",rank,ok);
  ga.clear();
  printf("p[%d] (main) Completed clear\n",rank);

  MPI_Comm comm = group->MPIComm();
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Put test PASSES\n");
  } else if (chk == 0) {
    printf("\n Put test FAILS\n");
  }
  delete [] buf;
  env->finalize();
  MPI_Finalize();
  return 0;
}
