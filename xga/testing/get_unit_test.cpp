#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#define DIM  4
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
    printf("\nTesting GET on a  %d x %d matrix",dims[0],dims[1]);
    printf(" running on %d processors\n",size);
  }
  XGA::GlobalArray<double> ga(group, ndim, dims);
  ga.allocate();

  int64_t lo[2], hi[2], ld;
  if (rank == 0) {
    printf("\n Testing get on whole blocks\n");
  }
  ga.distribution(rank,lo,hi);
  void *vptr;
  ga.accessPtr(lo, hi, &vptr, &ld);
  double *dptr = static_cast<double*>(vptr);
  /* initialize global array */
  int64_t idim = hi[0]-lo[0]+1;
  int64_t jdim = hi[1]-lo[1]+1;
  int64_t i, j;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      dptr[j+jdim*i] = static_cast<double>(j+lo[1] + (i+lo[0])*dims[1]);
    }
  }
  ga.sync();
  int nghbr = (rank+1)%size;
  ga.distribution(nghbr,lo,hi);
  int64_t nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  double *buf = new double[nelems];
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  ga.get(lo,hi,buf,&jdim);
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  int ok = 1;
  int chk;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (buf[j+jdim*i] != static_cast<double>(j+lo[1] + (i+lo[0])*dims[1])) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i+lo[0],j+lo[1],buf[j+jdim*i],
            static_cast<double>(j+lo[1] + (i+lo[0])*dims[1]));
        ok = 0;
      }
    }
  }

  MPI_Comm comm = group->MPIComm();
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Full block get test PASSES\n");
  } else if (chk == 0) {
    printf("\n Full block get test FAILS\n");
  }
  if (rank == 0) {
    printf("\n Testing get on partial blocks\n");
    printf("\n Zero values in local array\n");
  }
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      buf[j+jdim*i] = 0.0;
    }
  }
  ga.sync();
  int64_t plo[2], phi[2];
  nghbr = (rank+1)%size;
  ga.distribution(nghbr,lo,hi);
  nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  /* initialize local buffer */
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  int n;

  /* divide each processor block into 4 sub-blocks */
  for (n=0; n<4; n++) {
    if (n==0) {
      plo[0] = lo[0];
      phi[0] = lo[0]+(hi[0]-lo[0])/2;
      plo[1] = lo[1];
      phi[1] = lo[1]+(hi[1]-lo[1])/2;
    } else if (n==1) {
      plo[0] = lo[0]+(hi[0]-lo[0])/2 + 1;
      phi[0] = hi[0];
      plo[1] = lo[1];
      phi[1] = lo[1]+(hi[1]-lo[1])/2;
    } else if (n==2) {
      plo[0] = lo[0];
      phi[0] = lo[0]+(hi[0]-lo[0])/2;
      plo[1] = lo[1]+(hi[1]-lo[1])/2 + 1;
      phi[1] = hi[1];
    } else if (n==3) {
      plo[0] = lo[0]+(hi[0]-lo[0])/2 + 1;
      phi[0] = hi[0];
      plo[1] = lo[1]+(hi[1]-lo[1])/2 + 1;
      phi[1] = hi[1];
    }
    int64_t ii, jj;
    double *tbuf = buf + plo[1]-lo[1]+(plo[0]-lo[0])*jdim;
    ga.get(plo,phi,tbuf,&jdim);
  }
  ga.sync();
  ga.distribution(rank,lo,hi);
  ga.accessPtr(lo, hi, &vptr, &ld);
  dptr = static_cast<double*>(vptr);
  ok = 1;
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (dptr[j+jdim*i] != static_cast<double>(j+lo[1] + (i+lo[0])*dims[1])) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i+lo[0],j+lo[1], dptr[j+jdim*i],
            static_cast<double>(j+lo[1] + (i+lo[0])*dims[1]));
        ok = 0;
      }
    }
  }

  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Partial block get test PASSES\n");
  } else if (chk == 0) {
    printf("\n Partial block get test FAILS\n");
  }
  if (rank == 0) {
    printf("\n Testing single large get for whole array\n");
    printf("\n Zero values in array\n");
  }
  ga.sync();
  nghbr = (rank+1)%size;
  idim = dims[0];
  jdim = dims[1];
  nelems = idim*jdim;
  /* initialize local buffer to zero*/
  delete [] buf;
  buf = new double[nelems];

  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      buf[j+jdim*i] = 0.0;
    }
  }
  /* copy full array to buffer */
  plo[0] = 0;
  phi[0] = idim-1;
  plo[1] = 0;
  phi[1] = jdim-1;
  ga.get(plo,phi,buf,&jdim);
  ga.sync();
  ok = 1;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (buf[j+jdim*i] != static_cast<double>((j+plo[1]) + (i+plo[0])*dims[1])) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i,j, buf[j+jdim*i],
            static_cast<double>((j+plo[1]) + (i+plo[0])*dims[1]));
        ok = 0;
      }
    }
  }

  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Single large get test PASSES\n\n");
  } else if (chk == 0) {
    printf("\n Single large get test FAILS\n\n");
  }
  ga.clear();
  delete [] buf;
  env->finalize();
  MPI_Finalize();
  return 0;
}
