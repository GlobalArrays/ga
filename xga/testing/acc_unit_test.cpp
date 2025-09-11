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
  double  r_one = 1.0;
  dims[0] = DIM;
  dims[1] = 2*DIM;
  if (rank == 0) {
    printf("\nTesting ACC on a  %d x %d matrix",dims[0],dims[1]);
    printf(" running on %d processors\n",size);
  }
  XGA::GlobalArray<double> ga(group, ndim, dims);
  ga.allocate();

  /* initialize global array */
  int64_t lo[2], hi[2], ld;
  ga.distribution(rank,lo,hi);
  void *vptr;
  ga.accessPtr(lo, hi, &vptr, &ld);
  double *dptr = static_cast<double*>(vptr);
  int64_t idim = hi[0]-lo[0]+1;
  int64_t jdim = hi[1]-lo[1]+1;
  int64_t i, j;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      dptr[j+jdim*i] = static_cast<double>(j+lo[1] + (i+lo[0])*dims[1]);
    }
  }
  if (rank == 0) {
    printf("\n Testing acc on whole blocks\n");
  }
  int nghbr = (rank+1)%size;
  ga.distribution(nghbr,lo,hi);
  int64_t nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  double *buf = new double[nelems];
  /* initialize local buffer */
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      buf[j+jdim*i] = static_cast<double>(j+lo[1] + (i+lo[0])*dims[1]);
    }
  }
  ga.acc(lo,hi,buf,&jdim,&r_one);
  ga.sync();
  ga.distribution(rank,lo,hi);
  ga.accessPtr(lo, hi, &vptr, &ld);
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  int ok = 1;
  int chk;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (dptr[j+jdim*i] != static_cast<double>(2*(j+lo[1]+(i+lo[0])*dims[1]))) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i+lo[0],j+lo[1],dptr[j+jdim*i],
            static_cast<double>(2*(j+lo[1] + (i+lo[0])*dims[1])));
        ok = 0;
      }
    }
  }

  MPI_Comm comm = group->MPIComm();
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Full block acc test PASSES\n");
  } else if (chk == 0) {
    printf("\n Full block acc test FAILS\n");
  }
  if (rank == 0) {
    printf("\n Testing acc on partial blocks\n");
    printf("\n Zero values in array\n");
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
    for (i=plo[0]; i<=phi[0]; i++) {
      ii = i-plo[0];
      for (j=plo[1]; j<=phi[1]; j++) {
        jj = j-plo[1];
        buf[jj+jdim*ii] = static_cast<double>(j + i*dims[1]);
      }
    }
    ga.acc(plo,phi,buf,&jdim,&r_one);
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
      if (dptr[j+jdim*i] != static_cast<double>(3*(j+lo[1]+(i+lo[0])*dims[1]))) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i+lo[0],j+lo[1], dptr[j+jdim*i],
            static_cast<double>(3*(j+lo[1] + (i+lo[0])*dims[1])));
        ok = 0;
      }
    }
  }

  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Partial block acc test PASSES\n");
  } else if (chk == 0) {
    printf("\n Partial block acc test FAILS\n");
  }
  if (rank == 0) {
    printf("\n Testing single large acc to whole array\n");
    printf("\n Zero values in array\n");
  }
  ga.sync();
  nghbr = (rank+1)%size;
  idim = dims[0];
  jdim = dims[1];
  nelems = idim*jdim;
  /* initialize local buffer with values for whole array*/
  delete [] buf;
  buf = new double[nelems];

  if (rank == 0) {
    for (i=0; i<idim; i++) {
      for (j=0; j<jdim; j++) {
        buf[j+jdim*i] = static_cast<double>(j+jdim*i);
      }
    }
    /* copy buffer to full array */
    plo[0] = 0;
    phi[0] = idim-1;
    plo[1] = 0;
    phi[1] = jdim-1;
    ga.acc(plo,phi,buf,&jdim,&r_one);
  }
  ga.sync();
  ga.distribution(rank,lo,hi);
  ga.accessPtr(lo, hi, &vptr, &ld);
  dptr = static_cast<double*>(vptr);
  ok = 1;
  idim = (hi[0]-lo[0]+1);
  jdim = (hi[1]-lo[1]+1);
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (dptr[j+jdim*i] != static_cast<double>(4*((j+lo[1])+(i+lo[0])*dims[1]))) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i,j, dptr[j+jdim*i],
            static_cast<double>(4*((j+lo[1]) + (i+lo[0])*dims[1])));
        ok = 0;
      }
    }
  }

  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Single large acc test PASSES\n\n");
  } else if (chk == 0) {
    printf("\n Single large acc test FAILS\n\n");
  }
  ga.clear();
  delete [] buf;
  env->finalize();
  MPI_Finalize();
  return 0;
}
