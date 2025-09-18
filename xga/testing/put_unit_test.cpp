#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#define DIM  2048
template <typename idx_type, typename data_type>
void put_test()
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

  /* initialize global array using put */
  idx_type lo[2], hi[2], ld;
  if (rank == 0) {
    printf("\n Testing put on whole blocks\n");
  }
  int nghbr = (rank+1)%size;
  ga.distribution(nghbr,lo,hi);
  idx_type nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  data_type *buf = new data_type[nelems];
  /* initialize local buffer */
  idx_type idim = hi[0]-lo[0]+1;
  idx_type jdim = hi[1]-lo[1]+1;
  idx_type i, j;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      buf[j+jdim*i] = static_cast<data_type>(j+lo[1] + (i+lo[0])*dims[1]);
    }
  }
  ga.put(lo,hi,buf,&jdim);
  ga.sync();
  ga.distribution(rank,lo,hi);
  void *vptr;
  ga.accessPtr(lo, hi, &vptr, &ld);
  data_type *dptr = static_cast<data_type*>(vptr);
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  int ok = 1;
  int chk;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (dptr[j+jdim*i] != static_cast<data_type>(j+lo[1] + (i+lo[0])*dims[1])) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i+lo[0],j+lo[1],dptr[j+jdim*i],
            static_cast<data_type>(j+lo[1] + (i+lo[0])*dims[1]));
        ok = 0;
      }
    }
  }

  MPI_Comm comm = group->MPIComm();
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Full block put test PASSES\n");
  } else if (chk == 0) {
    printf("\n Full block put test FAILS\n");
  }
  if (rank == 0) {
    printf("\n Testing put on partial blocks\n");
    printf("\n Zero values in array\n");
  }
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      dptr[j+jdim*i] = 0.0;
    }
  }
  ga.sync();
  idx_type plo[2], phi[2];
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
    idx_type ii, jj;
    for (i=plo[0]; i<=phi[0]; i++) {
      ii = i-plo[0];
      for (j=plo[1]; j<=phi[1]; j++) {
        jj = j-plo[1];
        buf[jj+jdim*ii] = static_cast<data_type>(j + i*dims[1]);
      }
    }
    ga.put(plo,phi,buf,&jdim);
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

  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Partial block put test PASSES\n");
  } else if (chk == 0) {
    printf("\n Partial block put test FAILS\n");
  }
  if (rank == 0) {
    printf("\n Testing single large put to whole array\n");
    printf("\n Zero values in array\n");
  }
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      dptr[j+jdim*i] = 0.0;
    }
  }
  ga.sync();
  nghbr = (rank+1)%size;
  idim = dims[0];
  jdim = dims[1];
  nelems = idim*jdim;
  /* initialize local buffer with values for whole array*/
  delete [] buf;
  buf = new data_type[nelems];

  if (rank == 0) {
    for (i=0; i<idim; i++) {
      for (j=0; j<jdim; j++) {
        buf[j+jdim*i] = static_cast<data_type>(j+jdim*i);
      }
    }
    /* copy buffer to full array */
    plo[0] = 0;
    phi[0] = idim-1;
    plo[1] = 0;
    phi[1] = jdim-1;
    ga.put(plo,phi,buf,&jdim);
  }
  ga.sync();
  ga.distribution(rank,lo,hi);
  ga.accessPtr(lo, hi, &vptr, &ld);
  dptr = static_cast<data_type*>(vptr);
  ok = 1;
  idim = (hi[0]-lo[0]+1);
  jdim = (hi[1]-lo[1]+1);
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (dptr[j+jdim*i] != static_cast<data_type>((j+lo[1]) + (i+lo[0])*dims[1])) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i,j, dptr[j+jdim*i],
            static_cast<data_type>((j+lo[1]) + (i+lo[0])*dims[1]));
        ok = 0;
      }
    }
  }

  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Single large put test PASSES\n\n");
  } else if (chk == 0) {
    printf("\n Single large put test FAILS\n\n");
  }
  ga.clear();
  delete [] buf;
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
    printf("\nTesting PUT on a  %d x %d matrix",dims[0],dims[1]);
    printf(" running on %d processors\n",size);
  }
  if (rank == 0) {
    printf("\nTesting PUT for ints and int64_t indices\n");
  }
  put_test<int64_t,int>();
  if (rank == 0) {
    printf("\nTesting PUT for longs and int64_t indices\n");
  }
  put_test<int64_t,long>();
  if (rank == 0) {
    printf("\nTesting PUT for long longs and int64_t indices\n");
  }
  put_test<int64_t,long long>();
  if (rank == 0) {
    printf("\nTesting PUT for floats and int64_t indices\n");
  }
  put_test<int64_t,float>();
  if (rank == 0) {
    printf("\nTesting PUT for doubles and int64_t indices\n");
  }
  put_test<int64_t,double>();
  if (rank == 0) {
    printf("\nTesting PUT for complex floats and int64_t indices\n");
  }
  put_test<int64_t,std::complex<float> >();
  if (rank == 0) {
    printf("\nTesting PUT for complex doubles and int64_t indices\n");
  }
  put_test<int64_t,std::complex<double> >();
  if (rank == 0) {
    printf("\nTesting PUT for ints and int indices\n");
  }
  put_test<int,int>();
  if (rank == 0) {
    printf("\nTesting PUT for longs and int indices\n");
  }
  put_test<int,long>();
  if (rank == 0) {
    printf("\nTesting PUT for long longs and int indices\n");
  }
  put_test<int,long long>();
  if (rank == 0) {
    printf("\nTesting PUT for floats and int indices\n");
  }
  put_test<int,float>();
  if (rank == 0) {
    printf("\nTesting PUT for doubles and int indices\n");
  }
  put_test<int,double>();
  if (rank == 0) {
    printf("\nTesting PUT for complex floats and int indices\n");
  }
  put_test<int,std::complex<float> >();
  if (rank == 0) {
    printf("\nTesting PUT for complex doubles and int indices\n");
  }
  put_test<int,std::complex<double> >();
  env->finalize();
  MPI_Finalize();
}
