#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#define DIM  2048
template<typename idx_type, typename data_type>
void get_test()
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

  idx_type lo[2], hi[2], ld;
  if (rank == 0) {
    printf("\n Testing get on whole blocks\n");
  }
  ga.distribution(rank,lo,hi);
  void *vptr;
  ga.accessPtr(lo, hi, &vptr, &ld);
  data_type *dptr = static_cast<data_type*>(vptr);
  /* initialize global array */
  idx_type idim = hi[0]-lo[0]+1;
  idx_type jdim = hi[1]-lo[1]+1;
  idx_type i, j;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      dptr[j+jdim*i] = static_cast<data_type>(j+lo[1] + (i+lo[0])*dims[1]);
    }
  }
  ga.sync();
  int nghbr = (rank+1)%size;
  ga.distribution(nghbr,lo,hi);
  idx_type nelems = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  data_type *buf = new data_type[nelems];
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  ga.get(lo,hi,buf,&jdim);
  idim = hi[0]-lo[0]+1;
  jdim = hi[1]-lo[1]+1;
  int ok = 1;
  int chk;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      if (buf[j+jdim*i] != static_cast<data_type>(j+lo[1] + (i+lo[0])*dims[1])) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i+lo[0],j+lo[1],buf[j+jdim*i],
            static_cast<data_type>(j+lo[1] + (i+lo[0])*dims[1]));
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
    data_type *tbuf = buf + plo[1]-lo[1]+(plo[0]-lo[0])*jdim;
    ga.get(plo,phi,tbuf,&jdim);
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
  buf = new data_type[nelems];

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
      if (buf[j+jdim*i] != static_cast<data_type>((j+plo[1]) + (i+plo[0])*dims[1])) {
        printf("p[%d] Check fails for i: %d j: %d actual: %f expected: %f\n",
            wrank,i,j, buf[j+jdim*i],
            static_cast<data_type>((j+plo[1]) + (i+plo[0])*dims[1]));
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
    printf("\nTesting GET on a  %d x %d matrix",dims[0],dims[1]);
    printf(" running on %d processors\n",size);
  }
  if (rank == 0) {
    printf("\nTesting GET for ints and int64_t indices\n");
  }
  get_test<int64_t,int>();
  if (rank == 0) {
    printf("\nTesting GET for longs and int64_t indices\n");
  }
  get_test<int64_t,long>();
  if (rank == 0) {
    printf("\nTesting GET for long longs and int64_t indices\n");
  }
  get_test<int64_t,long long>();
  if (rank == 0) {
    printf("\nTesting GET for floats and int64_t indices\n");
  }
  get_test<int64_t,float>();
  if (rank == 0) {
    printf("\nTesting GET for doubles and int64_t indices\n");
  }
  get_test<int64_t,double>();
  if (rank == 0) {
    printf("\nTesting GET for complex floats and int64_t indices\n");
  }
  get_test<int64_t,std::complex<float> >();
  if (rank == 0) {
    printf("\nTesting GET for complex doubles and int64_t indices\n");
  }
  get_test<int64_t,std::complex<double> >();

  if (rank == 0) {
    printf("\nTesting GET for ints and int indices\n");
  }
  get_test<int,int>();
  if (rank == 0) {
    printf("\nTesting GET for longs and int indices\n");
  }
  get_test<int,long>();
  if (rank == 0) {
    printf("\nTesting GET for long longs and int indices\n");
  }
  get_test<int,long long>();
  if (rank == 0) {
    printf("\nTesting GET for floats and int indices\n");
  }
  get_test<int,float>();
  if (rank == 0) {
    printf("\nTesting GET for doubles and int indices\n");
  }
  get_test<int,double>();
  if (rank == 0) {
    printf("\nTesting GET for complex floats and int indices\n");
  }
  get_test<int,std::complex<float> >();
  if (rank == 0) {
    printf("\nTesting GET for complex doubles and int indices\n");
  }
  get_test<int,std::complex<double> >();
  env->finalize();
  MPI_Finalize();
  return 0;
}
