#include "xga_interface.hpp"
#include "xga_group.hpp"
#include "xga_environment.hpp"
#include <iostream>

#include "test_utilities.hpp"
#define DIM  2048
#define DIM3  128
#define BLOCKDIM 31
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
  ga.releasePtr(lo, hi);

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
      dptr[j+jdim*i] = static_cast<data_type>(0);
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
  ga.releasePtr(lo, hi);

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
      dptr[j+jdim*i] = static_cast<data_type>(0);
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
  ga.releasePtr(lo, hi);

  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n Single large put test PASSES\n");
  } else if (chk == 0) {
    printf("\n Single large put test FAILS\n");
  }
  delete [] buf;
  ga.clear();

  if (rank == 0) {
    printf("\n Testing put to three dimensional array\n");
    printf("\n Zero values in array\n");
  }
  int three = 3;
  idx_type dims3d[3], hi3[3], lo3[3], ld3[2];
  dims3d[0] = DIM3;
  dims3d[1] = 2*DIM3;
  dims3d[2] = 4*DIM3;
  XGA::GlobalArray<data_type> ga3d(group, three, dims3d);
  ga3d.allocate();
  nghbr = (rank+1)%size;
  ga3d.distribution(rank,lo3,hi3);
  ga3d.accessPtr(lo3, hi3, &vptr, ld3);
  dptr = static_cast<data_type*>(vptr);
  idx_type k, kdim;
  idim = hi3[0]-lo3[0]+1;
  jdim = hi3[1]-lo3[1]+1;
  kdim = hi3[2]-lo3[2]+1;
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      for (k=0; k<kdim; k++) {
        dptr[k+j*kdim+i*kdim*jdim] = static_cast<data_type>(0);
      }
    }
  }
  ga3d.releasePtr(lo3, hi3);
  ga3d.sync();

  ga3d.distribution(nghbr,lo3,hi3);
  idim = hi3[0]-lo3[0]+1;
  jdim = hi3[1]-lo3[1]+1;
  kdim = hi3[2]-lo3[2]+1;
  nelems = idim*jdim*kdim;
  /* initialize local buffer*/
  buf = new data_type[nelems];

  /* divide each processor block into 8 sub-blocks */
  idx_type plo3[3], phi3[3];
  ld3[0] = jdim;
  ld3[1] = kdim;
  for (n=0; n<8; n++) {
    if (n==0) {
      plo3[0] = lo3[0];
      phi3[0] = lo3[0]+(hi3[0]-lo3[0])/2;
      plo3[1] = lo3[1];
      phi3[1] = lo3[1]+(hi3[1]-lo3[1])/2;
      plo3[2] = lo3[2];
      phi3[2] = lo3[2]+(hi3[2]-lo3[2])/2;
    } else if (n==1) {
      plo3[0] = lo3[0]+(hi3[0]-lo3[0])/2 + 1;
      phi3[0] = hi3[0];
      plo3[1] = lo3[1];
      phi3[1] = lo3[1]+(hi3[1]-lo3[1])/2;
      plo3[2] = lo3[2];
      phi3[2] = lo3[2]+(hi3[2]-lo3[2])/2;
    } else if (n==2) {
      plo3[0] = lo3[0];
      phi3[0] = lo3[0]+(hi3[0]-lo3[0])/2;
      plo3[1] = lo3[1]+(hi3[1]-lo3[1])/2 + 1;
      phi3[1] = hi3[1];
      plo3[2] = lo3[2];
      phi3[2] = lo3[2]+(hi3[2]-lo3[2])/2;
    } else if (n==3) {
      plo3[0] = lo3[0]+(hi3[0]-lo3[0])/2 + 1;
      phi3[0] = hi3[0];
      plo3[1] = lo3[1]+(hi3[1]-lo3[1])/2 + 1;
      phi3[1] = hi3[1];
      plo3[2] = lo3[2];
      phi3[2] = lo3[2]+(hi3[2]-lo3[2])/2;
    } else if (n==4) {
      plo3[0] = lo3[0];
      phi3[0] = lo3[0]+(hi3[0]-lo3[0])/2;
      plo3[1] = lo3[1];
      phi3[1] = lo3[1]+(hi3[1]-lo3[1])/2;
      plo3[2] = lo3[2]+(hi3[2]-lo3[2])/2 + 1;
      phi3[2] = hi3[2];
    } else if (n==5) {
      plo3[0] = lo3[0]+(hi3[0]-lo3[0])/2 + 1;
      phi3[0] = hi3[0];
      plo3[1] = lo3[1];
      phi3[1] = lo3[1]+(hi3[1]-lo3[1])/2;
      plo3[2] = lo3[2]+(hi3[2]-lo3[2])/2 + 1;
      phi3[2] = hi3[2];
    } else if (n==6) {
      plo3[0] = lo3[0];
      phi3[0] = lo3[0]+(hi3[0]-lo3[0])/2;
      plo3[1] = lo3[1]+(hi3[1]-lo3[1])/2 + 1;
      phi3[1] = hi3[1];
      plo3[2] = lo3[2]+(hi3[2]-lo3[2])/2 + 1;
      phi3[2] = hi3[2];
    } else if (n==7) {
      plo3[0] = lo3[0]+(hi3[0]-lo3[0])/2 + 1;
      phi3[0] = hi3[0];
      plo3[1] = lo3[1]+(hi3[1]-lo3[1])/2 + 1;
      phi3[1] = hi3[1];
      plo3[2] = lo3[2]+(hi3[2]-lo3[2])/2 + 1;
      phi3[2] = hi3[2];
    }
    idx_type ii, jj, kk;
    for (i=plo3[0]; i<=phi3[0]; i++) {
      ii = i-plo3[0];
      for (j=plo3[1]; j<=phi3[1]; j++) {
        jj = j-plo3[1];
        for (k=plo3[2]; k<=phi3[2]; k++) {
          kk = k-plo3[2];
          buf[kk+jj*kdim+ii*kdim*jdim]
            = static_cast<data_type>(k + j*dims3d[2] + i*dims3d[2]*dims3d[1]);
        }
      }
    }
    ga3d.put(plo3,phi3,buf,ld3);
  }


  ga3d.sync();
  ga3d.distribution(rank,lo3,hi3);
  ga3d.accessPtr(lo3, hi3, &vptr, ld3);
  dptr = static_cast<data_type*>(vptr);
  ok = 1;
  idim = (hi3[0]-lo3[0]+1);
  jdim = (hi3[1]-lo3[1]+1);
  kdim = (hi3[2]-lo3[2]+1);
  for (i=0; i<idim; i++) {
    for (j=0; j<jdim; j++) {
      for (k=0; k<kdim; k++) {
        if (dptr[k+j*kdim+i*kdim*jdim]
            != static_cast<data_type>(k+lo3[2]+(j+lo3[1])*dims3d[2]
              + (i+lo3[0])*dims3d[2]*dims3d[1])) {
          printf("p[%d] Check fails for i: %d j: %d k: %d actual: %f expected: %f\n",
              wrank,i,j,k,dptr[k+j*kdim+i*kdim*jdim],
              static_cast<data_type>(k+lo3[2]+(j+lo3[1])*dims3d[2]
                + (i+lo3[0])*dims3d[2]*dims3d[1]));
          ok = 0;
        }
      }
    }
  }
  ga3d.releasePtr(lo3, hi3);

  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n 3D put test PASSES\n");
  } else if (chk == 0) {
    printf("\n 3D put test FAILS\n");
  }
  delete [] buf;
  ga3d.clear();

  /* Test lapack data layout */
  int pdims[3];
  factor(3, size, pdims);
  if (rank == 0) {
    printf("\n Testing put to three dimensional array\n");
    printf(" with ScaLAPACK data layout and a %d x %d x %d"
        " proc grid layout\n",pdims[0],pdims[1],pdims[2]);
  }
  XGA::GlobalArray<data_type> gala(group, three, dims3d);
  idx_type block_dims[3];
  block_dims[0] = BLOCKDIM;
  block_dims[1] = BLOCKDIM;
  block_dims[2] = BLOCKDIM;
  gala.setBlockLayout(block_dims, pdims);
  gala.allocate();
  idim = static_cast<idx_type>(static_cast<double>(dims3d[0])
      /static_cast<double>(pdims[0]));
  jdim = static_cast<idx_type>(static_cast<double>(dims3d[1])
      /static_cast<double>(pdims[1]));
  kdim = static_cast<idx_type>(static_cast<double>(dims3d[2])
      /static_cast<double>(pdims[2]));
  buf = new data_type[idim*jdim*kdim];
  /* find proc grid coordinates of this processor */
  int ix, iy, iz;
  n = rank;
  iz = n%pdims[2];
  n = (n-iz)/pdims[2];
  iy = n%pdims[1];
  ix = (n-iy)/pdims[1];
  /* calculate bounds of block used to initialize global array */
  lo3[0] = ix*idim;
  lo3[1] = iy*jdim;
  lo3[2] = iz*kdim;
  if (ix < pdims[0]-1) {
    hi3[0] = (ix+1)*idim-1;
  } else {
    hi3[0] = dims3d[0]-1;
  }
  if (iy < pdims[1]-1) {
    hi3[1] = (iy+1)*jdim-1;
  } else {
    hi3[1] = dims3d[1]-1;
  }
  if (iz < pdims[2]-1) {
    hi3[2] = (iz+1)*kdim-1;
  } else {
    hi3[2] = dims3d[2]-1;
  }
  /* initialize local buffer */
  for (i=lo3[0]; i<=hi3[0]; i++) {
    for (j=lo3[1]; j<=hi3[1]; j++) {
      for (k=lo3[2]; k<=hi3[2]; k++) {
        buf[k-lo3[2]+(j-lo3[1])*kdim+(i-lo3[0])*kdim*jdim]
          = static_cast<data_type>(k+j*dims3d[2]+i*dims3d[2]*dims3d[1]);
      }
    }
  }
  ld3[0] = jdim;
  ld3[1] = kdim;
  gala.put(lo3, hi3, buf, ld3);
  gala.sync();
  /* check results. Start by finding number of blocks in each direction */
  int nx, ny, nz;
  nx = dims3d[0]/block_dims[0];
  if (nx*block_dims[0] < dims3d[0]) nx++;
  ny = dims3d[1]/block_dims[1];
  if (ny*block_dims[1] < dims3d[1]) ny++;
  nz = dims3d[2]/block_dims[2];
  if (nz*block_dims[2] < dims3d[2]) nz++;
  /* loop over all blocks held by this process */
  int index[3];
  ok = true;
  int chkcnt = 0;
  for (i=ix; i<nx; i+=pdims[0]) {
    index[0] = i;
    lo3[0] = i*block_dims[0];
    hi3[0] = (i+1)*block_dims[0]-1;
    if (hi3[0] >= dims3d[0]) hi3[0] = dims3d[0]-1;
    for (j=iy; j<ny; j+=pdims[1]) {
      index[1] = j;
      lo3[1] = j*block_dims[1];
      hi3[1] = (j+1)*block_dims[1]-1;
      if (hi3[1] >= dims3d[1]) hi3[1] = dims3d[1]-1;
      for (k=iz; k<nz; k+=pdims[2]) {
        index[2] = k;
        lo3[2] = k*block_dims[2];
        hi3[2] = (k+1)*block_dims[2]-1;
        if (hi3[2] >= dims3d[2]) hi3[2] = dims3d[2]-1;
        gala.accessBlockGridPtr(index,&vptr,ld3);
        dptr = static_cast<data_type*>(vptr);
        int l, m;
        for (l=lo3[0]; l<=hi3[0]; l++) {
          for (m=lo3[1]; m<=hi3[1]; m++) {
            for (n=lo3[2]; n<=hi3[2]; n++) {
              if (dptr[n-lo3[2]+(m-lo3[1])*ld3[1]+(l-lo3[0])*ld3[0]*ld3[1]]
                  != static_cast<data_type>(n+m*dims3d[2]
                    +l*dims3d[2]*dims3d[1])) {
                if (ok) printf("p[%d] Check fails for ijk: [%d:%d:%d]"
                    " lmn: [%d:%d:%d] actual: %f expected: %f\n",
                    rank,i,j,k,l,m,n,
                    static_cast<data_type>(dptr[n-lo3[2]+(m-lo3[1])*ld3[1]
                      +(l-lo3[0])*ld3[0]*ld3[1]]),
                    static_cast<data_type>(n+m*dims3d[2]
                      +l*dims3d[2]*dims3d[1]));
                if (ok) printf("p[%d] number of successful checks: %d\n",
                    rank,chkcnt);
                ok = false;
              }
              else chkcnt++;
            }
          }
        }
        gala.releaseBlockGridPtr(index);
      }
    }
  }
  MPI_Allreduce(&ok, &chk, 1, MPI_INT, MPI_PROD, comm);
  if (chk==1 && rank == 0) {
    printf("\n ScaLAPACK layout put test PASSES\n\n");
  } else if (chk == 0) {
    printf("\n ScaLAPACK layout put test FAILS\n\n");
  }
  delete [] buf;
  gala.clear();
}

int main(int argc, char **argv)
{
  XGA::Environment *env = XGA::Environment::instance(&argc,&argv);
  XGA::Group *group = env->getWorldGroup();
  int rank = group->rank();
  int size = group->size();
  int64_t dims[2],dims3d[3];
  dims[0] = DIM;
  dims[1] = 2*DIM;
  dims3d[0] = DIM3;
  dims3d[1] = 2*DIM3;
  dims3d[2] = 4*DIM3;
  if (rank == 0) {
    printf("\nTesting PUT on a 2D %ld x %ld matrix and a\n",dims[0],dims[1]);
    printf(" 3D %ld x %ld x %ld array",dims3d[0],dims3d[1],dims3d[2]);
    printf(" running on %d processors\n",size);
  }
  if (rank == 0) {
    printf("\nTesting PUT for ints and int64_t indices\n");
  }
  put_test<int64_t,int>();
#if 1
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
#endif
  if (rank == 0) {
    printf("\nCompleted all tests\n");
  }
  env->finalize();
  MPI_Finalize();
  return 0;
}
