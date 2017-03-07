#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

#define USE_HYPRE 0
#define IMAX 200
#define JMAX 200
#define KMAX 200
#define LMAX IMAX*JMAX*KMAX

#define bb_a(ib) bb_v(bb_i + (ib))
#define cc_a(ib) cc_v(cc_i + (ib))

#define Integer int
#define MAX_FACTOR 10000

/**
 * If this test is built standalone and then linked to the Hypre library,
 * the Hypre sparse matrix-vector multiply routines can be used to check the
 * correctness of the answers
 */
#if USE_HYPRE
#include "HYPRE.h"
#include "HYPRE_struct_mv.h"
#include "mpi.h"
#endif

void grid_factor(int p, int xdim, int ydim, int zdim, int *idx, int *idy, int *idz) {
  int i, j; 
  int ip, ifac, pmax, prime[MAX_FACTOR];
  int fac[MAX_FACTOR];
  int ix, iy, iz, ichk;

  i = 1;
/**
 *   factor p completely
 *   first, find all prime numbers, besides 1, less than or equal to 
 *   the square root of p
 */
  ip = (int)(sqrt((double)p))+1;
  pmax = 0;
  for (i=2; i<=ip; i++) {
    ichk = 1;
    for (j=0; j<pmax; j++) {
      if (i%prime[j] == 0) {
        ichk = 0;
        break;
      }
    }
    if (ichk) {
      pmax = pmax + 1;
      if (pmax > MAX_FACTOR) printf("Overflow in grid_factor\n");
      prime[pmax-1] = i;
    }
  }
/**
 *   find all prime factors of p
 */
  ip = p;
  ifac = 0;
  for (i=0; i<pmax; i++) {
    while(ip%prime[i] == 0) {
      ifac = ifac + 1;
      fac[ifac-1] = prime[i];
      ip = ip/prime[i];
    }
  }
/**
 *  p is prime
 */
  if (ifac==0) {
    ifac++;
    fac[0] = p;
  }
/**
 *    find three factors of p of approximately the
 *    same size
 */
  *idx = 1;
  *idy = 1;
  *idz = 1;
  for (i = ifac-1; i >= 0; i--) {
    ix = xdim/(*idx);
    iy = ydim/(*idy);
    iz = zdim/(*idz);
    if (ix >= iy && ix >= iz && ix > 1) {
      *idx = fac[i]*(*idx);
    } else if (iy >= ix && iy >= iz && iy > 1) {
      *idy = fac[i]*(*idy);
    } else if (iz >= ix && iz >= iy && iz > 1) {
      *idz = fac[i]*(*idz);
    } else {
      printf("Too many processors in grid factoring routine\n");
    }
  }
}

/**
 *  Short subroutine for multiplying sparse matrix block with vector segment
 */
void loc_matmul(double *a_mat, int *jvec, int *ivec,
                double *bvec, double *cvec, int nrows) {
  double tempc;
  int i, j, jj, jmin,jmax;
  for (i=0; i<nrows; i++) {
    jmin = ivec[i];
    jmax = ivec[i+1]-1;
    tempc = 0.0;
    for (j=jmin; j<jmax; j++) {
      jj = jvec[j];
      tempc = tempc + a_mat[j]*bvec[jj];
    }
    cvec[i] = cvec[i] + tempc;
  }
}
/**
 *   Random number generator
 */
double ran3(int *idum) {
  static int iff = 0;
  double randnum;
  if (idum < 0 || iff == 0) {
    iff = 1;
    srand((unsigned int)abs(*idum));
    *idum = 1;
  }
  randnum = ((double)rand())/((double)RAND_MAX);
  return randnum;
}

/**
 *  create a sparse matrix in compressed row form corresponding to a Laplacian
 *  differential operator using a 7-point stencil for a grid on a lattice of
 *  dimension idim x jdim x kdim grid points
 */
void create_laplace_mat(int idim, int jdim, int kdim, int pdi, int pdj, int pdk,
                        int *g_data, int *g_i, int *g_j, int *total_procs,
                        int **proclist, int **proc_inv, int **icnt, int **voffset,
                        int **nsize, int **offset, int *tsize, int **imapc)
{
/**
 *  idim: i-dimension of grid
 *  jdim: j-dimension of grid
 *  kdim: k-dimension of grid
 *  pdi: i-dimension of processor grid
 *  pdj: j-dimension of processor grid
 *  pdk: k-dimension of processor grid
 *  g_data: global array of values
 *  g_j: global array containing j indices (using local indices)
 *  g_i: global array containing starting location of each row in g_j
 *       (using local indices)
 *  total_procs: number of processors this proc interacts with
 *  proclist: list of processors that this process interacts with
 *  proc_inv: given a processor, map it to position in proclist
 *  icnt: number of elements in submatrix
 *  voffset: offset for each submatrix block
 *  nsize: number of elements in right hand side vector on each processor
 *  offset: starting index of right hand side vector on each processor
 *  tsize: total number of non-zero elements in matrix
 *  imapc: map array for vectors
 */
  int ltotal_procs, ltsize;
  int *lproclist, *lproc_inv,  *lvoffset, *lnsize, *loffset, *licnt, *limapc;
  int nprocs, me, imin, imax, jcnt, jmin, jmax;
  int ix, iy, iz, idx;
  double x, dr;
  double *rval;
  int isize, idbg;
  int *jval,  *ival,  *ivalt;
  int i, j, k, itmp, one, tlo, thi, ld;
  int idum, ntot, indx, nghbrs[7], ncnt;
  int ixn[7],iyn[7],izn[7], procid[7];
  int status;
  int lo[3], hi[3], ip, jp, kp, ldi, ldj, jdx, joff;
  int il, jl, kl, ldmi, ldpi, ldmj, ldpj;
  int *xld, *yld, *zld, *tmapc;
  int *ecnt, *total_distr;
  int total_max;
  FILE *fp, *fopen();

  me = GA_Nodeid();
  nprocs = GA_Nnodes();
  idum = -(12345+me);
  x = ran3(&idum);
  one = 1;

  if (me == 0) {
    printf("\n Dimension of grid: \n\n");
    printf(" I Dimension: %d\n",idim);
    printf(" J Dimension: %d\n",jdim);
    printf(" K Dimension: %d\n\n",kdim);
  }
/**
 * Find position of processor in processor grid and calulate minimum
 * and maximum values of indices
 */
  i = me;
  ip = i%pdi;
  i = (i-ip)/pdi;
  jp = i%pdj;
  kp = (i-jp)/pdj;
 
  lo[0] = (int)((((double)idim)*((double)ip))/((double)pdi));
  if (ip < pdi-1) {
    hi[0] = (int)((((double)idim)*((double)(ip+1)))/((double)pdi))-1;
  } else {
    hi[0] = idim - 1;
  } 

  lo[1] = (int)((((double)jdim)*((double)jp))/((double)pdj));
  if (jp < pdj-1) {
    hi[1] = (int)((((double)jdim)*((double)(jp+1)))/((double)pdj))-1;
  } else {
    hi[1] = jdim - 1;
  } 

  lo[2] = (int)((((double)kdim)*((double)kp))/((double)pdk));
  if (kp < pdk-1) {
    hi[2] = (int)((((double)kdim)*((double)(kp+1)))/((double)pdk))-1;
  } else {
    hi[2] = kdim - 1;
  } 
 
/**
 * Determine stride lengths. Start with stride lengths for grid block
 * owned by this processor
 */
  ldi = hi[0]-lo[0]+1;
  ldj = hi[1]-lo[1]+1;
 
/**
 * Find stride lengths for blocks owned by other processors
 */
  xld = (int*)malloc(pdi*sizeof(int));
  for (i=0; i<pdi; i++) {
    if (i<pdi-1) {
      xld[i] = (int)((((double)idim)*((double)(i+1)))/((double)pdi));
    } else {
      xld[i] = idim;
    }
    xld[i] = xld[i] - (int)((((double)idim)*((double)(i)))/((double)pdi));
  }

  yld = (int*)malloc(pdj*sizeof(int));
  for (i=0; i<pdj; i++) {
    if (i<pdj-1) {
      yld[i] = (int)((((double)jdim)*((double)(i+1)))/((double)pdj));
    } else {
      yld[i] = jdim;
    }
    yld[i] = yld[i] - (int)((((double)jdim)*((double)(i)))/((double)pdj));
  }

  zld = (int*)malloc(pdk*sizeof(int));
  for (i=0; i<pdk; i++) {
    if (i<pdk-1) {
      zld[i] = (int)((((double)kdim)*((double)(i+1)))/((double)pdk));
    } else {
      zld[i] = jdim;
    }
    zld[i] = zld[i] - (int)((((double)kdim)*((double)(i)))/((double)pdk));
  }

/**
 *   Determine number of rows per processor
 */

  lnsize = (int*)malloc(nprocs*sizeof(int));
  *nsize = lnsize;
  loffset = (int*)malloc(nprocs*sizeof(int));
  *offset = loffset;
  for (i=0; i<nprocs; i++) {
    lnsize[i] = 0;
    loffset[i] = 0;
  }
  lnsize[me] = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1)*(hi[2]-lo[2]+1);
  GA_Igop(lnsize,nprocs,"+");
  loffset[0] = 0;
  for (i=1; i<nprocs; i++) {
    loffset[i] = loffset[i-1] + lnsize[i-1];
  }
 
  ntot = idim*jdim*kdim;
  NGA_Sync();
/**
 *   Scan over rows of lattice
 */
  imin = loffset[me];
  imax = loffset[me]+lnsize[me]-1;
/**
 *   Find out how many other processors couple to this row of blocks. Start by
 *   initializing ecnt array to zero
 */
  ecnt = (int*)malloc(nprocs*sizeof(int));
  for (i=0; i<nprocs; i++) {
    ecnt[i] = 0;
  }

/**
 * Loop over all rows owned by this processor
 */
  for (i=imin; i<=imax; i++) {
/**
 *   Compute local indices of grid point corresponding to row i (this
 *   corresponds to an element of the grid)
 */
    indx = i - imin;
    ix = indx%ldi;
    indx = (indx - ix)/ldi;
    iy = indx%ldj;
    iz = (indx - iy)/ldj;
    ix = ix + lo[0];
    iy = iy + lo[1];
    iz = iz + lo[2];
 
/**
 * Check all neighbors. Mark the ecnt element corresponding to the processor
 * that owns the element
 */
    ecnt[me] = ecnt[me] + 1;
    if (ix+1 <= idim-1) {
      if (ix+1 > hi[0]) {
        jdx = kp*pdi*pdj + jp*pdi + ip + 1;
        ecnt[jdx] = ecnt[jdx] + 1;
      } else {
        ecnt[me] = ecnt[me] + 1;
      }
    }
    if (ix-1 >= 0) {
      if (ix-1 < lo[0]) {
        jdx = kp*pdi*pdj + jp*pdi + ip - 1;
        ecnt[jdx] = ecnt[jdx] + 1;
      } else {
        ecnt[me] = ecnt[me] + 1;
      }
    }
    if (iy+1 <= jdim-1) {
      if (iy+1 > hi[1]) {
        jdx = kp*pdi*pdj + (jp+1)*pdi + ip;
        ecnt[jdx] = ecnt[jdx] + 1;
      } else {
        ecnt[me] = ecnt[me] + 1;
      }
    }
    if (iy-1 >= 0) {
      if (iy-1 < lo[1]) {
        jdx = kp*pdi*pdj + (jp-1)*pdi + ip;
        ecnt[jdx] = ecnt[jdx] + 1;
      } else {
        ecnt[me] = ecnt[me] + 1;
      }
    }
    if (iz+1 <= kdim-1) {
      if (iz+1 > hi[2]) {
        jdx = (kp+1)*pdi*pdj + jp*pdi + ip;
        ecnt[jdx] = ecnt[jdx] + 1;
      } else {
        ecnt[me] = ecnt[me] + 1;
      }
    }
    if (iz-1 >= 0) {
      if (iz-1 < lo[2]) {
        jdx = (kp-1)*pdi*pdj + jp*pdi + ip;
        ecnt[jdx] = ecnt[jdx] + 1;
      } else {
        ecnt[me] = ecnt[me] + 1;
      }
    }
  }

/**
 *  Create a list of processors that this processor is coupled to. Also count
 *  the total number of grid points that this processor couples to. This number
 *  is equal to the total number of matrix elements held by this processor.
 */
  ltotal_procs = 0;
  ncnt = 0;
  for (i=0; i<nprocs; i++) {
    if (ecnt[i] > 0) {
      ltotal_procs += 1;
      ncnt += ecnt[i];
    }
  }
  *total_procs = ltotal_procs;
  lproclist = (int*)malloc(ltotal_procs*sizeof(int));
  *proclist = lproclist;
  lproc_inv = (int*)malloc(nprocs*sizeof(int));
  *proc_inv = lproc_inv;
  licnt = (int*)malloc(ltotal_procs*sizeof(int));
  *icnt = licnt;

/**
 * Set up conventional CSR arrays to hold portion of matrix owned by this
 * processor
 */
  rval = (double*)malloc(ncnt*sizeof(double));
  idbg = ncnt;
  jval = (int*)malloc(ncnt*sizeof(int));
  ival = (int*)malloc((imax-imin+2)*ltotal_procs*sizeof(int));
  ivalt = (int*)malloc((imax-imin+2)*ltotal_procs*sizeof(int));

  for (i=0; i<ncnt; i++) {
    rval[i] = 0.0;
    jval[i] = 0;
  }
  
  j = (imax-imin+2)*ltotal_procs;
  for (i=0; i<j; i++) {
    ival[i] = 0;
    ivalt[i] = 0;
  }

  j = 0;
  for (i=0; i<nprocs; i++) {
    lproc_inv[i] = -1;
  }
/**
 * voffset represents vertical partitions in the row block that divide it into
 * blocks that couple to grid elements on other processors
 */
  lvoffset = (int*)malloc(ltotal_procs*sizeof(int));
  *voffset = lvoffset;
  lvoffset[0] = 0;
  for (i=0; i<nprocs; i++) {
    if (ecnt[i] > 0) {
      lproclist[j] = i;
      if (j > 0) {
        lvoffset[j] = ecnt[lproclist[j-1]]+lvoffset[j-1];
      }
      lproc_inv[i] = j;
      j++;
    }
  }
  free(ecnt);

  isize = imax-imin+2;
  for (i=0; i<ltotal_procs; i++) {
    licnt[i] = 0;
  }
  ltsize = 0;
  for (i=imin; i<=imax; i++) {
/**
 *   compute local indices of grid point corresponding to row i
 */
    indx = i - imin;
    ix = indx%ldi;
    indx = (indx - ix)/ldi;
    iy = indx%ldj;
    iz = (indx - iy)/ldj;
    ix = ix + lo[0];
    iy = iy + lo[1];
    iz = iz + lo[2];
/**
 *   find locations of neighbors in 7-point stencil (if they are on the grid)
 */
    ncnt = 0;
    ixn[ncnt] = ix;
    iyn[ncnt] = iy;
    izn[ncnt] = iz;
    il = ix - lo[0];
    jl = iy - lo[1];
    kl = iz - lo[2];
    idx = kl*ldi*ldj + jl*ldi + il;
    nghbrs[ncnt] = idx;
    procid[ncnt] = me;
    if (ix+1 <= idim - 1) {
      ncnt++;
      ixn[ncnt] = ix + 1;
      iyn[ncnt] = iy;
      izn[ncnt] = iz;
      if (ix+1 > hi[0]) {
        jdx = kp*pdi*pdj + jp*pdi + ip + 1;
        il = 0;
        jl = iy - lo[1];
        kl = iz - lo[2];
        ldpi = xld[ip+1];
      } else {
        jdx = me;
        il = ix - lo[0] + 1;
        jl = iy - lo[1];
        kl = iz - lo[2];
        ldpi = ldi;
      }
      idx = kl*ldpi*ldj + jl*ldpi + il;
      nghbrs[ncnt] = idx;
      procid[ncnt] = jdx;
    }
    if (ix-1 >= 0) {
      ncnt++;
      ixn[ncnt] = ix - 1;
      iyn[ncnt] = iy;
      izn[ncnt] = iz;
      if (ix-1 < lo[0]) {
        jdx = kp*pdi*pdj + jp*pdi + ip - 1;
        il = xld[ip-1] - 1;
        jl = iy - lo[1];
        kl = iz - lo[2];
        ldmi = xld[ip-1];
      } else {
        jdx = me;
        il = ix - lo[0] - 1;
        jl = iy - lo[1];
        kl = iz - lo[2];
        ldmi = ldi;
      }
      idx = kl*ldmi*ldj + jl*ldmi + il;
      nghbrs[ncnt] = idx;
      procid[ncnt] = jdx;
    }
    if (iy+1 <= jdim-1) {
      ncnt++;
      ixn[ncnt] = ix; 
      iyn[ncnt] = iy + 1;
      izn[ncnt] = iz;
      if (iy+1 > hi[1]) {
        jdx = kp*pdi*pdj + (jp+1)*pdi + ip;
        il = ix - lo[0];
        jl = 0;
        kl = iz - lo[2];
        ldpj = yld[jp+1];
      } else {
        jdx = me;
        il = ix - lo[0];
        jl = iy - lo[1] + 1;
        kl = iz - lo[2];
        ldpj = ldj;
      }
      idx = kl*ldi*ldpj + jl*ldi + il;
      nghbrs[ncnt] = idx;
      procid[ncnt] = jdx;
    }
    if (iy-1 >= 0) {
      ncnt++;
      ixn[ncnt] = ix;
      iyn[ncnt] = iy - 1;
      izn[ncnt] = iz;
      if (iy-1 < lo[1]) {
        jdx = kp*pdi*pdj + (jp-1)*pdi + ip;
        il = ix - lo[0];
        jl = yld[jp-1] - 1;
        kl = iz - lo[2];
        ldmj = yld[jp-1];
      } else {
        jdx = me;
        il = ix - lo[0];
        jl = iy - lo[1] - 1;
        kl = iz - lo[2];
        ldmj = ldj;
      }
      idx = kl*ldi*ldmj + jl*ldi + il;
      nghbrs[ncnt] = idx;
      procid[ncnt] = jdx;
    }
    if (iz+1 <= kdim-1) {
      ncnt++;
      ixn[ncnt] = ix;
      iyn[ncnt] = iy;
      izn[ncnt] = iz + 1;
      if (iz+1 > hi[2]) {
        jdx = (kp+1)*pdi*pdj + jp*pdi + ip;
        il = ix - lo[0];
        jl = iy - lo[1];
        kl = 0;
      } else {
        jdx = me;
        il = ix - lo[0];
        jl = iy - lo[1];
        kl = iz - lo[2] + 1;
      }
      idx = kl*ldi*ldj + jl*ldi + il;
      nghbrs[ncnt] = idx;
      procid[ncnt] = jdx;
    }
    if (iz-1 >= 0) {
      ncnt++;
      ixn[ncnt] = ix;
      iyn[ncnt] = iy;
      izn[ncnt] = iz - 1;
      if (iz-1 < lo[2]) {
        jdx = (kp-1)*pdi*pdj + jp*pdi + ip;
        il = ix - lo[0];
        jl = iy - lo[1];
        kl = zld[kp-1] - 1;
      } else {
        jdx = me;
        il = ix - lo[0];
        jl = iy - lo[1];
        kl = iz - lo[2] - 1;
      }
      idx = kl*ldi*ldj + jl*ldi + il;
      nghbrs[ncnt] = idx;
      procid[ncnt] = jdx;
    }
/**
 *  sort j indices. This uses a simple bubble sort, but ncnt should be small. A
 *  more sophisticated approach could be taken if this is too time consuming.
 */
    ncnt++;
    for (j=0; j<ncnt; j++) {
      for (k=j+1; k<ncnt; k++) {
        if (nghbrs[j] > nghbrs[k]) {
          itmp = nghbrs[j];
          nghbrs[j] = nghbrs[k];
          nghbrs[k] = itmp;
          itmp = ixn[j];
          ixn[j] = ixn[k];
          ixn[k] = itmp;
          itmp = iyn[j];
          iyn[j] = iyn[k];
          iyn[k] = itmp;
          itmp = izn[j];
          izn[j] = izn[k];
          izn[k] = itmp;
          itmp = procid[j];
          procid[j] = procid[k];
          procid[k] = itmp;
        }
      }
    }
    for (k=0; k<ncnt; k++) {
      if (nghbrs[k] < 0 || nghbrs[k] >= ntot) {
        printf("p[%d] Invalid neighbor %d\n",me,nghbrs[k]);
      }
    }
/**
 *  create array elements
 *   for (j=0; j<ltotal_procs; j++) {
 *     printf("p[%d] lvoffset[%d]: %d\n",me,j,lvoffset[j]);
 *   }
 */
    for (j=0; j<ncnt; j++) {
      jdx = procid[j];
      idx = lproc_inv[jdx];
      if (ix == ixn[j] && iy == iyn[j] && iz == izn[j]) {
        rval[lvoffset[idx]+licnt[idx]] = 6.0;
      } else {
        rval[lvoffset[idx]+licnt[idx]] = -1.0;
      }
      if (lvoffset[idx]+licnt[idx]>=idbg) {
      }
      /* TODO: Check this carefully */
      jval[lvoffset[idx]+licnt[idx]] = nghbrs[j];
      ivalt[idx*isize+i-imin] = ivalt[idx*isize+i-imin]+1;
      licnt[idx]++;
    }
  }
  for (i=0; i<ltotal_procs; i++) {
    ival[i*isize] = lvoffset[i];
    for (j=1; j<isize; j++) {
      ival[i*isize+j] = ival[i*isize+j-1] + ivalt[i*isize+j-1];
    }
  }
  free(ivalt);
  isize = 0;
  for (i=0; i<ltotal_procs; i++) {
    isize = isize + licnt[i];
  }
  ltsize = isize;
  GA_Igop(&ltsize,one,"+");
/**
 *   Local portion of sparse matrix has been evaluated and decomposed into blocks
 *   that match partitioning of right hand side across processors. The following
 *   data is available at this point:
 *      1) total_procs: the number of processors that are coupled to this one via
 *         the sparse matrix
 *      2) proc_list(total_procs): a list of processor IDs that are coupled to
 *         this processor
 *      3) proc_inv(nprocs): The entry in proc_list that corresponds to a given
 *         processor. If the entry is zero then that processor does not couple to
 *         this processor.
 *      4) icnt(total_procs): The number of non-zero entries in the sparse matrix
 *         that couple the process represented by proc_list(j) to this process
 *      5) voffset(total_procs): The offsets for the non-zero data in the arrays
 *         rval and jval for the blocks that couple this processor to other
 *         processes in proc_list
 *      6) offset(nprocs): the offset array for the distributed right hand side
 *         vector
 *      7) nsize(nprocs): number of vector elements per processor for right hand
 *         side vector (equivalent to the number of rows per processor)
 * 
 *    These arrays describe how the sparse matrix is layed out both locally and
 *    across processors. In addition, the actual data for the distributed sparse
 *    matrix is found in the following arrays:
 *      1) rval: values of matrix for all blocks on this processor
 *      2) jval: j-indices of matrix for all blocks on this processor
 *      3) ival(total_procs*(nsize(me)+1)): starting index in rval and jval for
 *         each row in each block
 * 
 *    create global arrays to hold sparse matrix
 */
  tmapc = (int*)malloc(nprocs*sizeof(int));
  limapc = (int*)malloc(nprocs*sizeof(int));
  for (i=0; i<nprocs; i++) {
    tmapc[i] = 0;
  }
  for (i=0; i<ltotal_procs; i++) {
    tmapc[me] = tmapc[me] + licnt[i];
  }
  GA_Igop(tmapc,nprocs,"+");
  isize = 0;
  for (i=0; i<nprocs; i++) {
    isize += tmapc[i];
  }
  limapc[0] = 0;
  for (i=1; i<nprocs; i++) {
    limapc[i] = limapc[i-1] + tmapc[i-1];
  }

  *g_data = NGA_Create_handle();
  NGA_Set_data(*g_data,one,&isize,C_DBL);
  NGA_Set_irreg_distr(*g_data,limapc,&nprocs);
  status = NGA_Allocate(*g_data);
 
  *g_j = NGA_Create_handle();
  NGA_Set_data(*g_j,one,&isize,C_INT);
  NGA_Set_irreg_distr(*g_j,limapc,&nprocs);
  status = NGA_Allocate(*g_j);

  for (i=0; i<nprocs; i++) {
    tmapc[i] = 0;
  }
  tmapc[me] = tmapc[me] + (lnsize[me]+1)*ltotal_procs;
  GA_Igop(tmapc,nprocs,"+");
  ntot = 0;
  for (i=0; i<nprocs; i++) {
    ntot += tmapc[i];
  }
  limapc[0] = 0; 
  for (i=1; i<nprocs; i++) {
    limapc[i] = limapc[i-1] + tmapc[i-1];
  }

  *g_i = NGA_Create_handle();;
  NGA_Set_data(*g_i,one,&ntot,C_INT);
  NGA_Set_irreg_distr(*g_i,limapc,&nprocs);
  status = NGA_Allocate(*g_i);

  free(tmapc);

  GA_Sync();
/**
 *  fill global arrays with local data
 */
  NGA_Distribution(*g_data,me,&tlo,&thi);
  ld = thi-tlo+1;
  NGA_Put(*g_data, &tlo, &thi, rval, &ld);
  NGA_Put(*g_j, &tlo, &thi, jval, &ld);

  NGA_Distribution(*g_i,me,&tlo,&thi);
  ld = thi-tlo+1;
  NGA_Put(*g_i, &tlo, &thi, ival, &ld);

  GA_Sync();

  free(rval);
  free(jval);
  free(ival);
  free(xld);
  free(yld);
  free(zld);

  limapc[0] = 0;
  for (i=1; i<nprocs; i++) {
    limapc[i] = limapc[i-1] + lnsize[i-1];
  }
  *imapc = limapc;
  *tsize = isize;
  return;
}

int main(int argc, char **argv) {
  int nmax, nprocs, me;
  int g_a_data, g_a_i, g_a_j, isize;
  int g_b, g_c;
  int i, j, jj, k, one, jcnt;
  int chunk, kp1, ld;
  int *p_i, *p_j;
  double *p_data, *p_b, *p_c;
  double t_beg, t_beg2, t_ga_tot, t_get, t_mult, t_cnstrct, t_mpi_in, t_ga_in;
  double t_hypre_strct;
  double prdot, dotga, dothypre, tempc;
  double prtot, gatot, hypretot;
  int status;
  int idim, jdim, kdim, idum, memsize;
  int jmin, jmax, lsize, ntot;
  int heap=200000, fudge=100, stack=200000, ma_heap;
  double *cbuf, *vector;
  int pdi, pdj, pdk, ip, jp, kp, ncells;
  int lo[3],hi[3];
  int blo[3], bhi[3];
  int ld_a, ld_b, ld_c, ld_i, ld_j, irows, ioff, joff, total_procs;
  int iproc, iblock, btot;
  double *amat, *bvec;
  int *ivec, *jvec;
  int *proclist, *proc_inv, *icnt;
  int *voffset, *nsize, *offset, *mapc;
  int iloop;
  int LOOPNUM = 100;
/**
 *  Hypre declarations. These are only used if linking to Hypre library for
 *  performance testing
 */
  int ierr;
#if USE_HYPRE
  HYPRE_StructGrid grid;
  HYPRE_StructStencil stencil;
  HYPRE_StructMatrix matrix;
  HYPRE_StructVector vec_x, vec_y;
  int i4, j4, ndim, nelems, offsets[7][3];
  int stencil_indices[7], hlo[3], hhi[3];
  double weights[7];
  double *values;
  double alpha, beta;
  int *rows, *cols;
#endif
/**
 *  Intitialize a message passing library
 */
  one = 1;
  MP_INIT(argc,argv);
/**
 *    Initialize GA
 *
 *    There are 2 choices: ga_initialize or ga_initialize_ltd.
 *    In the first case, there is no explicit limit on memory usage.
 *    In the second, user can set limit (per processor) in bytes.
 */
  t_beg = GA_Wtime();
  GA_Initialize();
  t_ga_in = GA_Wtime() - t_beg;
  GA_Dgop(&t_ga_in,one,"+");
#if 0
  memsize = 500000000
  call ga_initialize_ltd(memsize)
  memsize = 500000
  status = ma_init(MT_DBL,memsize,memsize)
#endif

  t_ga_tot = 0.0;
  t_mult = 0.0;
  t_get = 0.0;
  t_hypre_strct = 0.0;
  prtot = 0.0;
  gatot = 0.0;
  hypretot = 0.0;

  me = GA_Nodeid();
  nprocs = GA_Nnodes();
  if (me == 0) {
   printf("Time to initialize GA:                                 %12.4f\n",
          t_ga_in/((double)nprocs));
  }
/**
 *  we can also use GA_set_memory_limit BEFORE first ga_create call
 */
  ma_heap = heap + fudge;
/* call GA_set_memory_limit(util_mdtob(ma_heap)) */
 
  if (me == 0) {
    printf("\nNumber of cores used: %d\n\nGA initialized\n\n",nprocs);
  }
/**
 *  Initialize the MA package
 *     MA must be initialized before any global array is allocated
 */
  if (!MA_init(MT_DBL, stack, ma_heap)) GA_Error("ma_init failed",-1);
/**
 *   create a sparse LMAX x LMAX matrix and two vectors of length
 *   LMAX. The matrix is stored in compressed row format.
 *   One of the vectors is filled with random data and the other
 *   is filled with zeros.
 */
  idim = IMAX;
  jdim = JMAX;
  kdim = KMAX;
  ntot = idim*jdim*kdim;
  if (me == 0) {
    printf("\nDimension of matrix: %d\n\n",ntot);
  }
  t_beg = GA_Wtime();
  grid_factor(nprocs,idim,jdim,kdim,&pdi,&pdj,&pdk);
  if (me == 0) {
    printf("\nProcessor grid configuration\n");
    printf("  PDX: %d\n",pdi);
    printf("  PDY: %d\n",pdj);
    printf("  PDZ: %d\n\n",pdk);
  }

  create_laplace_mat(idim,jdim,kdim,pdi,pdj,pdk,
                     &g_a_data,&g_a_i,&g_a_j,&total_procs,&proclist,&proc_inv,
                     &icnt,&voffset,&nsize,&offset,&isize,&mapc);
  t_cnstrct = GA_Wtime() - t_beg;
  if (me == 0) {
    printf("\nNumber of non-zero elements in compressed matrix: %d\n",isize);
  }

  g_b = GA_Create_handle();
  GA_Set_data(g_b,one,&ntot,C_DBL);
  GA_Set_irreg_distr(g_b,mapc,&nprocs);
  status = GA_Allocate(g_b);
/**
 *  fill g_b with random values
 */
  NGA_Distribution(g_b,me,blo,bhi);
  NGA_Access(g_b,blo,bhi,&p_b,&ld);
  ld = bhi[0]-blo[0]+1;
  btot = ld;
  vector = (double*)malloc(ld*sizeof(double));
  for (i=0; i<ld; i++) {
    idum  = 0;
    p_b[i] = ran3(&idum);
    vector[i] = p_b[i];
  }
  NGA_Release(g_b,blo,bhi);
  GA_Sync();

  g_c = GA_Create_handle();
  NGA_Set_data(g_c,one,&ntot,C_DBL);
  NGA_Set_irreg_distr(g_c,mapc,&nprocs);
  status = GA_Allocate(g_c);
  NGA_Zero(g_c);
#if USE_HYPRE
/**
 *  Assemble HYPRE grid and use that to create matrix. Start by creating
 *  grid partition
 */
  ndim = 3;
  i = me;
  ip = i%pdi;
  i = (i-ip)/pdi;
  jp = i%pdj;
  kp = (i-jp)/pdj;
  lo[0] = (int)(((double)idim)*((double)ip)/((double)pdi));
  if (ip < pdi-1) {
    hi[0] = (int)(((double)idim)*((double)(ip+1))/((double)pdi)) - 1;
  } else {
    hi[0] = idim - 1;
  }
  lo[1] = (int)(((double)jdim)*((double)jp)/((double)pdj));
  if (jp < pdj-1) {
    hi[1] = (int)(((double)jdim)*((double)(jp+1))/((double)pdj)) - 1;
  } else {
    hi[1] = jdim - 1;
  }
  lo[2] = (int)(((double)kdim)*((double)kp)/((double)pdk));
  if (kp < pdk-1) {
    hi[2] = (int)(((double)kdim)*((double)(kp+1))/((double)pdk)) - 1;
  } else {
    hi[2] = kdim - 1;
  }
/**
 *  Create grid
 */
  hlo[0] = lo[0];
  hlo[1] = lo[1];
  hlo[2] = lo[2];
  hhi[0] = hi[0];
  hhi[1] = hi[1];
  hhi[2] = hi[2];
  ierr = HYPRE_StructGridCreate(MPI_COMM_WORLD, ndim, &grid);
  ierr = HYPRE_StructGridSetExtents(grid, hlo, hhi);
  ierr = HYPRE_StructGridAssemble(grid);
/**
 *  Create stencil
 */
  offsets[0][0] = 0;
  offsets[0][1] = 0;
  offsets[0][2] = 0;

  offsets[1][0] = 1;
  offsets[1][1] = 0;
  offsets[1][2] = 0;

  offsets[2][0] = 0;
  offsets[2][1] = 1;
  offsets[2][2] = 0;

  offsets[3][0] = 0;
  offsets[3][1] = 0;
  offsets[3][2] = 1;

  offsets[4][0] = -1;
  offsets[4][1] = 0;
  offsets[4][2] = 0;

  offsets[5][0] = 0;
  offsets[5][1] = -1;
  offsets[5][2] = 0;

  offsets[6][0] = 0;
  offsets[6][1] = 0;
  offsets[6][2] = -1;

  nelems = 7;
  ierr = HYPRE_StructStencilCreate(ndim, nelems, &stencil);
  for (i=0; i<nelems; i++) {
    ierr = HYPRE_StructStencilSetElement(stencil, i, offsets[i]);
  }

  ncells = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1)*(hi[2]-lo[2]+1);
  jcnt = 7*ncells;
  values = (double*)malloc(jcnt*sizeof(double));
  jcnt = 0;
  weights[0] = 6.0;
  weights[1] = -1.0;
  weights[2] = -1.0;
  weights[3] = -1.0;
  weights[4] = -1.0;
  weights[5] = -1.0;
  weights[6] = -1.0;
  for (i=0; i<ncells; i++) {
    for (j=0; j<7; j++) {
      values[jcnt] = weights[j];
      jcnt++;
    }
  }

  ierr = HYPRE_StructMatrixCreate(MPI_COMM_WORLD, grid, stencil, &matrix);
  ierr = HYPRE_StructMatrixInitialize(matrix);
  for (i=0; i<7; i++) {
    stencil_indices[i] = i;
  }
  ierr = HYPRE_StructMatrixSetBoxValues(matrix, hlo, hhi, 7, stencil_indices, values);
  free(values);
/**
 *  Check all six sides of current box to see if any are boundaries.
 *  Set values to zero if they are.
 */
  if (hi[0] == idim-1) {
    ncells = (hi[1]-lo[1]+1)*(hi[2]-lo[2]+1);
    hlo[0] = idim-1;
    hhi[0] = idim-1;
    hlo[1] = lo[1];
    hhi[1] = hi[1];
    hlo[2] = lo[2];
    hhi[2] = hi[2];
    values = (double*)malloc(ncells*sizeof(double));
    for (i=0; i<ncells; i++) values[i] = 0.0;
    i4 = 1;
    j4 = 1;
    ierr = HYPRE_StructMatrixSetBoxValues(matrix, hlo, hhi, i4, &j4, values);
    free(values);
  }
  if (hi[1] == jdim-1) {
    ncells = (hi[0]-lo[0]+1)*(hi[2]-lo[2]+1);
    hlo[0] = lo[0];
    hhi[0] = hi[0];
    hlo[1] = jdim-1;
    hhi[1] = jdim-1;
    hlo[2] = lo[2];
    hhi[2] = hi[2];
    values = (double*)malloc(ncells*sizeof(double));
    for (i=0; i<ncells; i++) values[i] = 0.0;
    i4 = 1;
    j4 = 2;
    ierr = HYPRE_StructMatrixSetBoxValues(matrix, hlo, hhi, i4, &j4, values);
    free(values);
  } 
  if (hi[2] == kdim-1) {
    ncells = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
    hlo[0] = lo[0];
    hhi[0] = hi[0];
    hlo[1] = lo[1];
    hhi[1] = hi[1];
    hlo[2] = kdim-1;
    hhi[2] = kdim-1;
    values = (double*)malloc(ncells*sizeof(double));
    for (i=0; i<ncells; i++) values[i] = 0.0;
    i4 = 1;
    j4 = 3;
    ierr = HYPRE_StructMatrixSetBoxValues(matrix, hlo, hhi, i4, &j4, values);
    free(values);
  }
  if (lo[0] == 0) {
    ncells = (hi[1]-lo[1]+1)*(hi[2]-lo[2]+1);
    hlo[0] = 0;
    hhi[0] = 0;
    hlo[1] = lo[1];
    hhi[1] = hi[1];
    hlo[2] = lo[2];
    hhi[2] = hi[2];
    values = (double*)malloc(ncells*sizeof(double));
    for (i=0; i<ncells; i++) values[i] = 0.0;
    i4 = 1;
    j4 = 4;
    ierr = HYPRE_StructMatrixSetBoxValues(matrix, hlo, hhi, i4, &j4, values);
    free(values);
  }
  if (lo[1] == 0) {
    ncells = (hi[0]-lo[0]+1)*(hi[2]-lo[2]+1);
    hlo[0] = lo[0];
    hhi[0] = hi[0];
    hlo[1] = 0;
    hhi[1] = 0;
    hlo[2] = lo[2];
    hhi[2] = hi[2];
    values = (double*)malloc(ncells*sizeof(double));
    for (i=0; i<ncells; i++) values[i] = 0.0;
    i4 = 1;
    j4 = 5;
    ierr = HYPRE_StructMatrixSetBoxValues(matrix, hlo, hhi, i4, &j4, values);
    free(values);
  }
  if (lo[2] == 1) {
    ncells = (hi[1]-lo[1]+1)*(hi[2]-lo[2]+1);
    hlo[0] = lo[0];
    hhi[0] = hi[0];
    hlo[1] = lo[1];
    hhi[1] = hi[1];
    hlo[2] = 0;
    hhi[2] = 0;
    values = (double*)malloc(ncells*sizeof(double));
    for (i=0; i<ncells; i++) values[i] = 0.0;
    i4 = 1;
    j4 = 6;
    ierr = HYPRE_StructMatrixSetBoxValues(matrix, hlo, hhi, i4, &j4, values);
    free(values);
  }
  ierr = HYPRE_StructMatrixAssemble(matrix);
/**
 *   Create vectors for matrix-vector multiply
 */
  ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &vec_x);
  ierr = HYPRE_StructVectorInitialize(vec_x);
  hlo[0] = lo[0];
  hlo[1] = lo[1];
  hlo[2] = lo[2];
  hhi[0] = hi[0];
  hhi[1] = hi[1];
  hhi[2] = hi[2];
  ierr = HYPRE_StructVectorSetBoxValues(vec_x, hlo, hhi, vector);
  ierr = HYPRE_StructVectorAssemble(vec_x);
  NGA_Distribution(g_a_i,me,blo,bhi);

  if (bhi[1] > ntot-1) {
    bhi[1] = ntot-1;
  }

  btot = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1)*(hi[2]-lo[2]+1);

  for (i=0; i<btot; i++) vector[i] = 0.0;
  hlo[0] = lo[0];
  hlo[1] = lo[1];
  hlo[2] = lo[2];
  hhi[0] = hi[0];
  hhi[1] = hi[1];
  hhi[2] = hi[2];
  ierr = HYPRE_StructVectorGetBoxValues(vec_x, hlo, hhi, vector);

  for (i=0; i<btot; i++) vector[i] = 0.0;
  ierr = HYPRE_StructVectorCreate(MPI_COMM_WORLD, grid, &vec_y);
  ierr = HYPRE_StructVectorInitialize(vec_y);
  ierr = HYPRE_StructVectorSetBoxValues(vec_y, hlo, hhi, vector);
  ierr = HYPRE_StructVectorAssemble(vec_y);
#endif
/**
 *  Multiply sparse matrix. Start by accessing pointers to local portions of
 *  g_a_data, g_a_j, g_a_i
 */
  for (iloop=0; iloop<LOOPNUM; iloop++) {
    t_beg2 = GA_Wtime();
    NGA_Distribution(g_a_data,me,blo,bhi);
    NGA_Access(g_a_data,blo,bhi,&p_data,&ld_a);

    NGA_Distribution(g_a_j,me,blo,bhi);
    NGA_Access(g_a_j,blo,bhi,&p_j,&ld_j);

    NGA_Distribution(g_a_i,me,blo,bhi);
    NGA_Access(g_a_i,blo,bhi,&p_i,&ld_i);

    NGA_Distribution(g_c,me,blo,bhi);
    NGA_Access(g_c,blo,bhi,&p_c,&ld_c);
    for (i = 0; i<bhi[0]-blo[0]+1; i++) {
      p_c[i] = 0.0;
    }
    /**
     * Loop through matrix blocks
     */
    ioff = 0;
    for (iblock = 0; iblock<total_procs; iblock++) {
      iproc = proclist[iblock];
      NGA_Distribution(g_b,iproc,blo,bhi);
      bvec = (double*)malloc((bhi[0]-blo[0]+1)*sizeof(double));
      j = 0;
      t_beg = GA_Wtime();
      NGA_Get(g_b,blo,bhi,bvec,&j);
      t_get = t_get + GA_Wtime() - t_beg;
      t_beg = GA_Wtime();
      irows = nsize[me]-1;
      for (i=0; i<=irows; i++) {
        jmin = p_i[ioff+i];
        jmax = p_i[ioff+i+1]-1;
        tempc = 0.0;
        for (j = jmin; j<=jmax; j++) {
          jj = p_j[j];
          tempc = tempc + p_data[j]*bvec[jj];
        }
        p_c[i] = p_c[i] + tempc;
      }
      ioff = ioff + nsize[me] + 1;
      t_mult = t_mult + GA_Wtime() - t_beg;
      free(bvec);
    }
    t_ga_tot = t_ga_tot + GA_Wtime() - t_beg2;

    NGA_Distribution(g_a_data,me,blo,bhi);
    NGA_Release(g_a_data,blo,bhi);
    NGA_Distribution(g_a_j,me,blo,bhi);
    NGA_Release(g_a_j,blo,bhi);
    NGA_Distribution(g_a_i,me,blo,bhi);
    NGA_Release(g_a_i,blo,bhi);
    NGA_Distribution(g_c,me,blo,bhi);
    NGA_Release(g_c,blo,bhi);

#if USE_HYPRE
    alpha = 1.0;
    beta = 0.0;
    t_beg = GA_Wtime();
    ierr = HYPRE_StructMatrixMatvec(alpha, matrix, vec_x, beta, vec_y);
    t_hypre_strct = GA_Wtime() - t_beg;
    hlo[0] = lo[0];
    hlo[1] = lo[1];
    hlo[2] = lo[2];
    hhi[0] = hi[0];
    hhi[1] = hi[1];
    hhi[2] = hi[2];
    ierr = HYPRE_StructVectorGetBoxValues(vec_y, hlo, hhi, vector);
    NGA_Distribution(g_c,me,lo,hi);
    cbuf = (double*)malloc((hi[0]-lo[0]+1)*sizeof(double));
    NGA_Get(g_c,lo,hi,cbuf,&one);
    prdot = 0.0;
    dotga = 0.0;
    dothypre = 0.0;
    for (i=0; i<(hi[0]-lo[0]+1); i++) {
      dothypre = dothypre + vector[i]*vector[i];
      dotga = dotga + cbuf[i]*cbuf[i];
      prdot = prdot + (vector[i]-cbuf[i])*(vector[i]-cbuf[i]);
    }
    NGA_Dgop(&dotga,1,"+");
    NGA_Dgop(&dothypre,1,"+");
    NGA_Dgop(&prdot,1,"+");
    gatot += sqrt(dotga);
    hypretot += sqrt(dothypre);
    prtot += sqrt(prdot);
    free(cbuf);
#endif
  }
#if USE_HYPRE
  if (me == 0) {
    printf("Magnitude of GA solution:                         %e\n",
        gatot/((double)LOOPNUM));
    printf("Magnitude of HYPRE solution:                      %e\n",
        hyprtot/((double)LOOPNUM));
    printf("Difference between GA and HYPRE (Struct) results: %e\n",
        prtot/((double)LOOPNUM));
  }
#endif

  free(vector);
/**
 *  Clean up arrays
 */
  NGA_Destroy(g_b);
  NGA_Destroy(g_c);
  NGA_Destroy(g_a_data);
  NGA_Destroy(g_a_i);
  NGA_Destroy(g_a_j);
#if USE_HYPRE
#if USE_STRUCT
  ierr = HYPRE_StructStencilDestroy(stencil);
  ierr = HYPRE_StructGridDestroy(grid);
  ierr = HYPRE_StructMatrixDestroy(matrix);
  ierr = HYPRE_StructVectorDestroy(vec_x);
  ierr = HYPRE_StructVectorDestroy(vec_y);
#endif
#endif

  NGA_Dgop(&t_cnstrct,1,"+");
  NGA_Dgop(&t_get,1,"+");
  NGA_Dgop(&t_mult,1,"+");
  NGA_Dgop(&t_ga_tot,1,"+");
#if USE_HYPRE
  NGA_Dgop(&t_hypre_strct,1,"+");
#endif
  free(proclist);
  free(proc_inv);
  free(icnt);
  free(voffset);
  free(nsize);
  free(offset);
  free(mapc);

  if (me == 0) {
    printf("Time to create sparse matrix:                         %12.4f\n",
      t_cnstrct/((double)nprocs));
    printf("Time to get right hand side vector:                   %12.4f\n",
      t_get/((double)nprocs));
    printf("Time for sparse matrix block multiplication:          %12.4f\n",
      t_mult/((double)nprocs));
    printf("Time for total sparse matrix multiplication:          %12.4f\n",
      t_ga_tot/((double)nprocs));
#if USE_HYPRE
#if USE_STRUCT
    printf("Total time for HYPRE (Struct)  matrix-vector multiply:%12.4f\n",
      t_hypre_strct/((double)nprocs));
#endif
#endif
  }
  NGA_Terminate();
/**
 *  Tidy up after message-passing library
 */
  MP_FINALIZE();
}
