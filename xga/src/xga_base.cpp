/* XGA private implementation */
#include "xga_private.hpp"

namespace XGA {

/**
 * Constructor
 * @param[in] group home group for global array
 * @param[in] t_ndim dimension of global array
 * @param[in] t_dims dimensions of global array
 * @param[in] type data type of array
 */
p_GA::p_GA(Group * group, int t_ndim, int64_t *t_dims, xga_types type)
{
  p_datatype = type;
  p_group = group;
  p_ndim = t_ndim;
  int i;

  p_env = Environment::instance();
  p_alloc = NULL;

  if (t_ndim < 1 || t_ndim > MAXDIM) {
    p_env->error("Unsupported number of dimensions",t_ndim);
  }
  for (i=0; i<t_ndim; i++) {
    if (t_dims[i] < 1)
      p_env->error("Illegal dimension specified",t_dims[i]);
  }

  for (i=0; i<t_ndim; i++) p_dims[i] = t_dims[i];
  for (i=0; i<t_ndim; i++) width[i] = 0;
  for (i=0; i<t_ndim; i++) chunk[i] = 0;
  p_mapc = NULL;

  p_distr = REGULAR;
  if (type == XGA_INT) {
    p_elemsize = sizeof(int);
  } else if (type == XGA_LONG) {
    p_elemsize = sizeof(long);
  } else if (type == XGA_LONGLONG) {
    p_elemsize = sizeof(long long);
  } else if (type == XGA_FLOAT) {
    p_elemsize = sizeof(float);
  } else if (type == XGA_DOUBLE) {
    p_elemsize = sizeof(double);
  } else if (type == XGA_COMPLEX) {
    p_elemsize = sizeof(std::complex<float>);
  } else if (type == XGA_DCOMPLEX) {
    p_elemsize = sizeof(std::complex<double>);
  } else if (type == XGA_UNKNOWN) {
    p_elemsize = -1;
  } else {
    p_env->error("Unknown data type requested",type);
  }
  block_total = -1;
  p_active = false;
}

/**
 * Basic destructor
 */
p_GA::~p_GA()
{
  if (p_mapc) delete [] p_mapc;
  p_alloc->free();
  delete p_alloc;
  delete [] ptr;
}

/**
 * Set data distribution type
 * @param[in] distr data distribution type
 */
void p_GA::setDataDistribution(XGA::data_distribution distr)
{
  if (p_active) {
    p_env->error("(setDataDistribution) array has already been allocated",-1);
  }
  p_distr = distr;
}

/**
 * Set block sizes and processor grid for ScaLAPACK-style data
 * distribution. This is only strictly an ScaLAPACK distribution in
 * 2 dimensions but the generalization to higher dimensions is
 * straightforward
 * @param[in] dims dimensions of individual blocks
 * @param[in] prod_grid dimension of processor grid
 */
void p_GA::setBlockLayout(int64_t *dims, int *proc_grid)
{
  if (p_active) {
    p_env->error("(setBlockLayout) array has already been allocated",-1);
  }
  int i;
  for (i=0; i<p_ndim; i++) {
    if (dims[i] < 0 || dims[i] > p_dims[i]) {
      p_env->error("(setBlockLayout) illegal block dimension",dims[i]);
    }
  }
  int ntot = 1;
  for (i=0; i<p_ndim; i++) ntot *= proc_grid[i];
  if (ntot != p_group->size()) {
      p_env->error("(setBlockLayout) proc_grid incompatible"
          " with group size",ntot);
  }
  block_total = 1;
  for (i=0; i<p_ndim; i++) {
    p_proc_grid[i] = proc_grid[i];
    blk_dims[i] = dims[i];
    int jsize = p_dims[i]/dims[i];
    if (p_dims[i]%dims[i] != 0) jsize++;
    num_blks[i] = jsize;
    block_total *= num_blks[i];
  }
  p_distr = SCALAPACK;
}

/**
 * @param[in] mapc array containing partitions along each axis
 * @param[in] nblock array containing processor decomposition
 */
void p_GA::setIrregularDistribution(int64_t *mapc, int *nblock)
{
  int rank = p_group->rank();
  if (p_active) {
    p_env->error("(setIrregularDistribution) array has already been allocated",-1);
  }
  int i, j;
  int ncnt = 1;
  for (i=0; i<p_ndim; i++) ncnt *= nblock[i];
  if (ncnt != nproc) {
    p_env->error("(setIrregularDistribution) number of processors in"
        "nblock does not match size of group",ncnt);
  }
  ncnt = 0;
  for (i=0; i<p_ndim; i++) ncnt += nblock[i];
  p_mapc = new int64_t[ncnt];
  for (i=0; i<ncnt; i++) p_mapc[i] = mapc[i];
  for (i=0; i<p_ndim; i++) p_proc_grid[i] = nblock[i];
}

/**
 * Allocate resources to create global array
 */
void p_GA::allocate()
{
  if (!p_env) {
    p_env->error("(allocate) environment has not been initialized",-1);
  }
  if (p_ndim < 1) {
    p_env->error("(allocate) insufficient data to create global array",-1);
  }
  if (p_elemsize < 1) {
    p_env->error("(allocate) data element size has not be specified",-1);
  }
  int64_t block_size;
  int64_t mem_size;
  int i;
  if (p_mapc == NULL && p_distr == REGULAR) {
    int64_t blk[MAXDIM];
    int pe[MAXDIM];
    int d;
    if (chunk[0] != 0) {
      for (d=0; d<p_ndim; d++) 
        blk[d] = chunk[d] <= p_dims[d] ? chunk[d] : p_dims[d];
    } else {
      for (d=0; d<p_ndim; d++) blk[d] = -1;
    }
    /* eliminate dimensions = 1 from ddb analysis */
    for (d=0; d<p_ndim; d++) if (p_dims[d] == 1) blk[d] = 1;
    /* data is normally distributed on processors */
    ddb_h2(p_ndim, p_dims, p_group->size(), 0.0, 0, blk, pe);
    int64_t *mapAll = new int64_t[p_group->size()+MAXDIM-1];
    int64_t *map = mapAll;
    for (d=0; d<p_ndim; d++) {
      int64_t p, nblocks, pcut;
      /* RJH ... don't leave some processors without data if possible,
       * but respect the users block size */
      if (chunk[d] > 1) {
        int64_t dnom = chunk[d] <= p_dims[d] ? chunk[d] : p_dims[d];
        int64_t ddim = ((p_dims[d]-1)/dnom+1);
        pcut = (ddim-(blk[d]-1)*pe[d]);
      } else {
        pcut = (p_dims[d]-(blk[d]-1)*pe[d]);
      }
      for (nblocks=i=p=0; (p<pe[d]) && (i<p_dims[d]); p++, nblocks++) {
        int64_t b = blk[d];
        if (p >= pcut) b = b-1;
// bjp        map[nblocks] = i+1;
        map[nblocks] = i;
        if (chunk[d]>1) b *= chunk[d] <= p_dims[d] ? chunk[d] : p_dims[d];
        i += b;
      }
      pe[d] = pe[d] <= nblocks ? pe[d] : nblocks;
      map += pe[d];
    }
    int64_t maplen = 0;
    for (i=0; i<p_ndim; i++) {
      p_proc_grid[i] = pe[i];
      maplen += pe[i];
    }
    p_mapc = new int64_t[maplen+1];
    for (i=0; i<maplen; i++) {
      p_mapc[i] = mapAll[i];
    }
    p_mapc[maplen] = -1;
    delete [] mapAll;
  } else if (p_distr == SCALAPACK) {
    /* ScaLAPACK block-cyclic data distribution has been specified. Figure
     * out how much memory is needed by each processor to store blocks */
    int64_t i, j, jtot, skip, imin, imax;
    int64_t index[MAXDIM];
    XGA_FIND_PROC_INDICES_M(p_group->rank(),index);
    block_size = 1;
    for (i=0; i<p_ndim; i++) {
      skip = p_proc_grid[i];
      jtot = 0;
      for (j=index[i]; j<num_blks[i]; j += skip) {
        imin = j*blk_dims[i];
        imax = (j+1)*blk_dims[i]-1;
        if (imax >= p_dims[i]) imax = p_dims[i]-1;
        jtot += (imax-imin+1);
      }
      blk_size[i] = jtot;
      block_size *= jtot;
    }
  } else if (p_distr == TILED) {
    /* Tiled data distribution has been specified. Figure
     * out how much memory is needed by each processor to store blocks */
    int64_t j, jtot, skip, imin, imax;
    int64_t index[MAXDIM];
    XGA_FIND_TILE_PROC_INDICES_M(p_group->rank(),index);
    block_size = 1;
    for (i=0; i<p_ndim; i++) {
      skip = p_proc_grid[i];
      jtot = 0;
      for (j=index[i]; j<blk_num[i]; j += skip) {
        imin = j*blk_size[i] + 1;
        imax = (j+1)*blk_size[i];
        if (imax >= p_dims[i]) imax = p_dims[i]-1;
        jtot += (imax-imin+1);
      }
      block_size *= jtot;
    }
  } else if (p_distr == TILED_IRREG) {
    /* Tiled data distribution has been specified. Figure
     * out how much memory is needed by each processor to store blocks */
    int64_t j, jtot, skip, imin, imax;
    int64_t index[MAXDIM];
    int64_t offset = 0;
    XGA_FIND_TILE_PROC_INDICES_M(p_group->rank(),index);
    block_size = 1;
    for (i=0; i<p_ndim; i++) {
      skip = p_proc_grid[i];
      jtot = 0;
      for (j=index[i]; j<blk_num[i]; j += skip) {
        imin = p_mapc[offset+j];
        if (j<blk_num[i]-1) {
          imax = p_mapc[offset+j+1]-1;
        } else {
          imax = p_dims[i]-1;
        }
        jtot += (imax-imin+1);
      }
      block_size *= jtot;
      offset += blk_num[i];
    }
  }

  p_active = true;

  /* Set remaining parameters and determine memory size if regular data
   * distribution is being used */
  if (p_distr == REGULAR) {
    int64_t hi[MAXDIM];
    for (i=0; i<p_ndim; i++) {
      p_scale[i] = static_cast<double>(p_proc_grid[i])
        /static_cast<double>(p_dims[i]);
    }
    distribution(p_group->rank(),p_lo,hi);
    int64_t nelem = 1;
    for (i=0; i<p_ndim; i++) {
      nelem *= (hi[i]-p_lo[i]+1);
    }
    mem_size = nelem*p_elemsize;
  } else {
    mem_size = block_size*p_elemsize;
  }
  p_size = mem_size;

  /* create distributed allocation */
  p_alloc = new CMX::Allocation;
  if (!p_alloc->malloc(p_size, p_group->getCMXGroup())) {
    p_env->error("(allocate) unable to allocate memory",p_size);
  }
  std::vector<void*> pvec;
  p_alloc->access(pvec);
  nproc = p_group->size();
  ptr = new void*[nproc];
  for (i=0; i<nproc; i++) {
    ptr[i] = pvec[i];
  }
}

/**
 * Duplicate a global array. New array has same datatype, size and
 * data partition but individual values are not initialized.
 * @return pointer to new global array
 */
p_GA* p_GA::duplicate()
{
  p_GA *g_new = new p_GA(p_group, p_ndim, p_dims, p_datatype);
  g_new->setDataDistribution(p_distr);

  if (p_distr == REGULAR) {
    g_new->setIrregularDistribution(p_mapc, p_proc_grid);
  } else if (p_distr == SCALAPACK) {
    g_new->setBlockLayout(blk_dims, p_proc_grid);
  }
  g_new->allocate();
  return g_new;
}

/**
 * Copy contents of array B into calling array. Arrays must be same size and
 * datatype
 * @param g_b source array
 */
void p_GA::copy(p_GA *g_b)
{
  int  ndim, ndimb, type, typeb;
  int64_t dimsb[MAXDIM],i;
  char *ptr_a;
  int64_t _dims[MAXDIM];
  int64_t _ld[MAXDIM-1];
  int64_t _lo[MAXDIM];
  int64_t _hi[MAXDIM];

  if(this == g_b) p_env->error("arrays have to be different ", 0);
  if (!g_b->p_active) p_env->error("new array must be allocated",0);

  type = this->p_datatype;
  typeb = g_b->p_datatype;
  if(type != typeb) p_env->error("types not the same", 0);
  ndim = this->p_ndim;
  ndimb = g_b->p_ndim;
  if(ndim != ndimb) p_env->error("dimensions not the same", ndimb);
  for (i=0; i<ndim; i++) _dims[i] = this->p_dims[i];
  for (i=0; i<ndimb; i++) dimsb[i] = g_b->p_dims[i];

  for(i=0; i< ndim; i++) if(_dims[i]!=dimsb[i]) 
    p_env->error("dimension sizes not the same",i);

  this->localInit();
  while (this->nextLocalBlock(_lo,_hi,&ptr_a,_ld)) {
    g_b->put(_lo, _hi, ptr_a, _ld);
  }
}

/* test two patches to see if they have the same shape */
bool test_shape(int64_t *alo, int64_t *ahi, int64_t *blo,
    int64_t *bhi, int64_t andim, int64_t bndim)
{
  int64_t i;

  if(andim != bndim) return false;

  for(i=0; i<andim; i++)
    if((ahi[i] - alo[i]) != (bhi[i] - blo[i])) return false;

  return true;
}

/**
 * Copy a patch of array B to a patch in the calling array. Array must be the
 * same datatype.
 * @param trans flag signifying whether to transpose data when copying
 * @param alo, ahi bounding indices of target patch
 * @param g_b source array
 * @param blo, bhi bounding indices of source patch
 */
void p_GA::copyPatch(char trans, int64_t *alo, int64_t *ahi,
                    p_GA *g_b, int64_t *blo, int64_t *bhi)
{
  int64_t i, j;
  int64_t idx, factor;
  int64_t atype, btype, andim, adims[MAXDIM], bndim, bdims[MAXDIM];
  int64_t nelem;
  int64_t atotal, btotal;
  int64_t los[MAXDIM], his[MAXDIM];
  int64_t lod[MAXDIM], hid[MAXDIM];
  int64_t ld[MAXDIM], ald[MAXDIM], bld[MAXDIM];
  void *src_data_ptr, *tmp_ptr;
  int64_t *src_idx_ptr, *dst_idx_ptr;
  int64_t bvalue[MAXDIM], bunit[MAXDIM];
  int64_t factor_idx1[MAXDIM], factor_idx2[MAXDIM], factor_data[MAXDIM];
  int64_t base;
  int64_t me_a, me_b;
  Group *a_grp, *b_grp;
  int anproc, bnproc;
  int64_t num_blocks_a, num_blocks_b;
  bool use_put, has_intersection;
  
  if (anproc <= bnproc) {
    use_put = true;
  } else {
    use_put = false;
  }

  a_grp = this->p_group;
  b_grp = g_b->p_group;
  me_a = a_grp->rank();
  me_b = b_grp->rank();
  anproc = a_grp->size();
  bnproc = b_grp->size();

  atype = this->p_datatype;
  btype = g_b->p_datatype;
  andim = this->p_ndim;
  bndim = g_b->p_ndim;
  for (i=0; i<andim; i++) adims[i] = this->p_dims[i];
  for (i=0; i<bndim; i++) bdims[i] = g_b->p_dims[i];

  if(this == g_b) {
    /* they are the same patch */
    if(comp_patch(andim, alo, ahi, bndim, blo, bhi)) {
        return;
    /* they are in the same GA, but not the same patch */
    } else if (patch_intersect(alo, ahi, blo, bhi, andim)) {
      p_env->error("array patches cannot overlap when copying to self ", 0);
    }
  }

  if(atype != btype ) p_env->error("array datatype mismatch ", 0);

  /* check if patch indices and dims match */
  for(i=0; i<andim; i++)
    if(alo[i] <= 0 || ahi[i] > adims[i])
      p_env->error("g_a indices out of range ", 0);
  for(i=0; i<bndim; i++)
    if(blo[i] <= 0 || bhi[i] > bdims[i])
      p_env->error("g_b indices out of range ", 0);

  /* check if numbers of elements in two patches match each other */
  atotal = 1; btotal = 1;
  for(i=0; i<andim; i++) atotal *= (ahi[i] - alo[i] + 1);
  for(i=0; i<bndim; i++) btotal *= (bhi[i] - blo[i] + 1);
  if(atotal != btotal)
    p_env->error("capacities two of patches do not match ", 0);

  /* additional restrictions that apply if one or both arrays use
     block-cyclic data distributions */
  num_blocks_a = this->block_total;
  num_blocks_b = g_b->block_total;
  if (num_blocks_a >= 0 || num_blocks_b >= 0) {
    if (!(trans == 'n' || trans == 'N')) {
      p_env->error("Transpose option not supported for block-cyclic data", 0);
    }
    for(i=0; i<andim; i++)
      if((ahi[i] - alo[i]) != (bhi[i] - blo[i]))
        p_env->error("Change in shape not supported for block-cyclic data", 0);
  }

  if (num_blocks_a < 0 && num_blocks_b <0) {
    /* now find out cordinates of a patch that I own */
    if (use_put) {
      this->distribution(me_a, los, his);
    } else {
      g_b->distribution(me_b, los, his);
    }

    /* copy my share of data */
    if (use_put) {
      has_intersection = patch_intersect(alo, ahi, los, his, bndim);
    } else {
      has_intersection = patch_intersect(blo, bhi, los, his, bndim);
    }
    if(has_intersection){
      if (use_put) {
        this->accessPtr(los, his, &src_data_ptr, ld); 
      } else {
        g_b->accessPtr(los, his, &src_data_ptr, ld); 
      }

      /* calculate the number of elements in the patch that I own */
      nelem = 1; for(i=0; i<andim; i++) nelem *= (his[i] - los[i] + 1);

      for(i=0; i<andim; i++) ald[i] = ahi[i] - alo[i] + 1;
      for(i=0; i<bndim; i++) bld[i] = bhi[i] - blo[i] + 1;

      base = 0; factor = 1;
      for(i=andim-1; i>=0; i--) {
        base += los[i] * factor;
        if (i>0) factor *= ld[i-1];
      }

      /*** straight copy possible if there's no reshaping or transpose ***/
      if((trans == 'n' || trans == 'N') &&
          test_shape(alo, ahi, blo, bhi, andim, bndim)) { 
        /* find source[lo:hi] --> destination[lo:hi] */
        if (use_put) {
          dest_indices(andim, los, alo, ald, bndim, lod, blo, bld);
          dest_indices(andim, his, alo, ald, bndim, hid, blo, bld);
          g_b->put(lod, hid, src_data_ptr, ld);
          this->releasePtr(los, his);
        } else {
          dest_indices(bndim, los, blo, bld, andim, lod, alo, ald);
          dest_indices(bndim, his, blo, bld, andim, hid, alo, ald);
          this->get(lod, hid, src_data_ptr, ld);
          g_b->releasePtr(los, his);
        }
        /*** due to generality of this transformation scatter is required ***/
      } else{
        tmp_ptr = xga_malloc(nelem, atype);
        src_idx_ptr = reinterpret_cast<int64_t*>(xga_malloc((bndim*nelem),
              XGA_LONG));
        dst_idx_ptr = reinterpret_cast<int64_t*>(xga_malloc((bndim*nelem),
              XGA_LONG));
        /* calculate the destination indices */

        /* given los and his, find indices for each elements
         * bvalue: starting index in each dimension
         * bunit: stride in each dimension
         */
        for (i=andim-1; i>=0; i--) {
          bvalue[i] = los[i];
          if (i == andim-1) bunit[i] = 1;
          else bunit[i] = bunit[i+1] * (his[i+1] - los[i+1] + 1);
        }

        if (use_put) {
          /* source indices */
          for (i=0; i<nelem; i++) {
            for (j=andim-1; j>=0; j--){
              src_idx_ptr[i*andim+j] = bvalue[j];
              /* if the next element is the first element in
               * one dimension, increment the index by 1
               */
              if (((i+1) % bunit[j]) == 0) bvalue[j]++;
              /* if the index becomes larger than the upper
               * bound in one dimension, reset it.
               */
              if(bvalue[j] > his[j]) bvalue[j] = los[j];
            }
          }

          /* index factor: reshaping without transpose */
          factor_idx1[andim-1] = 1;
          for (j=andim-2; j>=0; j--)
            factor_idx1[j] = factor_idx1[j+1] * ald[j+1];

          /* index factor: reshaping with transpose */
          factor_idx2[0] = 1;
          for (j=1; j<andim; j++)
            factor_idx2[j] = factor_idx2[j-1] * ald[j-1];

          /* data factor */
          factor_data[andim-1] = 1;
          for (j=bndim-2; j>=0; j--)
            factor_data[j] = factor_data[j+1] * ld[j];

          /* destination indices */
          for(i=0; i<nelem; i++) {
            /* linearize the n-dimensional indices to one dimension */
            idx = 0;
            if (trans == 'n' || trans == 'N')
              for (j=0; j<andim; j++)
                idx += (src_idx_ptr[i*andim+j] - alo[j]) *
                  factor_idx1[j];
            else
              /* if the patch needs to be transposed, reverse
               * the indices: (i, j, ...) -> (..., j, i)
               */
              for (j=(andim-1); j>=0; j--)
                idx += (src_idx_ptr[i*bndim+j] - alo[j]) *
                  factor_idx2[j];

            /* convert the one dimensional index to n-dimensional
             * indices of destination
             */
            for (j=bndim-1; j>=0; j--) {
              dst_idx_ptr[i*bndim+j] = idx % bld[j] + blo[j];
              idx /= bld[j];
            }

            /* move the data block to create a new block */
            /* linearize the data indices */
            idx = 0;
            for (j=0; j<andim; j++)
              idx += (src_idx_ptr[i*andim+j]) * factor_data[j];

            /* adjust the position
             * base: starting address of the first element */
            idx -= base;
            /* move the element to the temporary location */
#define ASSIGN_M(_type, _dst, _src, _i, _idx)                         \
            reinterpret_cast<_type*>(_dst)[i]                         \
            = reinterpret_cast<_type*>(_src)[_idx]

            switch(atype) {
              case XGA_DOUBLE:
                ASSIGN_M(double, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_INT:
                ASSIGN_M(int, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_DCOMPLEX:
                ASSIGN_M(std::complex<double>, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_COMPLEX:
                ASSIGN_M(std::complex<float>, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_FLOAT:
                ASSIGN_M(float, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_LONG:
                ASSIGN_M(long, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_LONGLONG:
                ASSIGN_M(long long, tmp_ptr, src_data_ptr, i, idx);
            }
#undef ASSIGN_M
          }
          g_b->scatter(tmp_ptr, dst_idx_ptr, 0, nelem);
          this->releasePtr(los, his);
        } else {
          /* destination indices */
          for (i=0; i<nelem; i++) {
            for (j=bndim-1; j>=0; j--){
              src_idx_ptr[i*bndim+j] = bvalue[j];
              /* if the next element is the first element in
               * one dimension, increment the index by 1
               */
              if (((i+1) % bunit[j]) == 0) bvalue[j]++;
              /* if the index becomes larger than the upper
               * bound in one dimension, reset it.
               */
              if(bvalue[j] > his[j]) bvalue[j] = los[j];
            }
          }

          /* index factor: reshaping without transpose */
          factor_idx1[bndim-1] = 1;
          for (j=bndim-2; j>=0; j--) 
            factor_idx1[j] = factor_idx1[j+1] * bld[j+1];

          /* index factor: reshaping with transpose */
          factor_idx2[0] = 1;
          for (j=1; j<bndim; j++)
            factor_idx2[j] = factor_idx2[j-1] * bld[j-1];

          /* data factor */
          factor_data[bndim-1] = 1;
          for (j=bndim-2; j>=0; j--) 
            factor_data[j] = factor_data[j+1] * ld[j];

          /* destination indices */
          for(i=0; i<nelem; i++) {
            /* linearize the n-dimensional indices to one dimension */
            idx = 0;
            if (trans == 'n' || trans == 'N')
              for (j=0; j<andim; j++) 
                idx += (src_idx_ptr[i*bndim+j] - blo[j]) *
                  factor_idx1[j];
            else
              /* if the patch needs to be transposed, reverse
               * the indices: (i, j, ...) -> (..., j, i)
               */
              for (j=(andim-1); j>=0; j--) 
                idx += (src_idx_ptr[i*bndim+j] - blo[j]) *
                  factor_idx2[j];

            /* convert the one dimensional index to n-dimensional
             * indices of destination
             */
            for (j=andim-1; j>=0; j--) {
              dst_idx_ptr[i*bndim+j] = idx % ald[j] + alo[j]; 
              idx /= ald[j];
            }

            /* move the data block to create a new block */
            /* linearize the data indices */
            idx = 0;
            for (j=0; j<bndim; j++) 
              idx += (src_idx_ptr[i*bndim+j]) * factor_data[j];

            /* adjust the position
             * base: starting address of the first element */
            idx -= base;

            /* move the element to the temporary location */
#define ASSIGN_M(_type, _dst, _src, _i, _idx)                         \
            reinterpret_cast<_type*>(_dst)[i]                         \
            = reinterpret_cast<_type*>(_src)[_idx]
            switch(atype) {
              case XGA_DOUBLE:
                ASSIGN_M(double, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_INT:
                ASSIGN_M(int, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_DCOMPLEX:
                ASSIGN_M(std::complex<double>, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_COMPLEX:
                ASSIGN_M(std::complex<float>, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_FLOAT:
                ASSIGN_M(float, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_LONG:
                ASSIGN_M(long, tmp_ptr, src_data_ptr, i, idx);
                break;
              case XGA_LONGLONG:
                ASSIGN_M(long long, tmp_ptr, src_data_ptr, i, idx);
            }
#undef ASSIGN_M
          }
          this->gather(tmp_ptr, dst_idx_ptr, 0, nelem);
          g_b->releasePtr(los, his);
        }
        delete [] dst_idx_ptr;
        delete [] src_idx_ptr;
        xga_free(tmp_ptr, atype);
      }
    }
  } else {
#define SET_PTR_M(_type, _src, _offset)             \
    _src = reinterpret_cast<char*>                  \
    (reinterpret_cast<_type*>(_src)+_offset)
    int64_t offset, last, jtot;
    for (i=0; i<andim; i++) {
      ald[i] = ahi[i] - alo[i] + 1;
    }
    for (i=0; i<bndim; i++) {
      bld[i] = bhi[i] - blo[i] + 1;
    }
    if (use_put) {
      /* Array a is block-cyclic distributed */
      if (num_blocks_a >= 0) {
        char *block_ptr;
        this->localInit();
        while (this->nextLocalBlock(los, his, &block_ptr, ld)) {
            /* Copy limits since patch intersect modifies los array */
            for (j=0; j < andim; j++) {
              lod[j] = los[j];
              hid[j] = his[j];
            }
            if (patch_intersect(alo,ahi,los,his,andim)) {
              offset = 0;
              last = andim - 1;
              jtot = 1;
              for (j=last; j>0; j--) {
                offset += (los[j]-lod[j])*jtot;
                jtot *= ld[j-1];
              }
              offset += (los[0]-lod[0])*jtot;
              switch(atype) {
                case XGA_DOUBLE:
                  SET_PTR_M(double,block_ptr,offset);
                  break;
                case XGA_INT:
                  SET_PTR_M(int,block_ptr,offset);
                  break;
                case XGA_DCOMPLEX:
                  SET_PTR_M(std::complex<double>,block_ptr,offset);
                  break;
                case XGA_COMPLEX:
                  SET_PTR_M(std::complex<float>,block_ptr,offset);
                  break;
                case XGA_FLOAT:
                  SET_PTR_M(float,block_ptr,offset);
                  break;     
                case XGA_LONG:
                  SET_PTR_M(long,block_ptr,offset);
                  break;
                case XGA_LONGLONG:
                  SET_PTR_M(long long,block_ptr,offset);
                  break;
                default:
                  break;
              }
              dest_indices(andim, los, alo, ald, bndim, lod, blo, bld);
              dest_indices(andim, his, alo, ald, bndim, hid, blo, bld);
              g_b->put(lod, hid, block_ptr, ld);
            //  g_a->releaseBlock(i);
            }
          }
      } else {
        /* Only array b is block-cyclic distributed */
        this->distribution(me_a, los, his); 
        if (patch_intersect(alo,ahi,los,his,andim)) {
          this->accessPtr(los, his, &src_data_ptr, ld); 
          dest_indices(andim, los, alo, ald, bndim, lod, blo, bld);
          dest_indices(andim, his, alo, ald, bndim, hid, blo, bld);
          g_b->put(lod, hid, src_data_ptr, ld);
          this->releasePtr(los, his);
        }
      }
    } else {
      /* Array b is block-cyclic distributed */
      if (num_blocks_b >= 0) {
        char *block_ptr;
        g_b->localInit();
        while (g_b->nextLocalBlock(los, his, &block_ptr, ld)) {
            /* Copy limits since patch intersect modifies los array */
            for (j=0; j < andim; j++) {
              lod[j] = los[j];
              hid[j] = his[j];
            }
            if (patch_intersect(blo,bhi,los,his,andim)) {
              offset = 0;
              last = andim - 1;
              jtot = 1;
              offset = (los[last]-lod[last])*jtot;
              for (j=last-1; j>=0; j++) {
                jtot *= ld[j];
                offset += (los[j]-lod[j])*jtot;
              }
              switch(atype) {
                case XGA_DOUBLE:
                  SET_PTR_M(double,block_ptr,offset);
                  break;
                case XGA_INT:
                  SET_PTR_M(int,block_ptr,offset);
                  break;
                case XGA_DCOMPLEX:
                  SET_PTR_M(std::complex<double>,block_ptr,offset);
                  break;
                case XGA_COMPLEX:
                  SET_PTR_M(std::complex<float>,block_ptr,offset);
                  break;
                case XGA_FLOAT:
                  SET_PTR_M(float,block_ptr,offset);
                  break;     
                case XGA_LONG:
                  SET_PTR_M(long,block_ptr,offset);
                  break;
                case XGA_LONGLONG:
                  SET_PTR_M(long long,block_ptr,offset);
                  break;
                default:
                  break;
              }
              dest_indices(bndim, los, blo, bld, andim, lod, alo, ald);
              dest_indices(bndim, his, blo, bld, andim, hid, alo, ald);
              this->get(lod, hid, block_ptr, ld);
              //g_b->releaseBlock(i);
            }
          }
      } else {
        /* Array a is block-cyclic distributed */
        g_b->distribution(me_b, los, his); 
        if (patch_intersect(blo,bhi,los,his,bndim)) {
          g_b->accessPtr(los, his, &src_data_ptr, ld); 
          dest_indices(bndim, los, blo, bld, andim, lod, alo, ald);
          dest_indices(bndim, his, blo, bld, andim, hid, alo, ald);
          this->get(lod, hid, src_data_ptr, ld);
          g_b->releasePtr(los, his);
        }
      }
    }
#undef SET_PTR_M
  }
}

/**
 * Find block owned by processor proc
 * @param[in] proc processor being queried
 * @param[out] lo,hi lower and upper bounding indices of block
 *             owned by processor proc
 */
void p_GA::distribution(const int proc, int64_t *lo, int64_t *hi)
{
  int lproc = proc;
  XGA_OWNS_M(lproc, lo, hi); 
}

/**
 * Initialize n-dimensional loop by counting elements and
 * setting subscript=lo
 */
#define XGA_INITLOOP_M(_elems, _ndim, _subscript, _lo, _hi, _dims)  \
{                                                                   \
  int64_t  _i;                                                      \
  *_elems = 1;                                                      \
  for(_i=0; _i<_ndim; _i++){                                        \
    *_elems *= _hi[_i]-_lo[_i] + 1;                                 \
    _subscript[_i] = _lo[_i];                                       \
  }                                                                 \
}

/* This macro computes index (place in ordered set) for the element
 *  identified by _subscript in ndim-dimensional array of dimensions _dim[]
 *  assume that first subscript component changes first
 */
#define XGA_COMPUTEINDEX_M(_index, _ndim, _subscript, _dims)        \
{                                                                   \
    int64_t  _i, _factor=1;                                         \
    for(_i=0,*(_index)=0; _i<_ndim; _i++){                          \
      *(_index) += _subscript[_i]*_factor;                          \
      if(_i<_ndim-1)_factor *= _dims[_i];                           \
    }                                                               \
}

/* updates subscript corresponding to next element in a patch <lo[]:hi[]>
 */
#define XGA_UPDATESUBSCRIPT_M(_ndim, _subscript, _lo, _hi, _dims)   \
{                                                                   \
    int64_t  _i;                                                    \
  for(_i=0; _i<_ndim; _i++){                                        \
    if(_subscript[_i] < _hi[_i]) { _subscript[_i]++; break;}        \
    _subscript[_i] = _lo[_i];                                       \
  }                                                                 \
}

/**
 * @param[in] lo,hi lower and upper indices of patch in global array
 * @param[out] map list of lower and upper indices for portion of
 *             patch the exists on each processor containing a portion
 *             of the patch. The map is constructed so that for a D
 *             dimensional global array, the first D elements are the
 *             lower indices on the first processor in proclist, the
 *             next D elements are the upper indices of the first
 *             processor in proclist, the next D elements are the
 *             lower indices for the second processor in proclist and
 *             so on.
 * @param[out] proclist list of processors containing some portion of
 *             patch
 * @param[out] np total number of processors containing some portion
 *             of the patch
 * @return false if bounds of patch are invalid
 *
 * For a block cyclic data distribution, this function returns a list
 * of blocks that cover the region, along with the lower and upper
 * indices of each block.
 */
bool p_GA::locateRegion(const int64_t *lo, const int64_t *hi,
    std::vector<int64_t> &map, std::vector<int> &proclist, int *np)
{
  int d, dpos;
  int64_t i, nelems;
  for (d = 0; d < p_ndim; d++) {
    if ((lo[d] < 0 || hi[d] >= p_dims[d]) || lo[d] > hi[d]) return false;
  }
  
  if (p_distr == REGULAR) {
    /* find "processor coordinates" for the lower corner and store them
     * in ProcT */
    int procT[MAXDIM], procB[MAXDIM], proc_subscript[MAXDIM];
    for (d=0, dpos=0; d<p_ndim; d++) {
      XGA_FINDBLOCK_M(p_mapc+dpos,p_proc_grid[d], p_scale[d], lo[d], &procT[d]);
      dpos += p_proc_grid[d];
    }
    /* find "processor coordinates" for the upper corner and store them
     * in procB */
    for (d=0, dpos=0; d<p_ndim; d++) {
      XGA_FINDBLOCK_M(p_mapc+dpos,p_proc_grid[d], p_scale[d], hi[d], &procB[d]);
      dpos += p_proc_grid[d];
    }

    *np = 0;

    /* Find total number of processors containing data and return the
     * result in nelems. Also find the lowest "processor coordinates" of the
     * processor block containing data and return these in proc_subscript.
     */
    XGA_INITLOOP_M(&nelems, p_ndim, proc_subscript, procT, procB,
        p_proc_grid);
    proclist.resize(nelems);
    for (i=0; i<nelems; i++) {
      int64_t _lo[MAXDIM], _hi[MAXDIM];
      int _offset, proc;
      /* convert i to owner processor id using the current values in
         proc_subscript */
      XGA_COMPUTEINDEX_M(&proc, p_ndim, proc_subscript, p_proc_grid);
      /* get range of global array indices that are owned by owner */
      XGA_OWNS_M(proc, _lo, _hi);

      _offset = *np *(p_ndim*2); /* location in map to put patch range */

      for(d = 0; d<p_ndim; d++) {
//        map[d + _offset ] = lo[d] < _lo[d] ? _lo[d] : lo[d];
        map.push_back(lo[d] < _lo[d] ? _lo[d] : lo[d]);
      }
      for(d = 0; d<p_ndim; d++) {
//        map[p_ndim + d + _offset ] = hi[d] > _hi[d] ? _hi[d] : hi[d];
        map.push_back(hi[d] > _hi[d] ? _hi[d] : hi[d]);
      }
      proclist[i] = proc;
      /* Update to proc_subscript so that it corresponds to the next
       * processor in the block of processors containing the patch */
      XGA_UPDATESUBSCRIPT_M(p_ndim,proc_subscript,procT,procB,p_proc_grid);
      (*np)++;
    }
  } else {
    int64_t j, tlo[MAXDIM], thi[MAXDIM], cnt;
    int offset;
    bool chk;
    cnt = 0;
    for (i=0; i<block_total; i++) {
      /* check to see if this block overlaps with requested block
       * defined by lo and hi */
      chk = true;
      /* get limits on block i */
      distribution(i,tlo,thi);
      for (j=0; j<p_ndim && chk; j++) {
        /* check to see if at least one end point of the interval
         * represented by blo and bhi falls in the interval
         * represented by lo and hi */
        if (!((tlo[j] >= lo[j] && tlo[j] <= hi[j]) ||
              (thi[j] >= lo[j] && thi[j] <= hi[j]))) {
          chk = false;
        }
      }
      /* store blocks that overlap request region in
         proclist */
      if (chk) {
        proclist[cnt] = i;
        cnt++;
      }
    }
    *np = cnt;

    /* fill map array with block coordinates */
    for (i=0; i<cnt; i++) {
      offset = i*2*p_ndim;
      j = proclist[i];
      distribution(j,tlo,thi);
      for (j=0; j<p_ndim; j++) {
        map[offset + j] = lo[j] < tlo[j] ? tlo[j] : lo[j];
        map[offset + p_ndim + j] = hi[j] > thi[j] ? thi[j] : hi[j];
      }
    }
  }
  return true;
}

/**
 * Locate process that owns element corresponding to subscript
 * @param[in] subscript n-tuple identifiying element in array
 * @param[out] owner process that owns element
 * @return false if element out of bounds
 */
bool p_GA::locate(const int64_t *subscript, int *owner)
{
  int d, proc, dpos, proc_s[MAXDIM];
  for(d=0, *owner=-1; d< p_ndim; d++)
    if(subscript[d]< 0 || subscript[d]>=p_dims[d]) return false;
  if (p_distr == REGULAR) {
    for(d = 0, dpos = 0; d< p_ndim; d++){
      XGA_FINDBLOCK_M(p_mapc + dpos, p_proc_grid[d], p_scale[d],
          subscript[d], &proc_s[d]);
      dpos += p_proc_grid[d];
    }

    XGA_COMPUTEINDEX_M(&proc, p_ndim, proc_s, p_proc_grid);

    *owner = proc;
  } else {
    int i;
    int index[MAXDIM];
    XGA_FIND_BLOCK_INDICES_FROM_SUBSCRIPT_M(subscript,index);
    XGA_FIND_BLOCK_FROM_INDICES_M(i,index);
    *owner = i;
  }
  return true;
}

/**
 * Access data corresponding to a specific patch
 * @param[in] plo,phi lower and upper indices of patch
 * @param[out] rptr pointer to data
 * @param[out] ld array of strides for data
 */
void p_GA::accessPtr(int64_t *plo, int64_t *phi, void **rptr, int64_t *ld)
{
  int ow, i;
  char *lptr;
  /* lo, hi must be on this proc */
  if (!locate(plo, &ow))
    p_env->error("(accessPtr) locate lower corner failed",-1);
  if (p_group->rank() != ow)
    p_env->error("(accessPtr) cannot access lower corner of patch",-1);
  if (!locate(phi, &ow))
    p_env->error("(accessPtr) locate upper corner failed",-1);
  if (p_group->rank() != ow)
    p_env->error("(accessPtr) cannot access upper corner of patch",-1);
  for (i=0; i<p_ndim; i++) {
    if (plo[i]>phi[i])
      XGA_REGIONERROR_M(p_ndim, plo, phi, i);
  }
  XGA_LOCATION_M(ow, plo, &lptr, ld);
  *rptr = static_cast<void*>(lptr);
}

/**
 * Access data corresponding to a specific block
 * @param[in] index indices of block in proc grid or block cyclic layout
 * @param[out] rptr pointer to data
 * @param[out] ld array of strides for block
 */
void p_GA::accessBlockGridPtr(int *l_index, void **rptr, int64_t *ld)
{
  int i, j;
  int inode, last;
  int64_t ldims[MAXDIM], lld[MAXDIM];
  int64_t block_idx[MAXDIM], block_count[MAXDIM];
  int64_t ldidx[MAXDIM];
  int64_t tlo, thi, offset, factor;
  if (p_distr == REGULAR) {
    int iproc;
    int64_t lo[MAXDIM], hi[MAXDIM];
    XGA_FIND_BLOCK_FROM_INDICES_M(iproc,l_index);
    if (iproc != p_group->rank())
      p_env->error("(accessBlockGridPtr) block is not owned by this process",
          iproc);
    *rptr = ptr[iproc];
    XGA_OWNS_M(iproc, lo, hi);
    for (i=0; i<p_ndim; i++) ld[i] = hi[i]-lo[i]+1;
    return;
  } else if (p_distr == TILED) {
    /* find out what processor block is located on */
    XGA_FIND_TILE_PROC_FROM_INDICES_M(inode, l_index);

    /* get proc indices of processor that owns block */
    XGA_FIND_TILE_PROC_INDICES_M(inode, proc_index);
    last = p_ndim-1;

    for (i=0; i<p_ndim; i++)  {
      tlo = l_index[i]*blk_size[i]+1;
      thi = (l_index[i]+1)*blk_size[i];
      if (thi >= p_dims[i]) thi = p_dims[i]-1;
      ldims[i] = (thi - tlo + 1);
      if (i<last) ld[i] = ldims[i];
    }
  } else if (p_distr == TILED_IRREG) {
    /* find out what processor block is located on */
    XGA_FIND_TILE_PROC_FROM_INDICES_M(inode, l_index);

    /* get proc indices of processor that owns block */
    XGA_FIND_TILE_PROC_INDICES_M(inode, proc_index);
      last = p_ndim-1;

      offset = 0;
      for (i=0; i<p_ndim; i++) {
        lld[i] = 0;
        ldidx[i] = 0;
        tlo = p_mapc[offset+l_index[i]];
        if (l_index[i] < p_proc_grid[i]-1) {
          thi = p_mapc[offset+l_index[i]+1]-1;
        } else {
          thi = p_dims[i]-1;
        }
        ldims[i] = thi - tlo + 1;
        for (j = proc_index[i]; j<p_proc_grid[i]; j += p_proc_grid[i]) {
          tlo = p_mapc[offset+j];
          if (j < p_proc_grid[i]-1) {
            thi = p_mapc[offset+j+1]-1;
          } else {
            thi = p_dims[i]-1;
          }
          lld[i] += (thi-tlo+1);
          if (j < l_index[i]) {
            ldidx[i] += (thi-tlo+1);
          }
        }
        if (i<last) ld[i] = ldims[i];
        offset += p_proc_grid[i];
      }
  } else if (p_distr == SCALAPACK) {
    for (i=0; i<p_ndim; i++) {
      int nblocks = p_dims[i]/blk_dims[i];
      if (l_index[i] < 0 || l_index[i] >= nblocks)
        p_env->error("(accessBlockGridPtr) block index is outside allowed values",
            l_index[i]);
    }
    /* find out what processor block is located on */
    XGA_FIND_PROC_FROM_SL_INDICES_M(inode, l_index);
    if (inode != p_group->rank()) {
      p_env->error("(accessBlockGridPtr) cannot access block owned"
          " by another process",inode);
    }


    /* get proc indices of processor that owns block */
    XGA_FIND_PROC_INDICES_M(inode, proc_index);
    last = p_ndim-1;

    for (i=0; i<p_ndim; i++)  {
      blk_size[i] = blk_dims[i]*p_proc_grid[i];
      blk_num[i] = p_dims[i]/blk_size[i];
      blk_inc[i] = p_dims[i]-blk_num[i]*blk_size[i];
      blk_ld[i] = blk_num[i]*blk_dims[i];
      hlf_blk[i] = blk_inc[i]/blk_dims[i];
    }
    int64_t blk_jinc;
    for (i=last; i>0; i--)  {
      ld[i-1] = blk_ld[i];
      /* initialize this so that it works if first block is partial
       * block */
      blk_jinc = p_dims[i]%blk_dims[i];
      if (blk_inc[i] > 0) {
        /* may need to add an extra block or a partial block to stride */
        if (proc_index[i]<hlf_blk[i]) {
          /* add a full block */
          blk_jinc = blk_dims[i];
        } else if (proc_index[i] == hlf_blk[i]) {
          /* add a partial block */
          blk_jinc = blk_inc[i]%blk_dims[i];
        } else {
          /* add nothing */
          blk_jinc = 0;
        }
      }
      ld[i-1] += blk_jinc;
    }
  }

  /* Find the local grid index of block relative to local block grid and
     store result in block_idx.
     Find physical dimensions of locally held data and store in 
     lld and set values in ldim, which is used to evaluate the
     offset for the requested block. */
  if (p_distr == TILED) {
    for (i=0; i<p_ndim; i++) {
      block_idx[i] = 0;
      block_count[i] = 0;
      lld[i] = 0;
      tlo = 0;
      thi = -1;
      for (j=proc_index[i]; j<num_blks[i]; j += p_proc_grid[i]) {
        tlo = j*blk_size[i] + 1;
        thi = (j+1)*blk_size[i];
        if (thi > p_dims[i]) thi = p_dims[i]-1;
        lld[i] += (thi - tlo + 1);
        if (j<l_index[i]) block_idx[i]++;
        block_count[i]++;
      }
    }
    /* Evaluate offset for requested block. The algorithm used goes like this:
     *    The contribution from the fastest dimension is
     *      block_idx[0]*block_dims[0]*...*block_dims[p_ndim-1];
     *    The contribution from the second fastest dimension is
     *      block_idx[1]*lld[0]*block_dims[1]*...*block_dims[p_ndim-1];
     *    The contribution from the third fastest dimension is
     *      block_idx[2]*lld[0]*lld[1]*block_dims[2]*...*block_dims[p_ndim-1];
     *    etc.
     *    If block_idx[i] is equal to the total number of blocks contained on that
     *    processor minus 1 (the index is at the edge of the array) and the index
     *    i is greater than the index of the dimension, then instead of using the
     *    block dimension, use the fractional dimension of the edge block (which
     *    may be smaller than the block dimension)
     */
    offset = 0;
    for (i=0; i<p_ndim; i++) {
      factor = 1;
      for (j=0; j<i; j++) {
        factor *= lld[j];
      }
      for (j=i; j<p_ndim; j++) {
        if (j > i && block_idx[j] > block_count[j]-1) {
          factor *= ldims[j];
        } else {
          factor *= blk_size[j];
        }
      }
      offset += block_idx[i]*factor;
    }
  } else if (p_distr == TILED_IRREG) {
    /* Evaluate offset for requested block. This algorithm is similar to the
     * algorithm for reqularly tiled data layouts.
     *    The contribution from the fastest dimension is
     *      ldidx[0]*ldims[2]*...*ldims[p_ndim-1];
     *    The contribution from the second fastest dimension is
     *      lld[0]*ldidx[1]*ldims[2]*...*ldims[p_ndim-1];
     *    The contribution from the third fastest dimension is
     *      lld[0]*lld[1]*ldidx[2]*ldims[3]*...*ldims[p_ndim-1];
     *    etc.
     */
    offset = 0;
    for (i=0; i<p_ndim; i++) {
      factor = 1;
      for (j=0; j<i; j++) {
        factor *= lld[j];
      }
      factor *= ldidx[i];
      for (j=i+1; j<p_ndim; j++) {
        factor *= ldims[j];
      }
      offset += factor;
    }
  } else if (p_distr == SCALAPACK) {
    /* Evalauate offset for block */
    offset = 0;
    factor = 1;
    for (i = p_ndim-1; i>=0; i--) {
      offset += ((l_index[i]-proc_index[i])/p_proc_grid[i])*blk_dims[i]*factor;
      if (i>0) factor *= ld[i-1];
    }
//    printf("p[%d] (accessBlockGridPtr) offset: %ld\n",p_group->rank(),offset);
  }

  *rptr = static_cast<void*>(static_cast<char*>(ptr[inode])+offset*p_elemsize);

}

/**
 * Return pointer to data corresponding to block indexed by idx.
 * Assume C-style ordering
 * @param[in] idx index of block
 * @param[out] rptr pointer to data
 * @param[out] ld array of strides for block
 */
void p_GA::accessBlockPtr(int idx, void **rptr, int64_t *ld)
{
  if (p_distr == REGULAR) {
    if (idx != p_group->rank())
      p_env->error("(accessBlockPtr) can only access local data",idx);
    *rptr = ptr[idx];
    int64_t lo[MAXDIM], hi[MAXDIM];
    XGA_OWNS_NO_HANDLE_M(idx, lo, hi);
    int i;
    for (i=1; i<p_ndim; i++) ld[i-1] = hi[i]-lo[i]+1;
  } else if (p_distr == SCALAPACK || p_distr == TILED ||
      p_distr == TILED_IRREG) {
    int index[MAXDIM];
    if (idx < 0 || idx >= this->block_total)
      p_env->error("(accessBlockPtr) index out of bounds",idx);
    XGA_FIND_BLOCK_INDICES_M(idx,index);
    accessBlockGridPtr(index, rptr, ld);
  } else {
    p_env->error("(accessBlockPtr) unknown data distribution",idx);
  }
}

/**
 * Return pointer to data owned by this processors
 * @param[out] rptr pointer to local data
 * @param[out] nelem number of elements owned by this processor
 */
void p_GA::accessSegmentPtr(void **rptr, int64_t *nelem)
{
  *rptr = ptr[p_group->rank()];
  *nelem = p_size/p_elemsize;
}

/**
 * Release data corresponding to a specific patch
 * @param[in] plo,phi lower and upper indices of patch
 */
void p_GA::releasePtr(int64_t *plo, int64_t *phi)
{
  /* Currently implemented as a no-op */
}
void p_GA::releaseUpdatePtr(int64_t *plo, int64_t *phi)
{
  /* Currently implemented as a no-op */
}

/**
 * Release data corresponding to a specific block
 * in the proc grid array
 * @param[in] index indices of block in proc grid
 */
void p_GA::releaseBlockGridPtr(int *index)
{
  /* Currently implemented as a no-op */
}
void p_GA::releaseUpdateBlockGridPtr(int *index)
{
  /* Currently implemented as a no-op */
}

/**
 * Release data corresponding to a specific block indexed
 * using a C-style indexing convention
 * @param[in] index index of block
 */
void p_GA::releaseBlockPtr(int index)
{
  /* Currently implemented as a no-op */
}
void p_GA::releaseUpdateBlockPtr(int index)
{
  /* Currently implemented as a no-op */
}

/**
 * Release data corresponding to this process
 */
void p_GA::releaseSegmentPtr()
{
  /* Currently implemented as a no-op */
}
void p_GA::releaseUpdateSegmentPtr()
{
  /* Currently implemented as a no-op */
}

/**
 * Set all values in the array to zero
 */
void p_GA::zero()
{
  int rank = p_group->rank();
  size_t size = static_cast<size_t>(p_size);
  memset(ptr[rank], 0, size);
}

/**
 * Fill array with a single value
 * @param value pointer to value being filled
 */
void p_GA::fill(void *value)
{
  int64_t lo[MAXDIM], hi[MAXDIM];
  int me = p_group->rank();
  distribution(me, lo, hi);
  int64_t nelems = 1;
  int64_t i;
  for (i=0; i<p_ndim; i++) nelems *= hi[i]-lo[i]+1;
  switch (static_cast<int>(p_datatype)) {
    case XGA_INT: 
      {
        int *iptr = static_cast<int*>(ptr[me]);
        int ival = *static_cast<int*>(value);
        for (i=0; i<nelems; i++) {
          iptr[i] = ival;
        }
      }
      break;
    case XGA_LONG:
      {
        long *lptr = static_cast<long*>(ptr[me]);
        long lval = *static_cast<long*>(value);
        for (i=0; i<nelems; i++) {
          lptr[i] = lval;
        }
      }
      break;
    case XGA_LONGLONG:
      {
        long long *llptr = static_cast<long long*>(ptr[me]);
        long long llval = *static_cast<long long*>(value);
        for (i=0; i<nelems; i++) {
          llptr[i] = llval;
        }
      }
      break;
    case XGA_FLOAT:
      {
        float *fptr = static_cast<float*>(ptr[me]);
        float fval = *static_cast<float*>(value);
        for (i=0; i<nelems; i++) {
          fptr[i] = fval;
        }
      }
      break;
    case XGA_DOUBLE:
      {
        double *dptr = static_cast<double*>(ptr[me]);
        double dval = *static_cast<double*>(value);
        for (i=0; i<nelems; i++) {
          dptr[i] = dval;
        }
      }
      break;
    case XGA_COMPLEX:
      {
        std::complex<float> *cptr = static_cast<std::complex<float>*>(ptr[me]);
        std::complex<float> cval = *static_cast<std::complex<float>*>(value);
        for (i=0; i<nelems; i++) {
          cptr[i] = cval;
        }
      }
      break;
    case XGA_DCOMPLEX:
      {
        std::complex<double> *zptr = static_cast<std::complex<double>*>(ptr[me]);
        std::complex<double> zval = *static_cast<std::complex<double>*>(value);
        for (i=0; i<nelems; i++) {
          zptr[i] = zval;
        }
      }
      break;
    case XGA_UNKNOWN:
      {
        char *uptr = static_cast<char*>(ptr[me]);
        char *uval = static_cast<char*>(value);
        for (i=0; i<nelems; i++) {
          memcpy(uptr,uval,p_elemsize);
          uptr += p_elemsize;
        }
      }
      break;
    default:
      p_env->error("(fill) unknown data type",static_cast<int>(p_datatype));
  }
}

/**
 * Scale all elements of array
 * @param value scale factor for all elements
 */
void p_GA::scale(void *value)
{
  int64_t lo[MAXDIM], hi[MAXDIM];
  int me = p_group->rank();
  distribution(me, lo, hi);
  int64_t nelems = 1;
  int64_t i;
  for (i=0; i<p_ndim; i++) nelems *= hi[i]-lo[i]+1;
  switch (static_cast<int>(p_datatype)) {
    case XGA_INT: 
      {
        int *iptr = static_cast<int*>(ptr[me]);
        int ival = *static_cast<int*>(value);
        for (i=0; i<nelems; i++) {
          iptr[i] *= ival;
        }
      }
      break;
    case XGA_LONG:
      {
        long *lptr = static_cast<long*>(ptr[me]);
        long lval = *static_cast<long*>(value);
        for (i=0; i<nelems; i++) {
          lptr[i] *= lval;
        }
      }
      break;
    case XGA_LONGLONG:
      {
        long long *llptr = static_cast<long long*>(ptr[me]);
        long long llval = *static_cast<long long*>(value);
        for (i=0; i<nelems; i++) {
          llptr[i] *= llval;
        }
      }
      break;
    case XGA_FLOAT:
      {
        float *fptr = static_cast<float*>(ptr[me]);
        float fval = *static_cast<float*>(value);
        for (i=0; i<nelems; i++) {
          fptr[i] *= fval;
        }
      }
      break;
    case XGA_DOUBLE:
      {
        double *dptr = static_cast<double*>(ptr[me]);
        double dval = *static_cast<double*>(value);
        for (i=0; i<nelems; i++) {
          dptr[i] *= dval;
        }
      }
      break;
    case XGA_COMPLEX:
      {
        std::complex<float> *cptr = static_cast<std::complex<float>*>(ptr[me]);
        std::complex<float> cval = *static_cast<std::complex<float>*>(value);
        for (i=0; i<nelems; i++) {
          cptr[i] = cval;
        }
      }
      break;
    case XGA_DCOMPLEX:
      {
        std::complex<double> *zptr = static_cast<std::complex<double>*>(ptr[me]);
        std::complex<double> zval = *static_cast<std::complex<double>*>(value);
        for (i=0; i<nelems; i++) {
          zptr[i] = zval;
        }
      }
      break;
    default:
      p_env->error("(scale) unknown data type",static_cast<int>(p_datatype));
  }
}

/**
 * Scale all elements in a patch of an array
 * @param lo, hi bounding indices of patch
 * @param value scale factor for elements in patch
 */
void p_GA::scalePatch(int64_t *lo, int64_t *hi, void *value)
{
  int ndim, type;
  int64_t loA[MAXDIM], hiA[MAXDIM];
  int64_t ld[MAXDIM];
  char *src_data_ptr;
  int num_blocks, nproc;
  int me= p_group->rank();

  type = this->p_datatype;
  ndim = this->p_ndim;
  num_blocks = this->block_total;

  this->localInit();
  while (nextLocalBlock(loA,hiA,&src_data_ptr,ld)) {
    int64_t offset, j, jtmp, chk;
    int64_t loS[MAXDIM];
    /* loA is changed by patch_intersect, so
       save a copy */
    for (j=0; j<ndim; j++) {
      loS[j] = loA[j];
    }

    /*  determine subset of my local patch to access  */
    /*  Output is in loA and hiA */
    if (patch_intersect(lo, hi, loA, hiA, ndim)) {
      /* Check for partial overlap */
      bool chk = true;
      for (j=0; j<ndim; j++) {
        if (loS[j] < loA[j]) {
          chk=false;
          break;
        }
      }
      if (!chk) {
        /* Evaluate additional offset for pointer */
        jtmp = 1;
        offset += (loA[ndim-1]-loS[ndim-1])*jtmp;
        for (j=ndim-2; j>=0; j--) {
          jtmp *= ld[j];
          offset += (loA[j]-loS[j])*jtmp;
        }
#define SET_OFFSET_M(_type, _ptr, _offset)                          \
{                                                                   \
  _ptr = reinterpret_cast<char*>(reinterpret_cast<_type*>(_ptr)     \
      +offset);                                                     \
}
        switch (type){
          case XGA_INT:
            SET_OFFSET_M(int,src_data_ptr,offset);
            break;
          case XGA_DCOMPLEX:
            SET_OFFSET_M(std::complex<double>,src_data_ptr,offset);
            break;
          case XGA_COMPLEX:
            SET_OFFSET_M(std::complex<float>,src_data_ptr,offset);
            break;
          case XGA_DOUBLE:
            SET_OFFSET_M(double,src_data_ptr,offset);
            break;
          case XGA_FLOAT:
            SET_OFFSET_M(float,src_data_ptr,offset);
            break;
          case XGA_LONG:
            SET_OFFSET_M(long,src_data_ptr,offset);
            break;
          case XGA_LONGLONG:
            SET_OFFSET_M(long long,src_data_ptr,offset);
            break;
          default: p_env->error(" (scalePatch) wrong data type ",type);
        }
      }
#undef SET_OFFSET_M

      /* scale all values in patch by value */
      void *src_ptr = reinterpret_cast<void*>(src_data_ptr);
      switch (type){
        case XGA_INT:
          scale_patch_values<int>(value, loA, hiA, ld, src_ptr);
          break;
        case XGA_DCOMPLEX:
          scale_patch_values<std::complex<double> >(value, loA, hiA, ld,
              src_ptr);
          break;
        case XGA_COMPLEX:
          scale_patch_values<std::complex<float> >(value, loA, hiA, ld,
              src_ptr);
          break;
        case XGA_DOUBLE:
          scale_patch_values<double>(value, loA, hiA, ld, src_ptr);
          break;
        case XGA_FLOAT:
          scale_patch_values<float>(value, loA, hiA, ld, src_ptr);
          break;
        case XGA_LONG:
          scale_patch_values<long>(value, loA, hiA, ld, src_ptr);
          break;
        case XGA_LONGLONG:
          scale_patch_values<long long>(value, loA, hiA, ld, src_ptr);
          break;
        default: p_env->error(" (scalePatch) wrong data type ",type);
      }
    }
  }
}

/**
 * Utility function to print subscripts
 * @param[in] pre character string before subscript
 * @param[in] ndim dimension of subscript
 * @param[in] subscript array containing subscript values
 * @param[in] post character string after subscript
 */
void p_GA::printSubscript(const char *pre, const int ndim,
    const int64_t *subscript, const char *post)
{
  int i;
  printf("%s [",pre);
  for (i=0; i<ndim; i++) {
    printf("%ld",subscript[i]);
    if (i==ndim-1) printf("] %s",post);
    else printf(",");
  }
}

/**
 * Wrapper for error function in environment class
 * @param msg message to print with error
 * @param code error code to exit with
 */
void p_GA::error(const char *msg, int code)
{
  p_env->error(msg, code);
}


/**
 * Utility function to convert XGA datatype into an actual size
 * @param type XGA datatype
 */
int p_GA::xga_sizeof(int type) {
  if (type == XGA_INT) {
    return sizeof(int);
  } else if (type == XGA_DOUBLE) {
    return sizeof(double);
  } else if (type == XGA_FLOAT) {
    return sizeof(float);
  } else if (type == XGA_LONG) {
    return sizeof(long);
  } else if (type == XGA_LONGLONG) {
    return sizeof(long long);
  } else if (type == XGA_COMPLEX) {
    return sizeof(std::complex<float>);
  } else if (type == XGA_DCOMPLEX) {
    return sizeof(std::complex<double>);
  }
  return 0;
}

/**
 * Utility function to allocate n XGA datatype elements
 * @param n number of elements
 * @param type XGA datatype
 * @return pointer to allocated data
 */
void* p_GA::xga_malloc(int64_t n, int type)
{
  if (type == XGA_INT) {
    return new int[n*sizeof(int)];
  } else if (type == XGA_DOUBLE) {
    return new double[n*sizeof(double)];
  } else if (type == XGA_FLOAT) {
    return new float[n*sizeof(float)];
  } else if (type == XGA_LONG) {
    return new long[n*sizeof(long)];
  } else if (type == XGA_LONGLONG) {
    return new long long[n*sizeof(long long)];
  } else if (type == XGA_COMPLEX) {
    return new std::complex<float>[n*sizeof(std::complex<float>)];
  } else if (type == XGA_DCOMPLEX) {
    return new std::complex<double>[n*sizeof(std::complex<double>)];
  }
  return NULL;
}

/**
 * Utility function to free memory allocated by xga_malloc
 * @param ptr void pointer allocated by xga_malloc
 * @param type XGA datatype
 */
void p_GA::xga_free(void *ptr, int type)
{
  if (type == XGA_INT) {
    int *iptr = reinterpret_cast<int*>(ptr);
    delete [] iptr;
  } else if (type == XGA_DOUBLE) {
    double *dptr = reinterpret_cast<double*>(ptr);
    delete [] dptr;
  } else if (type == XGA_FLOAT) {
    float *fptr = reinterpret_cast<float*>(ptr);
    delete [] fptr;
  } else if (type == XGA_LONG) {
    long *lptr = reinterpret_cast<long*>(ptr);
    delete [] lptr;
  } else if (type == XGA_LONGLONG) {
    long long *llptr = reinterpret_cast<long long*>(ptr);
    delete [] llptr;
  } else if (type == XGA_COMPLEX) {
    std::complex<float> *cptr = reinterpret_cast<std::complex<float>*>(ptr);
    delete [] cptr;
  } else if (type == XGA_DCOMPLEX) {
    std::complex<double> *zptr = reinterpret_cast<std::complex<double>*>(ptr);
    delete [] zptr;
  }
}

}
