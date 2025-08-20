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
  p_active = false;
}

/**
 * Basic destructor
 */
p_GA::~p_GA()
{
  if (p_mapc) delete [] p_mapc;
  p_alloc->free();
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
 * @param[in] mapc array containing partitions along each axis
 * @param[in] nblock array containing processor decomposition
 */
void p_GA::setIrregularDistribution(int64_t *mapc, int *nblock)
{
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
  for (i=0; i=p_ndim; i++) proc_grid[i] = nblock[i];
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
      nblock[i] = pe[i];
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
      skip = nblock[i];
      jtot = 0;
      for (j=index[i]; j<blk_num[i]; j += skip) {
        imin = j*blk_size[i] + 1;
        imax = (j+1)*blk_size[i];
        if (imax >= p_dims[i]) imax = p_dims[i]-1;
        jtot += (imax-imin+1);
      }
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
      skip = nblock[i];
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
      skip = nblock[i];
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
      scale[i] = static_cast<double>(nblock[i])/static_cast<double>(p_dims[i]);
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
      XGA_FINDBLOCK_M(p_mapc+dpos,nblock[d], scale[d], lo[d], &procT[d]);
      dpos += nblock[d];
    }
    /* find "processor coordinates" for the upper corner and store them
     * in procB */
    for (d=0, dpos=0; d<p_ndim; d++) {
      XGA_FINDBLOCK_M(p_mapc+dpos,nblock[d], scale[d], hi[d], &procB[d]);
      dpos += nblock[d];
    }

    *np = 0;

    /* Find total number of processors containing data and return the
     * result in nelems. Also find the lowest "processor coordinates" of the
     * processor block containing data and return these in proc_subscript.
     */
    XGA_INITLOOP_M(&nelems, p_ndim, proc_subscript, procT, procB,
        nblock);
    proclist.resize(nelems);
    for (i=0; i<nelems; i++) {
      int64_t _lo[MAXDIM], _hi[MAXDIM];
      int _offset, proc;
      /* convert i to owner processor id using the current values in
         proc_subscript */
      XGA_COMPUTEINDEX_M(&proc, p_ndim, proc_subscript, nblock);
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
      XGA_UPDATESUBSCRIPT_M(p_ndim,proc_subscript,procT,procB,nblock);
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
      XGA_FINDBLOCK_M(p_mapc + dpos, nblock[d], scale[d],
          subscript[d], &proc_s[d]);
      dpos += nblock[d];
    }

    XGA_COMPUTEINDEX_M(&proc, p_ndim, proc_s, nblock);

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
  for (i=0; i<p_ndim; i++) {
    if (l_index[i] < 0 || l_index[i] >= blk_num[i])
      p_env->error("(accessBlockGridPtr) block index is outside allowed values",
          l_index[i]);
  }
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
      tlo = index[i]*blk_size[i]+1;
      thi = (index[i]+1)*blk_size[i];
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
        if (l_index[i] < nblock[i]-1) {
          thi = p_mapc[offset+l_index[i]+1]-1;
        } else {
          thi = p_dims[i]-1;
        }
        ldims[i] = thi - tlo + 1;
        for (j = proc_index[i]; j<nblock[i]; j += proc_grid[i]) {
          tlo = p_mapc[offset+j];
          if (j < nblock[i]-1) {
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
        offset += nblock[i];
      }
  } else if (p_distr == SCALAPACK) {
    /* find out what processor block is located on */
    XGA_FIND_PROC_FROM_SL_INDICES_M(inode, index);

    /* get proc indices of processor that owns block */
    XGA_FIND_PROC_INDICES_M(inode, proc_index);
    last = p_ndim-1;

    int64_t blk_jinc;
    for (i=0; i<last; i++)  {
      blk_dims[i] = blk_size[i]*proc_grid[i];
      blk_num[i] = p_dims[i]/blk_dims[i];
      blk_inc[i] = p_dims[i] - blk_num[i]*blk_dims[i];
      blk_ld[i] = blk_num[i]*blk_size[i];
      hlf_blk[i] = blk_inc[i]/blk_size[i];
      ld[i] = blk_ld[i];
      blk_jinc = p_dims[i]%blk_size[i];
      if (blk_inc[i] > 0) {
        if (proc_index[i]<hlf_blk[i]) {
          blk_jinc = blk_size[i];
        } else if (proc_index[i] == hlf_blk[i]) {
          blk_jinc = blk_inc[i]%blk_size[i];
        } else {
          blk_jinc = 0;
        }
      }
      ld[i] += blk_jinc;
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
      for (j=proc_index[i]; j<nblock[i]; j += proc_grid[i]) {
        tlo = j*blk_size[i] + 1;
        thi = (j+1)*blk_size[i];
        if (thi > p_dims[i]) thi = p_dims[i]-1;
        lld[i] += (thi - tlo + 1);
        if (j<index[i]) block_idx[i]++;
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
    for (i = 0; i<p_ndim; i++) {
      offset += ((index[i]-proc_index[i])/proc_grid[i])*blk_size[i]*factor;
      if (i<p_ndim-1) factor *= ld[i];
    }
  }

  *rptr = static_cast<void*>(static_cast<char*>(ptr[inode])+offset*p_elemsize);

}

}
