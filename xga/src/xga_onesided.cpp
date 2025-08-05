/* XGA private implementation */
#include "xga_environment.hpp"
#include "xga_private.hpp"

/* compute index of point subscripted by plo relative to point
 * subscripted by lo, for a block with dimensions dims */
#define XGA_COMPUTEPATCHINDEX_M(_ndim, _lo, _plo, _dims, _pidx){        \
  int64_t _d, _factor;                                                  \
  *_pidx = _plo[_ndim-1] -_lo[_ndim-1];                                 \
  for(_d=_ndim-1,_factor=1; _d>0; _d--){                                \
    _factor *= (_dims[_d]);                                             \
    *_pidx += _factor * (_plo[_d-1]-_lo[_d-1]);                         \
  }                                                                     \
}

/* compute count array */
#define XGA_COMPUTECOUNT_M(_ndim, _lo, _hi, _count){                    \
  int _d;                                                               \
  for (_d=0; _d<_ndim; _d++) _count[_ndim-1-_d] = _hi[_d]-_lo[_d] + 1;  \
}

/* this macro computes the strides on both the remote and local
 * processors that map out the data. ld and ldrem are the physical dimensions
 * of the memory on both the local and remote processors. */
#define XGA_SETSTRIDE_M(_ndim,_size,_ld,_ldrem,_stride_rem,_stride_loc){\
  int _i;                                                               \
  _stride_rem[0]= _stride_loc[0] = _size;                               \
  for(_i=_ndim-1;_i>0;_i--){                                            \
    _stride_rem[_ndim-1-_i] *= _ldrem[_i-1];                            \
    _stride_loc[_ndim-1-_i] *= _ld[_i-1];                               \
    _stride_rem[_ndim-1-_i+1] = _stride_rem[_ndim-1-_i];                \
    _stride_loc[_ndim-1-_i+1] = _stride_loc[_ndim-1-_i];                \
  }                                                                     \
}

namespace XGA {

/**
 * Copy data from local buffer to global array
 * @param[in] lo,hi bounding indices of block in global array
 * @param[in] buf pointer to first element in local buffer
 * @param[in] ld strides in local buffer
 */
void p_GA::put(int64_t *lo, int64_t *hi, void* buf, int64_t *ld)
{
  xga_request *req;
  putCommon(lo, hi, buf, ld, &req);
  p_env->wait(req);
}

/**
 * Internal implementation of put call that handles both blocking and
 * non-blocking variants
 * @param[in] lo,hi bounding indices of block in global array
 * @param[in] buf pointer to first element in local buffer
 * @param[in] ld strides in local buffer
 * @param[out] req non-blocking request handle
 */
void p_GA::putCommon(int64_t *lo, int64_t *hi, void* buf, int64_t *ld,
    xga_request **req)
{
  int counter = 0;
  int64_t stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
  int iproc;
  int stride_levels = p_ndim-1;
  printf("p[%d] (putCommon) got to 1 ld[0]: %ld\n",p_group->rank(),ld[0]);

  /* initial stride portion */

  printf("p[%d] (putCommon) Got to 2 lo[0]: %ld hi[0]: %ld lo[1]: %ld hi[1]: %ld\n",
      p_group->rank(),lo[0],hi[0],lo[1],hi[1]);
  initIterator(lo, hi);

  if (req != NULL) p_env->getXGARequest(req);

  int64_t ldrem[MAXDIM];
  int64_t idx_buf, *plo, *phi;
  char *pbuf, *prem;

  printf("p[%d] (putCommon) Got to 3\n",p_group->rank());
  while (nextBlock(&iproc, &plo, &phi, &prem, ldrem)) {
  printf("p[%d] (putCommon) Got to 4 proc: %d lo: %p hi: %p ldrem[0]: %ld\n",
      p_group->rank(),iproc,plo,phi,ldrem[0]);
  printf("p[%d] (putCommon) Got to 5 proc: %d lo[0]: %ld hi[0]: %ld lo[1]: %ld "
      "hi[1]: %ld\n",p_group->rank(),iproc,plo[0],phi[0],plo[1],phi[1]);
    /* find the right spot in the user buffer */
    XGA_COMPUTEPATCHINDEX_M(p_ndim, lo, plo, ld, &idx_buf);
    pbuf = p_elemsize*idx_buf + static_cast<char*>(buf);

    XGA_COMPUTECOUNT_M(p_ndim, plo, phi, count);
    printf("p[%d] (putCommon) count[0]: %ld count[1]: %ld ld[0]: %ld ldrem: %ld\n",
        p_group->rank(),count[0],count[1],ld[0],ldrem[0]);

    count[0] *= p_elemsize;
    XGA_SETSTRIDE_M(p_ndim, p_elemsize, ld, ldrem, stride_rem, stride_loc);
    printf("p[%d] (putCommon) stride_loc[0]: %ld stride_rem[0]: %ld\n",
        p_group->rank(),stride_loc[0],stride_rem[0]);
    if (req != NULL) {
      CMX::cmx_request *cmx_req = p_env->getCMXRequest(*req);
      p_alloc->nbputs(pbuf,stride_loc, prem, stride_rem,
          count, stride_levels, iproc, cmx_req);
    } else {
      p_alloc->puts(pbuf,stride_loc, prem, stride_rem,
          count, stride_levels, iproc);
    }
  }
  destroyIterator();
}

} // XGA namespace
