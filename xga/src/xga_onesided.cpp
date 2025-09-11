/* XGA private implementation */
#include "xga_environment.hpp"
#include "xga_private.hpp"

/* compute index of point subscripted by plo relative to point
 * subscripted by lo, for a block with dimensions dims */
#define XGA_COMPUTEPATCHINDEX_M(_ndim, _lo, _plo, _dims, _pidx){        \
  int64_t _d, _factor;                                                  \
  *_pidx = _plo[_ndim-1] -_lo[_ndim-1];                                 \
  for(_d=_ndim-1,_factor=1; _d>0; _d--){                                \
    _factor *= (_dims[_d-1]);                                           \
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
 * Synchronize global array across all processor that are hosting
 * the array
 */
void p_GA::sync()
{
  p_env->sync(p_group);
}

/**
 * Copy data from local buffer to global array
 * @param[in] lo,hi bounding indices of block in global array
 * @param[in] buf pointer to first element in local buffer
 * @param[in] ld strides in local buffer
 */
void p_GA::put(int64_t *lo, int64_t *hi, void* buf, int64_t *ld)
{
  xga_request *req;
  p_env->getXGARequest(&req);
  putCommon(lo, hi, buf, ld, req);
  p_env->wait(req);
}

/**
 * Copy data from global array to local buffer
 * @param[in] lo,hi bounding indices of block in global array
 * @param[in] buf pointer to first element in local buffer
 * @param[in] ld strides in local buffer
 */
void p_GA::get(int64_t *lo, int64_t *hi, void* buf, int64_t *ld)
{
  xga_request *req;
  p_env->getXGARequest(&req);
  getCommon(lo, hi, buf, ld, req);
  p_env->wait(req);
}

/**
 * Accumulate data from local buffer to global array
 * @param[in] lo,hi bounding indices of block in global array
 * @param[in] buf pointer to first element in local buffer
 * @param[in] ld strides in local buffer
 * @param[in] alpha scale factor for adding contents of buffer
 *            to global array
 */
void p_GA::acc(int64_t *lo, int64_t *hi, void* buf, int64_t *ld, void *alpha)
{
  xga_request *req;
  p_env->getXGARequest(&req);
  accCommon(lo, hi, buf, ld, alpha, req);
  p_env->wait(req);
}

/**
 * Scatter values to random locations in a global array
 * @param[in] v array containing values to be scattered to array. The
 *            type of values in v must match the type of values
 *            in the global array
 * @param[in] subscript array of indices representing locations of
 *            values in global array. Each ndim locations represents
 *            the index location of one value
 * @param[in] nv number of values to scattered
 * @param[in] idxtype flag indicating size of index type (0 for int,
 *            1 for int64_t)
 */
void p_GA::scatter(void *v, void *subscript, int64_t nv, int idxtype)
{
  xga_request *req;
  p_env->getXGARequest(&req);
  gatscatCommon(XGA_SCATTER, v, subscript, idxtype, nv, NULL, req);
  p_env->wait(req);
}

/**
 * Gather values from random locations in a global array
 * @param[in] v array containing values gathered from array. The
 *            type of values in v must match the type of values
 *            in the global array
 * @param[in] subscript array of indices representing locations of
 *            values in global array. Each ndim locations represents
 *            the index location of one value
 * @param[in] nv number of values to gathered
 * @param[in] idxtype flag indicating size of index type (0 for int,
 *            1 for int64_t)
 */
void p_GA::gather(void *v, void *subscript, int64_t nv, int idxtype)
{
  xga_request *req;
  p_env->getXGARequest(&req);
  gatscatCommon(XGA_GATHER, v, subscript, idxtype, nv, NULL, req);
  p_env->wait(req);
}

/**
 * Accumulate values to random locations in a global array
 * @param[in] v array containing values to be accumulated to array. The
 *            type of values in v must match the type of values
 *            in the global array
 * @param[in] subscript array of indices representing locations of
 *            values in global array. Each ndim locations represents
 *            the index location of one value
 * @param[in] nv number of values to accumulated
 * @param[in] scale scale factor to multiply each value by before being
 *            accumulated
 * @param[in] idxtype flag indicating size of index type (0 for int,
 *            1 for int64_t);
 */
void p_GA::scatterAcc(void *v, void *subscript, int64_t nv, void *alpha,
    int idxtype)
{
  xga_request *req;
  p_env->getXGARequest(&req);
  gatscatCommon(XGA_SCATTERACC, v, subscript, idxtype, nv, alpha, req);
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
    xga_request *req)
{
  int counter = 0;
  int64_t stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
  int iproc;
  int stride_levels = p_ndim-1;

  /* initial stride portion */

  initIterator(lo, hi);

  int64_t ldrem[MAXDIM];
  int64_t idx_buf, *plo, *phi;
  char *pbuf, *prem;

  while (nextBlock(&iproc, &plo, &phi, &prem, ldrem)) {
    /* find the right spot in the user buffer */
    XGA_COMPUTEPATCHINDEX_M(p_ndim, lo, plo, ld, &idx_buf);
    pbuf = p_elemsize*idx_buf + static_cast<char*>(buf);

    XGA_COMPUTECOUNT_M(p_ndim, plo, phi, count);

    count[0] *= p_elemsize;
    XGA_SETSTRIDE_M(p_ndim, p_elemsize, ld, ldrem, stride_rem, stride_loc);
    if (req != NULL) {
      CMX::cmx_request *cmx_req = p_env->getCMXRequest(req);
      p_alloc->nbputs(pbuf, stride_loc, prem, stride_rem,
          count, stride_levels, iproc, cmx_req);
    } else {
      p_alloc->puts(pbuf, stride_loc, prem, stride_rem,
          count, stride_levels, iproc);
    }
  }
  destroyIterator();
}

/**
 * Internal implementation of get call that handles both blocking and
 * non-blocking variants
 * @param[in] lo,hi bounding indices of block in global array
 * @param[in] buf pointer to first element in local buffer
 * @param[in] ld strides in local buffer
 * @param[out] req non-blocking request handle
 */
void p_GA::getCommon(int64_t *lo, int64_t *hi, void* buf, int64_t *ld,
    xga_request *req)
{
  int counter = 0;
  int64_t stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
  int iproc;
  int stride_levels = p_ndim-1;

  /* initial stride portion */

  initIterator(lo, hi);

  int64_t ldrem[MAXDIM];
  int64_t idx_buf, *plo, *phi;
  char *pbuf, *prem;

  while (nextBlock(&iproc, &plo, &phi, &prem, ldrem)) {
    /* find the right spot in the user buffer */
    XGA_COMPUTEPATCHINDEX_M(p_ndim, lo, plo, ld, &idx_buf);
    pbuf = p_elemsize*idx_buf + static_cast<char*>(buf);

    XGA_COMPUTECOUNT_M(p_ndim, plo, phi, count);

    count[0] *= p_elemsize;
    XGA_SETSTRIDE_M(p_ndim, p_elemsize, ld, ldrem, stride_rem, stride_loc);
    if (req != NULL) {
      CMX::cmx_request *cmx_req = p_env->getCMXRequest(req);
      p_alloc->nbgets(prem, stride_rem, pbuf, stride_loc,
          count, stride_levels, iproc, cmx_req);
    } else {
      p_alloc->puts(prem, stride_rem, pbuf, stride_loc,
          count, stride_levels, iproc);
    }
  }
  destroyIterator();
}

/**
 * Internal implementation of accumulate call that handles both blocking
 * and non-blocking variants
 * @param[in] lo,hi bounding indices of block in global array
 * @param[in] buf pointer to first element in local buffer
 * @param[in] ld strides in local buffer
 * @param[in] alpha scale factor for adding contents of buffer
 *            to global array
 * @param[out] req non-blocking request handle
 */
void p_GA::accCommon(int64_t *lo, int64_t *hi, void* buf, int64_t *ld,
    void* alpha, xga_request *req)
{
  int counter = 0;
  int64_t stride_rem[MAXDIM], stride_loc[MAXDIM], count[MAXDIM];
  int iproc;
  int stride_levels = p_ndim-1;
  int op;
  void *scale =&alpha;

  /* determine what operation is performed in CMX runtime */
  if (p_datatype == XGA_INT) {
    op = CMX_ACC_INT;
  } else if (p_datatype == XGA_LONG) {
    op = CMX_ACC_LNG;
  } else if (p_datatype == XGA_FLOAT) {
    op = CMX_ACC_FLT;
  } else if (p_datatype == XGA_DOUBLE) {
    op = CMX_ACC_DBL;
  } else if (p_datatype == XGA_COMPLEX) {
    op = CMX_ACC_CPL;
  } else if (p_datatype == XGA_DCOMPLEX) {
    op = CMX_ACC_DCP;
  } else {
    p_env->error("Accumulate operation not supported for this data type",
        p_datatype);
  }


  /* initial stride portion */

  initIterator(lo, hi);

  int64_t ldrem[MAXDIM];
  int64_t idx_buf, *plo, *phi;
  char *pbuf, *prem;

  while (nextBlock(&iproc, &plo, &phi, &prem, ldrem)) {
    /* find the right spot in the user buffer */
    XGA_COMPUTEPATCHINDEX_M(p_ndim, lo, plo, ld, &idx_buf);
    pbuf = p_elemsize*idx_buf + static_cast<char*>(buf);

    XGA_COMPUTECOUNT_M(p_ndim, plo, phi, count);

    count[0] *= p_elemsize;
    XGA_SETSTRIDE_M(p_ndim, p_elemsize, ld, ldrem, stride_rem, stride_loc);
    if (req != NULL) {
      CMX::cmx_request *cmx_req = p_env->getCMXRequest(req);
      p_alloc->nbaccs(op, alpha, pbuf, stride_loc, prem, stride_rem,
          count, stride_levels, iproc, cmx_req);
    } else {
      p_alloc->accs(op, alpha, pbuf, stride_loc, prem, stride_rem,
          count, stride_levels, iproc);
    }
  }
  destroyIterator();
}

/**
 * Generic routine for implementing gather, scatter and scatter-accumulate
 * operations
 * @param[in] op enum indicating which operation is being performed
 * @param[in] v pointer to array containing values to be scattered
 * @param[in] subscript array containing indices of values to be moved
 * @param[in] idxtype flag indicating size of index (0 int, 1 int64_t)
 * @param[in] nv number of values being moved
 * @param[in] alpha scale factor that is used in scatter-accumulate operation
 * @param[out] req non-blocking request handle
 */
void p_GA::gatscatCommon(int op, void *v, void *subscript, int idxtype,
    int64_t nv, void *alpha, xga_request *req)
{
  int me = p_group->rank();
  int nprocs = p_group->size();
  int64_t *header = new int64_t[nprocs];
  int64_t *list = new int64_t[nv];
  int64_t *nelems = new int64_t[nprocs];
  int *subs;
  int64_t *subs64;
  int rc;
  int64_t idx;
  cmx_giov_t desc;
  /* initialize linked list data structures */
  int64_t i, j;
  for (i=0; i<nprocs; i++) {
    nelems[i] = 0;
    header[i] = -1;
  }
  for (i=0; i<nv; i++) {
    list[i] = 0;
  }
  if (idxtype == 0) {
    subs = static_cast<int*>(subscript);
  } else if (idxtype == 1) {
    subs64 = static_cast<int64_t*>(subscript);
  } else {
    p_env->error("gatscatCommon: unknown index type",idxtype);
  }
  int64_t maxlen = 1;
  int64_t *subscript_ptr;
  if (idxtype == 0) {
    subscript_ptr = new int64_t[p_ndim];
  }
  /* set up linked list that partitions elements defined in subscripts array
   * into groups corresponding to elements remote processor
   */
  for (i=0; i<nv; i++) {
    if (idxtype == 0) {
      int *tsub;
      tsub = subs+i*p_ndim;
      int k;
      for (k=0; k<p_ndim; k++) 
        subscript_ptr[k] = static_cast<int64_t>(tsub[k]);
    } else {
      subscript_ptr = subs64+i*p_ndim;
    }
    int iproc;
    if (!locate(subscript_ptr, &iproc)) {
      printSubscript("Invalid subscript", p_ndim, subscript_ptr, "\n");
      p_env->error("failed element: ",i);
    }
    nelems[iproc]++;
    if (maxlen<nelems[iproc]) maxlen = nelems[iproc];
    j = header[iproc];
    header[iproc] = i;
    list[i] = j;
  }
  char *buf =  new char[2*maxlen*sizeof(void*)];
  void **ptr_loc = reinterpret_cast<void**>(buf);
  void **ptr_rem = reinterpret_cast<void**>(buf+maxlen*sizeof(void*));
  /* loop over processors */
  int iproc;
  for (iproc=0; iproc<nproc; iproc++) {
    if (nelems[iproc] > 0) {
      /* loop through linked list to find all data elements associated with
       * the remote processor iproc
       */
      idx = header[iproc];
      j = 0;
      while (idx > -1) {
        if (idxtype == 0) {
          int *tsub;
          tsub = subs+idx*p_ndim;
          int k;
          for (k=0; k<p_ndim; k++) 
            subscript_ptr[k] = static_cast<int64_t>(tsub[k]);
        } else {
          subscript_ptr = subs64+idx*p_ndim;
        }
        if (p_distr == REGULAR) {
          /* XGA_LOCATE_PTR_M modifies the value of the processor variable for
           * some data distributions so make a temporary copy
           */
          int tproc = iproc;
          XGA_LOCATE_PTR_M(tproc, (subscript_ptr), ptr_rem+j);
          ptr_loc[j] = reinterpret_cast<void*>(static_cast<char*>(v)+idx*p_elemsize);
        } else {
          p_env->error("(gatscatCommon) Data distribution not implemented",p_distr);
        }
        idx = list[idx];
        j++;
      }
      /* perform vector operation */
      int optype = -1;
      switch(op) {
        case XGA_GATHER:
          desc.bytes = static_cast<int64_t>(p_elemsize);
          desc.src = ptr_rem;
          desc.dst = ptr_loc;
          desc.count = nelems[iproc];
          if (req != NULL) {
            CMX::cmx_request *cmx_req = p_env->getCMXRequest(req);
            rc =  p_alloc->nbgetv(&desc,  1, iproc, cmx_req);
            if (rc) p_env->error("(gatscatCommon) gather failed",rc);
          } else {
            rc =  p_alloc->getv(&desc,  1, iproc);
            if (rc) p_env->error("(gatscatCommon) gather failed",rc);
          }
          break;
        case XGA_SCATTER:
          desc.bytes = static_cast<int64_t>(p_elemsize);
          desc.src = ptr_loc;
          desc.dst = ptr_rem;
          desc.count = nelems[iproc];
          if (req != NULL) {
            CMX::cmx_request *cmx_req = p_env->getCMXRequest(req);
            rc =  p_alloc->nbputv(&desc,  1, iproc, cmx_req);
            if (rc) p_env->error("(gatscatCommon) scatter failed",rc);
          } else {
            rc =  p_alloc->putv(&desc,  1, iproc);
            if (rc) p_env->error("(gatscatCommon) scatter failed",rc);
          }
          break;
        case XGA_SCATTERACC:
          if (alpha == NULL) {
            p_env->error("(gatscatCommon) scale factor unspecified",op);
          }
          desc.bytes = static_cast<int64_t>(p_elemsize);
          desc.src = ptr_loc;
          desc.dst = ptr_rem;
          desc.count = nelems[iproc];
          if (p_datatype == XGA_INT) optype = CMX_ACC_INT;
          else if (p_datatype == XGA_LONG) optype = CMX_ACC_LNG;
          else if (p_datatype == XGA_FLOAT) optype = CMX_ACC_FLT;
          else if (p_datatype == XGA_DOUBLE) optype = CMX_ACC_DBL;
          else if (p_datatype == XGA_COMPLEX) optype = CMX_ACC_CPL;
          else if (p_datatype == XGA_DCOMPLEX) optype = CMX_ACC_DCP;
          else {
            p_env->error("(gatscatCommon) unsupported data type",p_datatype);
          }
          if (req != NULL) {
            CMX::cmx_request *cmx_req = p_env->getCMXRequest(req);
            rc = p_alloc->nbaccv(optype, alpha, &desc,  1, iproc, cmx_req);
            if (rc) p_env->error("(gatscatCommon) scatter_acc failed",rc);
          } else {
            rc = p_alloc->accv(optype, alpha, &desc,  1, iproc);
            if (rc) p_env->error("(gatscatCommon) scatter_acc failed",rc);
          }
          break;
        default:
          p_env->error("(gatscatCommon) operation not supported",op);
      }
    }
  }

  /* clean up linked list */
  delete [] header;
  delete [] list;
  delete [] nelems;
  if (idxtype == 0) {
    delete [] subscript_ptr;
  }
}

} // XGA namespace
