/* XGA private implementation */
#include "xga_environment.hpp"
#include "xga_private.hpp"

namespace XGA {

/**
 * count number of elements in map array
 * @return sum of number of partitions in each dimension
 */
int p_GA::calc_maplen()
{
  if (p_mapc != NULL) {
    int i;
    int len = 0;
    if (p_distr != TILED_IRREG) {
      for (i=0; i<p_ndim; i++) {
        len += this->p_proc_grid[i];
      }
    } else {
      for (i=0; i<p_ndim; i++) {
        len += this->num_blks[i];
      }
    }
    return len;
  }
  return 0;
}

/**
 * compare data distribution of two arrays
 * @param[i] g_a comparision array
 * @return true if arrays have the same data distribution, false otherwise
 */
bool p_GA::compare_distr(p_GA *g_a)
{
  int h_a_maplen = g_a->calc_maplen();
  int h_b_maplen = this->calc_maplen();
  int i;
  // arrays are the same dimension
  if (this->p_ndim != g_a->p_ndim) return false;
  // arrays are the same size 
  for (i=0; i<p_ndim; i++)
    if (this->p_dims[i] != g_a->p_dims[i]) return false;
  // arrays have the same distribution model
  if (this->p_distr != g_a->p_distr) return false;

  if (this->p_distr == REGULAR || this->p_distr == TILED_IRREG) {
    // arrays have the same map layout (if applicable);
    if (h_a_maplen != h_b_maplen) return false;
    for(i=0; i <h_a_maplen; i++){
      if(this->p_mapc[i] != g_a->p_mapc[i]) return false;
      if(this->p_mapc[i] == -1) break;
    }
  } else if (this->p_distr == SCALAPACK || this->p_distr == TILED) {
    // arrays have the blocks sizes and number of blocks (if applicable);
    for (i=0; i<this->p_ndim; i++) {
      if (this->blk_dims[i] != g_a->blk_dims[i]) return false;
    }
    for (i=0; i<this->p_ndim; i++) {
      if (this->num_blks[i] != g_a->num_blks[i]) return false;
    }
  }
  // arrays have the same processor grid (if applicable);
  if (this->p_distr == SCALAPACK || this->p_distr == TILED ||
      this->p_distr == TILED_IRREG) {
    for (i=0; i<this->p_ndim; i++) {
      if (this->p_proc_grid[i] != g_a->p_proc_grid[i]) return true;
    }
  }
  return true;
}

/**
 * Compare two patches to see if they are identical
 * @param andim dimension of patch A
 * @param alo, ahi lower and upper dimensions of patch A
 * @param andim dimension of patch B
 * @param alo, ahi lower and upper dimensions of patch B
 * @return true if patches match
 */
bool p_GA::comp_patch(int andim, int64_t *alo, int64_t *ahi,
                      int bndim, int64_t *blo, int64_t *bhi)
{
  int i;
  int ndim;

  if(andim > bndim) {
    ndim = bndim;
    for(i=ndim; i<andim; i++)
      if(alo[i] != ahi[i]) return false;
  }
  else if(andim < bndim) {
    ndim = andim;
    for(i=ndim; i<bndim; i++)
      if(blo[i] != bhi[i]) return false;
  }
  else ndim = andim;

  for(i=0; i<ndim; i++)
    if((alo[i] != blo[i]) || (ahi[i] != bhi[i])) return false;

  return true;
}

/**
 * Check if two patches intersect and return the intersection
 * in second patch
 * @param lo, hi bounding indices for first block
 * @param lop, hip bounding indices for second block
 * @param ndim number of dimensions for both blocks
 */
bool p_GA::patch_intersect(int64_t *lo, int64_t *hi,
                     int64_t *lop, int64_t *hip, int ndim)
{
  int i;

  /* check consistency of patch coordinates */
  for(i=0; i<ndim; i++) {
    if(hi[i] < lo[i]) return false; /* inconsistent */
    if(hip[i] < lop[i]) return false; /* inconsistent */
  }

  /* find the intersection and update (ilop: ihip, jlop: jhip) */
  for(i=0; i<ndim; i++) {
    if(hi[i] < lop[i]) return false; /* don't intersect */
    if(hip[i] < lo[i]) return false; /* don't intersect */
  }

  for(i=0; i<ndim; i++) {
    lop[i] = XGA_MAX_M(lo[i], lop[i]);
    hip[i] = XGA_MIN_M(hi[i], hip[i]);
  }

  return true;
}

/**
 * compute index from subscript and convert it back to subscript
 * in another array
 * @param ndims number of dimensions in index of source block
 * @param los lower index of current block
 * @param blos lower index of source block
 * @param dimss array of strides for source block
 * @param ndimd number of dimensions in index of destination block
 * @param blos lower index of destination block
 * @param dimss array of strides for destination block
 */
void p_GA::dest_indices(int ndims, int64_t *los, int64_t *blos, int64_t *dimss,
               int ndimd, int64_t *lod, int64_t *blod, int64_t *dimsd)
{
  int64_t idx = 0, i, factor=1;

  for(i=ndims-1;i>=0;i--) {
    idx += (los[i] - blos[i])*factor;
    factor *= dimss[i];
  }

  for(i=ndimd-1;i>=0;i--) {
    lod[i] = idx % dimsd[i] + blod[i];
    idx /= dimsd[i];
  }
}

/**
 * Add two global arrays to get a third array. The calling array must be the
 * same size and dimension of the two arrays in the argument list, all three
 * arrays must also be the same data type. The calling array can also be the
 * same as one of the two arrays in the argument list. The parameters alpha and
 * beta can be used to scale the arrays before performing the sum.
 *   C = alpha*A + beta*B
 * If alpha or beta are NULL, assume that the are set to 1
 * @param alpha parameter to scale array A
 * @param g_a first array in sum
 * @param beta parameter to scale array B
 * @param g_b second array in sum
 */
void p_GA::add(void *alpha, p_GA *g_a, void *beta, p_GA *g_b)
{
  int  ndim, type, typeC, me;
  int64_t elems=0, elemsb=0, elemsa=0;
  int i;
  void *ptr_a, *ptr_b, *ptr_c;
  Group *a_grp, *b_grp, *c_grp;

  int64_t _dims[MAXDIM];
  int64_t _ld[MAXDIM-1];
  int64_t _lo[MAXDIM];
  int64_t _hi[MAXDIM];
  int64_t adims[MAXDIM];
  int64_t bdims[MAXDIM];
  int64_t cdims[MAXDIM];
  int andim, bndim, cndim;
 
  a_grp = g_a->p_group;
  b_grp = g_b->p_group;
  c_grp = this->p_group;
  if (a_grp != b_grp || b_grp != c_grp)
    p_env->error("(add) all three arrays must be on same group",0);
  if (this->p_datatype != g_a->p_datatype ||
      this->p_datatype != g_b->p_datatype || this->p_datatype == XGA_UNKNOWN)
    p_env->error("(add) all three arrays be the same datatype",0);

  me = a_grp->rank();
  if(!g_a->compare_distr(g_b) ||
      !this->compare_distr(g_a) || this->p_distr != REGULAR) {
    /* TODO: need to add patch algorithm */
#if 0
    /* distributions not identical */
    pnga_inquire(g_a, &type, &andim, adims);
    pnga_inquire(g_b, &type, &bndim, bdims);
    pnga_inquire(g_b, &type, &cndim, cdims);

    pnga_add_patch(alpha, g_a, one_arr, adims, beta, g_b, one_arr, bdims,
        g_c, one_arr, cdims);
#endif

    return;
  }

  this->sync();
  this->distribution(me, _lo, _hi);
  if (_lo[0]>0){
    this->accessSegmentPtr(&ptr_c, &elems);
  }

  if(g_a == this){
    ptr_a  = ptr_c;
    elemsa = elems;
  }else { 
    if (  _lo[0]>0 ){
      g_a->accessSegmentPtr(&ptr_a, &elemsa);
    }
  }

  if(g_b == this){
    ptr_b  = ptr_c;
    elemsb = elems;
  }else {
    if (  _lo[0]>0 ){
      g_b->accessSegmentPtr(&ptr_b, &elemsb);
    }
  }

  if (elems!= elemsb) p_env->error("(add) inconsistent number of elements a",
      elems-elemsb);
  if (elems!= elemsa) p_env->error("(add) inconsistent number of elements b",
      elems-elemsa);

#define SUM_M(_type, _ptr_a, _ptr_b, _ptr_c, _talpha, _tbeta, _nelems) \
{                                                                      \
  _type *_a, *_b, *_c;                                                 \
  int64_t _i;                                                          \
  _type _alpha = *reinterpret_cast<_type*>(_talpha);                   \
  _type _beta = *reinterpret_cast<_type*>(_tbeta);                     \
  _a = reinterpret_cast<_type*>(_ptr_a);                               \
  _b = reinterpret_cast<_type*>(_ptr_b);                               \
  _c = reinterpret_cast<_type*>(_ptr_c);                               \
  for (_i=0; _i<_nelems; _i++)                                         \
    _c[i] = _alpha*_a[_i]+_beta*_b[_i];                                \
}

  if (_lo[0]>0){
    /* operation on the "local" piece of data */
    switch(type){
      case XGA_DOUBLE:
        SUM_M(double, ptr_a, ptr_b, ptr_c, alpha, beta, elems);
        break;
      case XGA_DCOMPLEX:
        SUM_M(std::complex<double>, ptr_a, ptr_b, ptr_c, alpha, beta, elems);
        break;
      case XGA_COMPLEX:
        SUM_M(std::complex<float>, ptr_a, ptr_b, ptr_c, alpha, beta, elems);
        break;
      case XGA_FLOAT:
        SUM_M(float, ptr_a, ptr_b, ptr_c, alpha, beta, elems);
        break;
      case XGA_INT:
        SUM_M(int, ptr_a, ptr_b, ptr_c, alpha, beta, elems);
        break;
      case XGA_LONG:
        SUM_M(long, ptr_a, ptr_b, ptr_c, alpha, beta, elems);
        break;
      case XGA_LONGLONG:
        SUM_M(long long, ptr_a, ptr_b, ptr_c, alpha, beta, elems);
        break;
    }
#undef SUM_M

    /* release access to the data */
    this->releaseSegmentPtr();
    if(this != g_a) g_a->releaseSegmentPtr();
    if(this != g_b) g_b->releaseSegmentPtr();
  }
}

/**
 * Add patches of two global arrays to get a new patch in a third array.
 * The calling array must be the same type as the other two arrays and
 * the dimensions of the patches must be compatible. The calling array
 * can also be the same as one of the two arrays in the argument list.
 * The parameters alpha and beta can be used to scale the patches before
 * performing the sum.
 *   C = alpha*A + beta*B
 * If alpha or beta are NULL, assume that the are set to 1
 * @param alpha parameter to scale array A
 * @param g_a first array (A) in sum
 * @param alo, ahi bounding indices of patch in array A
 * @param beta parameter to scale array B
 * @param g_b second array (B) in sum
 * @param blo, bhi bounding indices of patch in array B
 * @param clo, chi bounding indices of patch in array C
 */
void p_GA::addPatch(void* alpha, p_GA *g_a,
    int64_t *alo, int64_t *ahi,
    void* beta,  p_GA *g_b, int64_t *blo, int64_t *bhi,
    int64_t *clo, int64_t *chi)
{
  int64_t i, j;
  int64_t compatible_a, compatible_b;
  int64_t atype, btype, ctype;
  int64_t adims[MAXDIM], bdims[MAXDIM], cdims[MAXDIM];
  int64_t loA[MAXDIM], hiA[MAXDIM], ldA[MAXDIM];
  int64_t loB[MAXDIM], hiB[MAXDIM], ldB[MAXDIM];
  int64_t loC[MAXDIM], hiC[MAXDIM], ldC[MAXDIM];
  void *A_ptr, *B_ptr, *C_ptr;
  int64_t n1dim;
  int64_t atotal, btotal;
  p_GA *g_A = g_a, *g_B = g_b;
  int me = p_group->rank();
  bool A_created=false, B_created=false;
  int nproc = p_group->size();
  int64_t num_blocks_a, num_blocks_b, num_blocks_c;
  char notrans='n';

  atype = g_a->p_datatype;
  btype = g_b->p_datatype;
  ctype = this->p_datatype;
  if(ctype != atype || ctype != btype )
    p_env->error("(addPatch) datatypes mismatch ", 0); 

  int andim = g_a->p_ndim;
  int bndim = g_b->p_ndim;
  int cndim = this->p_ndim;
  /* check if patch indices and dims match */
  for(i=0; i<andim; i++)
    if(alo[i] <= 0 || ahi[i] > adims[i])
      p_env->error("(addPatch) g_a indices out of range ", 0);
  for(i=0; i<bndim; i++)
    if(blo[i] <= 0 || bhi[i] > bdims[i])
      p_env->error("(addPatch) g_b indices out of range ", 0);
  for(i=0; i<cndim; i++)
    if(clo[i] <= 0 || chi[i] > cdims[i])
      p_env->error("(addPatch) result indices out of range ", 0);

  /* check if numbers of elements in patches match each other */
  n1dim = 1; for(i=0; i<cndim; i++) n1dim *= (chi[i] - clo[i] + 1);
  atotal = 1; for(i=0; i<andim; i++) atotal *= (ahi[i] - alo[i] + 1);
  btotal = 1; for(i=0; i<bndim; i++) btotal *= (bhi[i] - blo[i] + 1);

  if((atotal != n1dim) || (btotal != n1dim))
    p_env->error("(addPatch) capacities of patches do not match ", 0);

  num_blocks_a = g_a->block_total;
  num_blocks_b = g_b->block_total;
  num_blocks_c = this->block_total;

  if (num_blocks_a < 0 && num_blocks_b < 0 && num_blocks_c < 0) {
    /* find out coordinates of patches of g_a, g_b and g_c that I own */
    g_a->distribution(me, loA, hiA);
    g_b->distribution(me, loB, hiB);
    this->distribution(me, loC, hiC);

    /* test if the local portion of patches matches */
    if(comp_patch(andim, loA, hiA, cndim, loC, hiC) &&
       comp_patch(andim, alo, ahi, cndim, clo, chi)) compatible_a = 1;
    else compatible_a = 0;
    p_group->prod<int64_t>(&compatible_a, 1);
    if(comp_patch(bndim, loB, hiB, cndim, loC, hiC) &&
       comp_patch(bndim, blo, bhi, cndim, clo, chi)) compatible_b = 1;
    else compatible_b = 0;
    /* pnga_gop(pnga_type_f2c(MT_F_INT), &compatible_b, 1, "*"); */
    p_group->prod<int64_t>(&compatible_b, 1);
    if (compatible_a && compatible_b) {
      if(andim > bndim) cndim = bndim;
      if(andim < bndim) cndim = andim;

      if(!comp_patch(andim, loA, hiA, cndim, loC, hiC))
        p_env->error(" A patch mismatch (g_a)",0);
      if(!comp_patch(bndim, loB, hiB, cndim, loC, hiC))
        p_env->error(" A patch mismatch (g_b)",0);

      /*  determine subsets of my patches to access  */
      if (patch_intersect(clo, chi, loC, hiC, cndim)){
        g_a->accessPtr(loC, hiC, &A_ptr, ldA);
        g_b->accessPtr(loC, hiC, &B_ptr, ldB);
        this->accessPtr(loC, hiC, &C_ptr, ldC);

        switch(ctype) {
          case XGA_DOUBLE:
            add_patch_values<double>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_DCOMPLEX:
            add_patch_values<std::complex<double> >(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_COMPLEX:
            add_patch_values<std::complex<float> >(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_INT:
            add_patch_values<int>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_FLOAT:
            add_patch_values<float>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_LONG:
            add_patch_values<long>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_LONGLONG:
            add_patch_values<long long>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          default:
            p_env->error("(addPatch) unknown data type",ctype);
        }

        /* release access to the data */
        g_a->releasePtr(loC, hiC);
        g_b->releasePtr(loC, hiC); 
        this->releaseUpdatePtr(loC, hiC); 
      }
    } else if (!compatible_a && compatible_b) {
      /* either patches or distributions do not match:
       *  - create a temp array that matches distribution of g_c
       *  - do C<= A
       */
      if(g_b != this) {
        this->copyPatch(notrans, alo, ahi, g_B, clo, chi);
        andim = cndim;
        g_A = this;
        g_A->distribution(me, loA, hiA);
      } else {
        g_A = this->duplicate();
        g_a->copyPatch(notrans, alo, ahi, g_A, clo, chi);
        andim = cndim;
        A_created = 1;
        g_A->distribution(me, loA, hiA);
      }
      if(andim > bndim) cndim = bndim;
      if(andim < bndim) cndim = andim;

      if(!comp_patch(andim, loA, hiA, cndim, loC, hiC))
        p_env->error(" A patch mismatch ", 0); 
      if(!comp_patch(bndim, loB, hiB, cndim, loC, hiC))
        p_env->error(" B patch mismatch ", 0);

      /*  determine subsets of my patches to access  */
      if (patch_intersect(clo, chi, loC, hiC, cndim)){
        g_A->accessPtr(loC, hiC, &A_ptr, ldA);
        g_B->accessPtr(loC, hiC, &B_ptr, ldB);
        this->accessPtr(loC, hiC, &C_ptr, ldC);

        switch(ctype) {
          case XGA_DOUBLE:
            add_patch_values<double>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_DCOMPLEX:
            add_patch_values<std::complex<double> >(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_COMPLEX:
            add_patch_values<std::complex<float> >(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_INT:
            add_patch_values<int>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_FLOAT:
            add_patch_values<float>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_LONG:
            add_patch_values<long>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_LONGLONG:
            add_patch_values<long long>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          default:
            p_env->error("(addPatch) unknown data type",ctype);
        }

        /* release access to the data */
        g_A->releasePtr(loC, hiC);
        g_B->releasePtr(loC, hiC); 
        this->releaseUpdatePtr(loC, hiC); 
      }
    } else if (compatible_a && !compatible_b) {
      /* either patches or distributions do not match:
       *        - create a temp array that matches distribution of g_c
       *        - copy & reshape patch of g_b into g_B
       */
      g_B = this->duplicate();
      g_b->copyPatch(notrans, blo, bhi, g_B, clo, chi);
      bndim = cndim;
      B_created = 1;
      g_B->distribution(me, loB, hiB);

      if(andim > bndim) cndim = bndim;
      if(andim < bndim) cndim = andim;

      if(!comp_patch(andim, loA, hiA, cndim, loC, hiC))
        p_env->error(" A patch mismatch ", 0); 
      if(!comp_patch(bndim, loB, hiB, cndim, loC, hiC))
        p_env->error(" B patch mismatch ", 0);

      /*  determine subsets of my patches to access  */
      if (patch_intersect(clo, chi, loC, hiC, cndim)){
        g_A->accessPtr(loC, hiC, &A_ptr, ldA);
        g_B->accessPtr(loC, hiC, &B_ptr, ldB);
        this->accessPtr(loC, hiC, &C_ptr, ldC);

        switch(ctype) {
          case XGA_DOUBLE:
            add_patch_values<double>(alpha, beta,
                loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_DCOMPLEX:
            add_patch_values<std::complex<double> >(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_COMPLEX:
            add_patch_values<std::complex<float> >(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_INT:
            add_patch_values<int>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_FLOAT:
            add_patch_values<float>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_LONG:
            add_patch_values<long>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          case XGA_LONGLONG:
            add_patch_values<long long>(alpha, beta,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
            break;
          default:
            p_env->error("(addPatch) unknown data type",ctype);
        }

        /* release access to the data */
        g_A->releasePtr(loC, hiC);
        g_B->releasePtr(loC, hiC); 
        this->releaseUpdatePtr(loC, hiC); 
      }
    } else if (!compatible_a && !compatible_b) {
      /* there is no match between any of the global arrays */
      g_B = this->duplicate();
      g_b->copyPatch(notrans, blo, bhi, g_B, clo, chi);
      bndim = cndim;
      B_created = 1;
      if(andim > bndim) cndim = bndim;
      if(andim < bndim) cndim = andim;
      cndim = bndim;
      g_a->copyPatch(notrans, alo, ahi, this, clo, chi);
      this->scalePatch(clo, chi, alpha);
      /*  determine subsets of my patches to access  */
      if (patch_intersect(clo, chi, loC, hiC, cndim)){
        g_B->accessPtr(loC, hiC, &B_ptr, ldB);
        this->accessPtr(loC, hiC, &C_ptr, ldC);

        switch(ctype) {
          case XGA_DOUBLE:
            acc_patch_values<double>(beta, loC, hiC, ldC, A_ptr, C_ptr);
            break;
          case XGA_DCOMPLEX:
            acc_patch_values<std::complex<double> >(beta, loC, hiC, ldC,
                A_ptr, C_ptr);
            break;
          case XGA_COMPLEX:
            acc_patch_values<std::complex<float> >(beta, loC, hiC, ldC,
                A_ptr, C_ptr);
            break;
          case XGA_INT:
            acc_patch_values<int>(beta, loC, hiC, ldC, A_ptr, C_ptr);
            break;
          case XGA_FLOAT:
            acc_patch_values<float>(beta, loC, hiC, ldC, A_ptr, C_ptr);
            break;
          case XGA_LONG:
            acc_patch_values<long>(beta, loC, hiC, ldC, A_ptr, C_ptr);
            break;
          case XGA_LONGLONG:
            acc_patch_values<long long>(beta, loC, hiC, ldC, A_ptr, C_ptr);
            break;
          default:
            p_env->error("(addPatch) unknown data type",ctype);
        }
        /* release access to the data */
        g_B->releasePtr(loC, hiC); 
        this->releaseUpdatePtr(loC, hiC); 
      }
    }
  } else {
    char *aptr, *bptr, *cptr;
    /* create copies of arrays A and B that are identically distributed
       as C*/
    g_A = this->duplicate();
    g_a->copyPatch(notrans, alo, ahi, g_A, clo, chi);
    andim = cndim;
    A_created = 1;

    g_B = this->duplicate();
    g_b->copyPatch(notrans, blo, bhi, g_B, clo, chi);
    bndim = cndim;
    B_created = 1;

    g_A->localInit();
    g_B->localInit();
    this->localInit();
    while (this->nextLocalBlock(loC,hiC,&cptr,ldC)) {
      g_A->nextLocalBlock(loA,hiA,&aptr,ldA);
      g_B->nextLocalBlock(loB,hiB,&bptr,ldB);
      A_ptr = reinterpret_cast<void*>(aptr);
      B_ptr = reinterpret_cast<void*>(bptr);
      C_ptr = reinterpret_cast<void*>(cptr);
      int64_t idx, lod[MAXDIM]/*, hid[MAXDIM]*/;
      int64_t offset, jtot, last;
      /* make temporary copies of loC and hiC since pnga_patch_intersect
         destroys original versions */
      for (j=0; j<cndim; j++) {
        lod[j] = loC[j];
      }
      if (patch_intersect(clo, chi, loC, hiC, cndim)) {

        /* evaluate offsets for system */
        offset = 0;
        last = cndim - 1;
        jtot = 1;
        for (j=last; j>=0; j--) {
          offset += (loC[j] - lod[j])*jtot;
          if (j>0) jtot *= ldC[j];
        }

#define ASSIGN_ABC_M(_ltype, _aptr, _bptr, _cptr, _offset)           \
{                                                                    \
  _aptr = reinterpret_cast<char*>(reinterpret_cast<_ltype*>(_aptr)   \
      + _offset);                                                    \
  _bptr = reinterpret_cast<char*>(reinterpret_cast<_ltype*>(_bptr)   \
      + _offset);                                                    \
  _cptr = reinterpret_cast<char*>(reinterpret_cast<_ltype*>(_cptr)   \
      + _offset);                                                    \
  add_patch_values<_ltype>(alpha, beta, loC, hiC, ldC,               \
      A_ptr, B_ptr, C_ptr);                                          \
}
        switch(ctype) {
          case XGA_DOUBLE:
            ASSIGN_ABC_M(double,A_ptr,B_ptr,C_ptr,offset);
            break;
          case XGA_INT:
            ASSIGN_ABC_M(int,A_ptr,B_ptr,C_ptr,offset);
            break;
          case XGA_DCOMPLEX:
            ASSIGN_ABC_M(std::complex<double>,A_ptr,B_ptr,C_ptr,offset);
            break;
          case XGA_COMPLEX:
            ASSIGN_ABC_M(std::complex<float>,A_ptr,B_ptr,C_ptr,offset);
            break;
          case XGA_FLOAT:
            ASSIGN_ABC_M(float,A_ptr,B_ptr,C_ptr,offset);
            break;
          case XGA_LONG:
            ASSIGN_ABC_M(long,A_ptr,B_ptr,C_ptr,offset);
            break;
          case XGA_LONGLONG:
            ASSIGN_ABC_M(long long,A_ptr,B_ptr,C_ptr,offset);
            break;
          default:
            break;
        }
#undef ASSIGN_ABC_M
//        add_patch_values(alpha, beta, cndim,
//            loC, hiC, ldC, A_ptr, B_ptr, C_ptr);
      }
    }
  }

  if(A_created) delete g_A;
  if(B_created) delete g_B;
}

} // XGA namespace
