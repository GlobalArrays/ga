/* 
 * module: global.npatch.c
 * author: Jialin Ju
 * description: Implements the n-dimensional patch operations:
 *              - fill patch
 *              - copy patch
 *              - scale patch
 *              - dot patch
 *              - add patch
 * 
 * DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */

#include "message.h"
#include "global.h"
#include "globalp.h"
#include "armci.h"
#include <math.h>

#ifdef GA_USE_VAMPIR
#include "ga_vampir.h"
#endif

#ifdef CRAY
#      include <fortran.h>
#endif

#if defined(CRAY) || defined(WIN32)
#   define cptofcd(fcd)  _cptofcd((fcd),1)
#else
#      define cptofcd(fcd) (fcd)
#endif


/**********************************************************
 *  n-dimensional utilities                               *
 **********************************************************/

/*\ compute Index from subscript and convert it back to subscript
 *  in another array
\*/
void ngai_dest_indices(Integer ndims, Integer *los, Integer *blos, Integer *dimss,
               Integer ndimd, Integer *lod, Integer *blod, Integer *dimsd)
{
    Integer idx = 0, i, factor=1;
        
    for(i=0;i<ndims;i++) {
        idx += (los[i] - blos[i])*factor;
        factor *= dimss[i];
    }
        
    for(i=0;i<ndims;i++) {
        lod[i] = idx % dimsd[i] + blod[i];
        idx /= dimsd[i];
    }
}


/* check if I own data in the patch */
logical ngai_patch_intersect(Integer *lo, Integer *hi,
                        Integer *lop, Integer *hip, Integer ndim)
{
    Integer i;
    
    /* check consistency of patch coordinates */
    for(i=0; i<ndim; i++) {
        if(hi[i] < lo[i]) return FALSE; /* inconsistent */
        if(hip[i] < lop[i]) return FALSE; /* inconsistent */
    }
    
    /* find the intersection and update (ilop: ihip, jlop: jhip) */
    for(i=0; i<ndim; i++) {
        if(hi[i] < lop[i]) return FALSE; /* don't intersect */
        if(hip[i] < lo[i]) return FALSE; /* don't intersect */
    }
    
    for(i=0; i<ndim; i++) {
        lop[i] = MAX(lo[i], lop[i]);
        hip[i] = MIN(hi[i], hip[i]);
    }
    
    return TRUE;
}

/*\ check if patches are identical 
\*/
logical ngai_comp_patch(Integer andim, Integer *alo, Integer *ahi,
                          Integer bndim, Integer *blo, Integer *bhi)
{
    Integer i;
    Integer ndim;
    
    if(andim > bndim) {
        ndim = bndim;
        for(i=ndim; i<andim; i++)
            if(alo[i] != ahi[i]) return FALSE;
    }
    else if(andim < bndim) {
        ndim = andim;
        for(i=ndim; i<bndim; i++)
            if(blo[i] != bhi[i]) return FALSE;
    }
    else ndim = andim;
    
    for(i=0; i<ndim; i++)
        if((alo[i] != blo[i]) || (ahi[i] != bhi[i])) return FALSE;

    return TRUE; 
}

/* test two GAs to see if they have the same shape */
logical ngai_test_shape(Integer *alo, Integer *ahi, Integer *blo,
                          Integer *bhi, Integer andim, Integer bndim)
{
    Integer i;

    if(andim != bndim) return FALSE;
    
    for(i=0; i<andim; i++) 
        if((ahi[i] - alo[i]) != (bhi[i] - blo[i])) return FALSE;
        
    return TRUE;
}

/**********************************************************
 *  n-dimensional functions                               *
 **********************************************************/

/*\ COPY A PATCH AND POSSIBLY RESHAPE
 *
 *  . the element capacities of two patches must be identical
 *  . copy by column order - Fortran convention
\*/
void nga_copy_patch(char *trans,
                    Integer *g_a, Integer *alo, Integer *ahi,
                    Integer *g_b, Integer *blo, Integer *bhi)
{
  Integer i, j;
  Integer idx, factor;
  Integer atype, btype, andim, adims[MAXDIM], bndim, bdims[MAXDIM];
  Integer nelem;
  Integer atotal, btotal;
  Integer los[MAXDIM], his[MAXDIM];
  Integer lod[MAXDIM], hid[MAXDIM];
  Integer ld[MAXDIM], ald[MAXDIM], bld[MAXDIM];
  void *src_data_ptr, *tmp_ptr;
  Integer *src_idx_ptr, *dst_idx_ptr;
  Integer bvalue[MAXDIM], bunit[MAXDIM];
  Integer factor_idx1[MAXDIM], factor_idx2[MAXDIM], factor_data[MAXDIM];
  Integer base;
  Integer me_a, me_b;
  Integer a_grp, b_grp, anproc, bnproc;
  Integer num_blocks_a, num_blocks_b, chk;
  int use_put, has_intersection;
  int local_sync_begin,local_sync_end;

#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_COPY_PATCH,__FILE__,__LINE__);
#endif    

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  a_grp = ga_get_pgroup_(g_a);
  b_grp = ga_get_pgroup_(g_b);
  me_a = ga_pgroup_nodeid_(&a_grp);
  me_b = ga_pgroup_nodeid_(&b_grp);
  anproc = ga_get_pgroup_size_(&a_grp);
  bnproc = ga_get_pgroup_size_(&b_grp);
  if (anproc <= bnproc) {
    use_put = 1;
  }  else {
    use_put = 0;
  }

  /*if (a_grp != b_grp)
    ga_error("All matrices must be on same group for nga_copy_patch", 0L); */
  if(local_sync_begin) {
    if (anproc <= bnproc) {
      ga_pgroup_sync_(&a_grp);
    } else if (a_grp == ga_pgroup_get_world_() &&
        b_grp == ga_pgroup_get_world_()) {
      ga_sync_();
    } else {
      ga_pgroup_sync_(&b_grp);
    }
  }


  GA_PUSH_NAME("nga_copy_patch");

  nga_inquire_internal_(g_a, &atype, &andim, adims);
  nga_inquire_internal_(g_b, &btype, &bndim, bdims);

  if(*g_a == *g_b)
    /* they are the same patch */
    if(ngai_comp_patch(andim, alo, ahi, bndim, blo, bhi)) return;
  /* they are in the same GA, but not the same patch */
    else if (ngai_patch_intersect(alo, ahi, blo, bhi, andim))
      ga_error("array patches cannot overlap ", 0L);

  if(atype != btype ) ga_error("array type mismatch ", 0L);

  /* check if patch indices and dims match */
  for(i=0; i<andim; i++)
    if(alo[i] <= 0 || ahi[i] > adims[i])
      ga_error("g_a indices out of range ", 0L);
  for(i=0; i<bndim; i++)
    if(blo[i] <= 0 || bhi[i] > bdims[i])
      ga_error("g_b indices out of range ", 0L);



  /* check if numbers of elements in two patches match each other */
  atotal = 1; btotal = 1;
  for(i=0; i<andim; i++) atotal *= (ahi[i] - alo[i] + 1);
  for(i=0; i<bndim; i++) btotal *= (bhi[i] - blo[i] + 1);
  if(atotal != btotal)
    ga_error("capacities two of patches do not match ", 0L);

  /* additional restrictions that apply if one or both arrays use
     block-cyclic data distributions */
  num_blocks_a = ga_total_blocks_(g_a);
  num_blocks_b = ga_total_blocks_(g_b);
  if (num_blocks_a >= 0 || num_blocks_b >= 0) {
    if (!(*trans == 'n' || *trans == 'N')) {
      ga_error("Transpose option not supported for block-cyclic data", 0L);
    }
    if (!ngai_test_shape(alo, ahi, blo, bhi, andim, bndim)) {
      ga_error("Change in shape not supported for block-cyclic data", 0L);
    }
  }

  if (num_blocks_a < 0 && num_blocks_b <0) {
    /* now find out cordinates of a patch of g_a that I own */
    if (use_put) {
      nga_distribution_(g_a, &me_a, los, his);
    } else {
      nga_distribution_(g_b, &me_b, los, his);
    }

    /* copy my share of data */
    if (use_put) {
      has_intersection = ngai_patch_intersect(alo, ahi, los, his, andim);
    } else {
      has_intersection = ngai_patch_intersect(blo, bhi, los, his, bndim);
    }
    if(has_intersection){
      if (use_put) {
        nga_access_ptr(g_a, los, his, &src_data_ptr, ld); 
      } else {
        nga_access_ptr(g_b, los, his, &src_data_ptr, ld); 
      }

      /* calculate the number of elements in the patch that I own */
      nelem = 1; for(i=0; i<andim; i++) nelem *= (his[i] - los[i] + 1);

      for(i=0; i<andim; i++) ald[i] = ahi[i] - alo[i] + 1;
      for(i=0; i<bndim; i++) bld[i] = bhi[i] - blo[i] + 1;

      base = 0; factor = 1;
      for(i=0; i<andim; i++) {
        base += los[i] * factor;
        factor *= ld[i];
      }

      /*** straight copy possible if there's no reshaping or transpose ***/
      if((*trans == 'n' || *trans == 'N') &&
          ngai_test_shape(alo, ahi, blo, bhi, andim, bndim)) { 
        /* find source[lo:hi] --> destination[lo:hi] */
        if (use_put) {
          ngai_dest_indices(andim, los, alo, ald, bndim, lod, blo, bld);
          ngai_dest_indices(andim, his, alo, ald, bndim, hid, blo, bld);
          nga_put_(g_b, lod, hid, src_data_ptr, ld);
          nga_release_(g_a, los, his);
        } else {
          ngai_dest_indices(bndim, los, blo, bld, andim, lod, alo, ald);
          ngai_dest_indices(bndim, his, blo, bld, andim, hid, alo, ald);
          nga_get_(g_a, lod, hid, src_data_ptr, ld);
          nga_release_(g_b, los, his);
        }
        /*** due to generality of this transformation scatter is required ***/
      } else{
        if (use_put) {
          tmp_ptr = ga_malloc(nelem, atype, "v");
          src_idx_ptr = (Integer*) ga_malloc((andim*nelem), MT_F_INT, "si");
          dst_idx_ptr = (Integer*) ga_malloc((bndim*nelem), MT_F_INT, "di");

          /* calculate the destination indices */

          /* given los and his, find indices for each elements
           * bvalue: starting index in each dimension
           * bunit: stride in each dimension
           */
          for (i=0; i<andim; i++) {
            bvalue[i] = los[i];
            if (i == 0) bunit[i] = 1;
            else bunit[i] = bunit[i-1] * (his[i-1] - los[i-1] + 1);
          }

          /* source indices */
          for (i=0; i<nelem; i++) {
            for (j=0; j<andim; j++){
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
          factor_idx1[0] = 1;
          for (j=1; j<andim; j++) 
            factor_idx1[j] = factor_idx1[j-1] * ald[j-1];

          /* index factor: reshaping with transpose */
          factor_idx2[andim-1] = 1;
          for (j=(andim-1)-1; j>=0; j--)
            factor_idx2[j] = factor_idx2[j+1] * ald[j+1];

          /* data factor */
          factor_data[0] = 1;
          for (j=1; j<andim; j++) 
            factor_data[j] = factor_data[j-1] * ld[j-1];

          /* destination indices */
          for(i=0; i<nelem; i++) {
            /* linearize the n-dimensional indices to one dimension */
            idx = 0;
            if (*trans == 'n' || *trans == 'N')
              for (j=0; j<andim; j++) 
                idx += (src_idx_ptr[i*andim+j] - alo[j]) *
                  factor_idx1[j];
            else
              /* if the patch needs to be transposed, reverse
               * the indices: (i, j, ...) -> (..., j, i)
               */
              for (j=(andim-1); j>=0; j--) 
                idx += (src_idx_ptr[i*andim+j] - alo[j]) *
                  factor_idx2[j];

            /* convert the one dimensional index to n-dimensional
             * indices of destination
             */
            for (j=0; j<bndim; j++) {
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
            switch(atype) {
              case C_DBL: ((double*)tmp_ptr)[i] =
                          ((double*)src_data_ptr)[idx]; 
                          break;
              case C_INT:
                          ((int *)tmp_ptr)[i] = ((int *)src_data_ptr)[idx];
                          break;
              case C_DCPL:((DoubleComplex *)tmp_ptr)[i] =
                          ((DoubleComplex *)src_data_ptr)[idx];
                          break;
              case C_FLOAT: ((float *)tmp_ptr)[i] =
                            ((float *)src_data_ptr)[idx]; 
                            break;     
              case C_LONG: ((long *)tmp_ptr)[i] =
                           ((long *)src_data_ptr)[idx];     
            }
          }
          nga_release_(g_a, los, his);
          nga_scatter_(g_b, tmp_ptr, dst_idx_ptr, &nelem);
          ga_free(dst_idx_ptr);
          ga_free(src_idx_ptr);
          ga_free(tmp_ptr);
        } else {
          tmp_ptr = ga_malloc(nelem, atype, "v");
          src_idx_ptr = (Integer*) ga_malloc((bndim*nelem), MT_F_INT, "si");
          dst_idx_ptr = (Integer*) ga_malloc((andim*nelem), MT_F_INT, "di");

          /* calculate the destination indices */

          /* given los and his, find indices for each elements
           * bvalue: starting index in each dimension
           * bunit: stride in each dimension
           */
          for (i=0; i<andim; i++) {
            bvalue[i] = los[i];
            if (i == 0) bunit[i] = 1;
            else bunit[i] = bunit[i-1] * (his[i-1] - los[i-1] + 1);
          }

          /* destination indices */
          for (i=0; i<nelem; i++) {
            for (j=0; j<bndim; j++){
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
          factor_idx1[0] = 1;
          for (j=1; j<bndim; j++) 
            factor_idx1[j] = factor_idx1[j-1] * bld[j-1];

          /* index factor: reshaping with transpose */
          factor_idx2[bndim-1] = 1;
          for (j=(bndim-1)-1; j>=0; j--)
            factor_idx2[j] = factor_idx2[j+1] * bld[j+1];

          /* data factor */
          factor_data[0] = 1;
          for (j=1; j<bndim; j++) 
            factor_data[j] = factor_data[j-1] * ld[j-1];

          /* destination indices */
          for(i=0; i<nelem; i++) {
            /* linearize the n-dimensional indices to one dimension */
            idx = 0;
            if (*trans == 'n' || *trans == 'N')
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
            for (j=0; j<andim; j++) {
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
            switch(atype) {
              case C_DBL: ((double*)tmp_ptr)[i] =
                          ((double*)src_data_ptr)[idx]; 
                          break;
              case C_INT:
                          ((int *)tmp_ptr)[i] = ((int *)src_data_ptr)[idx];
                          break;
              case C_DCPL:((DoubleComplex *)tmp_ptr)[i] =
                          ((DoubleComplex *)src_data_ptr)[idx];
                          break;
              case C_FLOAT: ((float *)tmp_ptr)[i] =
                            ((float *)src_data_ptr)[idx]; 
                            break;     
              case C_LONG: ((long *)tmp_ptr)[i] =
                           ((long *)src_data_ptr)[idx];     
            }
          }
          nga_release_(g_b, los, his);
          nga_gather_(g_a, tmp_ptr, dst_idx_ptr, &nelem);
          ga_free(dst_idx_ptr);
          ga_free(src_idx_ptr);
          ga_free(tmp_ptr);
        }
      }
    }
  } else {
    Integer offset, last, jtot;
    for (i=0; i<andim; i++) {
      ald[i] = ahi[i] - alo[i] + 1;
    }
    for (i=0; i<bndim; i++) {
      bld[i] = bhi[i] - blo[i] + 1;
    }
    if (use_put) {
      /* Array a is block-cyclic distributed */
      if (num_blocks_a >= 0) {
        for (i = me_a; i < num_blocks_a; i += anproc) {
          nga_distribution_(g_a, &i, los, his); 
          /* make temporory copies of los, his since ngai_patch_intersection
             destroys original versions */
          for (j=0; j < andim; j++) {
            lod[j] = los[j];
            hid[j] = his[j];
          }
          if (ngai_patch_intersect(alo,ahi,los,his,andim)) {
            nga_access_block_ptr(g_a, &i, &src_data_ptr, ld);
            offset = 0;
            last = andim - 1;
            jtot = 1;
            for (j=0; j<last; j++) {
              offset += (los[j]-lod[j])*jtot;
              jtot = ld[j];
            }
            offset += (los[last]-lod[last])*jtot;
            switch(atype) {
              case C_DBL:
                src_data_ptr = (void*)((double*)(src_data_ptr) + offset); 
                break;
              case C_INT:
                src_data_ptr = (void*)((int*)(src_data_ptr) + offset); 
                break;
              case C_DCPL:
                src_data_ptr = (void*)((DoubleComplex*)(src_data_ptr) + offset); 
                break;
              case C_FLOAT:
                src_data_ptr = (void*)((float*)(src_data_ptr) + offset); 
                break;     
              case C_LONG:
                src_data_ptr = (void*)((long*)(src_data_ptr) + offset); 
                break;
              default:
                break;
            }
            ngai_dest_indices(andim, los, alo, ald, bndim, lod, blo, bld);
            ngai_dest_indices(andim, his, alo, ald, bndim, hid, blo, bld);
            nga_put_(g_b, lod, hid, src_data_ptr, ld);
            nga_release_block_(g_a, &i);
          }
        }
      } else {
        /* Array b is block-cyclic distributed */
        nga_distribution_(g_a, &me_a, los, his); 
        if (ngai_patch_intersect(alo,ahi,los,his,andim)) {
          nga_access_ptr(g_a, los, his, &src_data_ptr, ld); 
          ngai_dest_indices(andim, los, alo, ald, bndim, lod, blo, bld);
          ngai_dest_indices(andim, his, alo, ald, bndim, hid, blo, bld);
          nga_put_(g_b, lod, hid, src_data_ptr, ld);
          nga_release_(g_a, los, his);
        }
      }
    } else {
      /* Array b is block-cyclic distributed */
      if (num_blocks_b >= 0) {
        for (i = me_b; i < num_blocks_b; i += bnproc) {
          nga_distribution_(g_b, &i, los, his); 
          /* make temporory copies of los, his since ngai_patch_intersection
             destroys original versions */
          for (j=0; j < andim; j++) {
            lod[j] = los[j];
            hid[j] = his[j];
          }
          if (ngai_patch_intersect(blo,bhi,los,his,andim)) {
            nga_access_block_ptr(g_b, &i, &src_data_ptr, ld);
            offset = 0;
            last = bndim - 1;
            jtot = 1;
            for (j=0; j<last; j++) {
              offset += (los[j]-lod[j])*jtot;
              jtot = ld[j];
            }
            offset += (los[last]-lod[last])*jtot;
            switch(atype) {
              case C_DBL:
                src_data_ptr = (void*)((double*)(src_data_ptr) + offset); 
                break;
              case C_INT:
                src_data_ptr = (void*)((int*)(src_data_ptr) + offset); 
                break;
              case C_DCPL:
                src_data_ptr = (void*)((DoubleComplex*)(src_data_ptr) + offset); 
                break;
              case C_FLOAT:
                src_data_ptr = (void*)((float*)(src_data_ptr) + offset); 
                break;     
              case C_LONG:
                src_data_ptr = (void*)((long*)(src_data_ptr) + offset); 
                break;
              default:
                break;
            }
            ngai_dest_indices(bndim, los, blo, bld, andim, lod, alo, ald);
            ngai_dest_indices(bndim, his, blo, bld, andim, hid, alo, ald);
            nga_get_(g_a, lod, hid, src_data_ptr, ld);
            nga_release_block_(g_b, &i);
          }
        }
      } else {
        /* Array a is block-cyclic distributed */
        nga_distribution_(g_b, &me_b, los, his); 
        if (ngai_patch_intersect(blo,bhi,los,his,bndim)) {
          nga_access_ptr(g_b, los, his, &src_data_ptr, ld); 
          ngai_dest_indices(bndim, los, blo, bld, andim, lod, alo, ald);
          ngai_dest_indices(bndim, his, blo, bld, andim, hid, alo, ald);
          nga_get_(g_a, lod, hid, src_data_ptr, ld);
          nga_release_(g_b, los, his);
        }
      }
    }
  }
  GA_POP_NAME;
  if(local_sync_end) {
    if (anproc <= bnproc) {
      ga_pgroup_sync_(&a_grp);
    } else if (a_grp == ga_pgroup_get_world_() &&
        b_grp == ga_pgroup_get_world_()) {
      ga_sync_();
    } else {
      ga_pgroup_sync_(&b_grp);
    }
  }
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_COPY_PATCH,__FILE__,__LINE__);
#endif    
}

/*\ COPY A PATCH AND POSSIBLY RESHAPE
 *  Fortran interface
\*/
void FATR nga_copy_patch_(trans, g_a, alo, ahi, g_b, blo, bhi)
     Integer *g_a, *alo, *ahi;
     Integer *g_b, *blo, *bhi;
#if defined(CRAY) || defined(WIN32)
     _fcd    trans;
{nga_copy_patch(_fcdtocp(trans),g_a,alo,ahi,g_b,blo,bhi);}
#else 
     char*   trans;
{  nga_copy_patch(trans,g_a,alo,ahi,g_b,blo,bhi); }
#endif

void ngai_dot_local_patch(Integer atype, Integer andim, Integer *loA,
                          Integer *hiA, Integer *ldA, void *A_ptr, void *B_ptr,
                          int *alen, void *retval)
{
  int isum;
  double dsum;
  DoubleComplex zsum;
  float fsum;
  long lsum;
  Integer i, j, n1dim, idx;
  Integer bvalue[MAXDIM], bunit[MAXDIM], baseldA[MAXDIM];

  isum = 0; dsum = 0.; zsum.real = 0.; zsum.imag = 0.; fsum = 0;lsum=0;

  /* number of n-element of the first dimension */
  n1dim = 1; for(i=1; i<andim; i++) n1dim *= (hiA[i] - loA[i] + 1);

  /* calculate the destination indices */
  bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
  /* baseldA[0] = ldA[0]
   * baseldA[1] = ldA[0] * ldA[1]
   * baseldA[2] = ldA[0] * ldA[1] * ldA[2] .....
   */
  baseldA[0] = ldA[0]; baseldA[1] = baseldA[0] *ldA[1];
  for(i=2; i<andim; i++) {
    bvalue[i] = 0;
    bunit[i] = bunit[i-1] * (hiA[i-1] - loA[i-1] + 1);
    baseldA[i] = baseldA[i-1] * ldA[i];
  }

  /* compute "local" contribution to the dot product */
  switch (atype){
    case C_INT:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<andim; j++) {
          idx += bvalue[j] * baseldA[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          isum += ((int *)A_ptr)[idx+j] *
            ((int *)B_ptr)[idx+j];
      }
      *(int*)retval += isum;
      break;
    case C_DCPL:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<andim; j++) {
          idx += bvalue[j] * baseldA[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++) {
          DoubleComplex a = ((DoubleComplex *)A_ptr)[idx+j];
          DoubleComplex b = ((DoubleComplex *)B_ptr)[idx+j];
          zsum.real += a.real*b.real  - b.imag * a.imag;
          zsum.imag += a.imag*b.real  + b.imag * a.real;
        }
      }
      ((double*)retval)[0] += zsum.real;
      ((double*)retval)[1] += zsum.imag;
      break;
    case  C_DBL:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<andim; j++) {
          idx += bvalue[j] * baseldA[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          dsum += ((double*)A_ptr)[idx+j] *
            ((double*)B_ptr)[idx+j];
      }
      *(double*)retval += dsum;
      break;
    case C_FLOAT:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<andim; j++) {
          idx += bvalue[j] * baseldA[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          fsum += ((float *)A_ptr)[idx+j] *
            ((float *)B_ptr)[idx+j];
      }
      *(float*)retval += fsum;
      break;         
    case C_LONG:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<andim; j++) {
          idx += bvalue[j] * baseldA[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          lsum += ((long *)A_ptr)[idx+j] *
            ((long *)B_ptr)[idx+j];
      }
      *(long*)retval += lsum;
      break;                                     
  }
}

/*\ generic dot product routine
\*/
void ngai_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, retval)
     Integer *g_a, *alo, *ahi;    /* patch of g_a */
     Integer *g_b, *blo, *bhi;    /* patch of g_b */
     char    *t_a, *t_b;          /* transpose operators */
     void *retval;
{
  Integer i, j;
  Integer compatible;
  Integer atype, btype, andim, adims[MAXDIM], bndim, bdims[MAXDIM];
  Integer loA[MAXDIM], hiA[MAXDIM], ldA[MAXDIM];
  Integer loB[MAXDIM], hiB[MAXDIM], ldB[MAXDIM];
  Integer g_A = *g_a, g_B = *g_b;
  void *A_ptr, *B_ptr;
  Integer bvalue[MAXDIM], bunit[MAXDIM], baseldA[MAXDIM];
  Integer idx, n1dim, ctype;
  Integer atotal, btotal;
  int isum, alen;
  long lsum;
  double dsum;
  DoubleComplex zsum;
  float fsum;
  Integer me= ga_nodeid_(), temp_created=0;
  Integer nproc = ga_nnodes_();
  Integer num_blocks_a, num_blocks_b;
  char *tempname = "temp", transp, transp_a, transp_b;
  int local_sync_begin;
  Integer a_grp, b_grp;

  local_sync_begin = _ga_sync_begin; 
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  GA_PUSH_NAME("ngai_dot_patch");
  a_grp = ga_get_pgroup_(g_a);
  b_grp = ga_get_pgroup_(g_b);
  if (a_grp != b_grp)
    ga_error("Both arrays must be defined on same group",0L);
  me = ga_pgroup_nodeid_(&a_grp);

  nga_inquire_internal_(g_a, &atype, &andim, adims);
  nga_inquire_internal_(g_b, &btype, &bndim, bdims);

  if(atype != btype ) ga_error(" type mismatch ", 0L);

  /* check if patch indices and g_a dims match */
  for(i=0; i<andim; i++)
    if(alo[i] <= 0 || ahi[i] > adims[i])
      ga_error("g_a indices out of range ", *g_a);
  for(i=0; i<bndim; i++)
    if(blo[i] <= 0 || bhi[i] > bdims[i])
      ga_error("g_b indices out of range ", *g_b);

  /* check if numbers of elements in two patches match each other */
  atotal = 1; for(i=0; i<andim; i++) atotal *= (ahi[i] - alo[i] + 1);
  btotal = 1; for(i=0; i<bndim; i++) btotal *= (bhi[i] - blo[i] + 1);

  if(atotal != btotal)
    ga_error("  capacities of patches do not match ", 0L);

  /* is transpose operation required ? */
  /* -- only if for one array transpose operation requested*/
  transp_a = (*t_a == 'n' || *t_a =='N')? 'n' : 't';
  transp_b = (*t_b == 'n' || *t_b =='N')? 'n' : 't';
  transp   = (transp_a == transp_b)? 'n' : 't';

  /* Find out if distribution is block-cyclic */
  num_blocks_a = ga_total_blocks_(g_a);
  num_blocks_b = ga_total_blocks_(g_b);

  if (num_blocks_a >= 0 || num_blocks_b >= 0) {
    if (transp_a == 't' || transp_b == 't')
      ga_error("transpose not supported for block-cyclic data ", 0);
  }

  isum = 0; dsum = 0.; zsum.real = 0.; zsum.imag = 0.; fsum = 0;lsum=0;
  switch (atype){
    case C_INT:
      *(int*)retval = isum;
      alen = 1;
      break;                                     
    case C_DCPL:
      ((double*)retval)[0] = zsum.real;
      ((double*)retval)[1] = zsum.imag;
      alen = 2;
      break;                                     
    case  C_DBL:
      *(double*)retval = dsum;
      alen = 1;
      break;                                     
    case  C_FLOAT:
      *(float*)retval = fsum;
      alen = 1;
      break;                                     
    case C_LONG:
      *(long*)retval = lsum;
      alen = 1;
      break;                                     
  }


  if (num_blocks_a < 0 && num_blocks_b < 0) {
    /* find out coordinates of patches of g_A and g_B that I own */
    nga_distribution_(&g_A, &me, loA, hiA);
    nga_distribution_(&g_B, &me, loB, hiB);

    if(ngai_comp_patch(andim, loA, hiA, bndim, loB, hiB) &&
        ngai_comp_patch(andim, alo, ahi, bndim, blo, bhi)) compatible = 1;
    else compatible = 0;
    ga_igop(GA_TYPE_GSM, &compatible, 1, "*");
    if(!(compatible && (transp=='n'))) {
      /* either patches or distributions do not match:
       *        - create a temp array that matches distribution of g_a
       *        - copy & reshape patch of g_b into g_B
       */
      if (!ga_duplicate(g_a, &g_B, tempname))
        ga_error("duplicate failed",0L);

      nga_copy_patch(&transp, g_b, blo, bhi, &g_B, alo, ahi);
      bndim = andim;
      temp_created = 1;
      nga_distribution_(&g_B, &me, loB, hiB);
    }

    if(!ngai_comp_patch(andim, loA, hiA, bndim, loB, hiB))
      ga_error(" patches mismatch ",0);


    /* A[83:125,1:1]  <==> B[83:125] */
    if(andim > bndim) andim = bndim; /* need more work */

    /*  determine subsets of my patches to access  */
    if(ngai_patch_intersect(alo, ahi, loA, hiA, andim)){
      nga_access_ptr(&g_A, loA, hiA, &A_ptr, ldA);
      nga_access_ptr(&g_B, loA, hiA, &B_ptr, ldB);

      ngai_dot_local_patch(atype, andim, loA, hiA, ldA, A_ptr, B_ptr,
          &alen, retval);
      /* release access to the data */
      nga_release_(&g_A, loA, hiA);
      nga_release_(&g_B, loA, hiA);
    }
  } else {
    /* Create copy of g_b identical with identical distribution as g_a */
    if (!ga_duplicate(g_a, &g_B, tempname))
      ga_error("duplicate failed",0L);
    nga_copy_patch(&transp, g_b, blo, bhi, &g_B, alo, ahi);
    temp_created = 1;

    /* If g_a regular distribution, then just use normal dot product on patch */
    if (num_blocks_a < 0) {
      /* find out coordinates of patches of g_A and g_B that I own */
      nga_distribution_(&g_A, &me, loA, hiA);
      nga_distribution_(&g_B, &me, loB, hiB);

      if(!ngai_comp_patch(andim, loA, hiA, bndim, loB, hiB))
        ga_error(" patches mismatch ",0);

      /* A[83:125,1:1]  <==> B[83:125] */
      if(andim > bndim) andim = bndim; /* need more work */
      if(ngai_patch_intersect(alo, ahi, loA, hiA, andim)){
        nga_access_ptr(&g_A, loA, hiA, &A_ptr, ldA);
        nga_access_ptr(&g_B, loA, hiA, &B_ptr, ldB);

        ngai_dot_local_patch(atype, andim, loA, hiA, ldA, A_ptr, B_ptr,
            &alen, retval);
        /* release access to the data */
        nga_release_(&g_A, loA, hiA);
        nga_release_(&g_B, loA, hiA);
      }
    } else {
      Integer lo[MAXDIM], hi[MAXDIM];
      Integer offset, jtot, last;
      for (i=me; i<num_blocks_a; i += nproc) {
        nga_distribution_(&g_A, &i, loA, hiA);
        /* make copies of loA and hiA since ngai_patch_intersect destroys
           original versions */
        for (j=0; j<andim; j++) {
          lo[j] = loA[j];
          hi[j] = hiA[j];
        }
        if(ngai_patch_intersect(alo, ahi, loA, hiA, andim)){
          nga_access_block_ptr(&g_A, &i, &A_ptr, ldA);
          nga_access_block_ptr(&g_B, &i, &B_ptr, ldB);

          /* evaluate offsets for system */
          offset = 0;
          last = andim-1;
          jtot = 1;
          for (j=0; j<last; j++) {
            offset += (loA[j] - lo[j])*jtot;
            jtot = ldA[j];
          }
          offset += (loA[last]-lo[last])*jtot;

          /* offset pointers by correct amount */
          switch (atype){
            case C_INT:
              A_ptr = (void*)((int*)(A_ptr) + offset);
              B_ptr = (void*)((int*)(B_ptr) + offset);
              break;                                     
            case C_DCPL:
              A_ptr = (void*)((DoubleComplex*)(A_ptr) + offset);
              B_ptr = (void*)((DoubleComplex*)(B_ptr) + offset);
              break;                                     
            case  C_DBL:
              A_ptr = (void*)((double*)(A_ptr) + offset);
              B_ptr = (void*)((double*)(B_ptr) + offset);
              break;                                     
            case  C_FLOAT:
              A_ptr = (void*)((float*)(A_ptr) + offset);
              B_ptr = (void*)((float*)(B_ptr) + offset);
              break;                                     
            case C_LONG:
              A_ptr = (void*)((long*)(A_ptr) + offset);
              B_ptr = (void*)((long*)(B_ptr) + offset);
              break;                                     
          }
          ngai_dot_local_patch(atype, andim, loA, hiA, ldA, A_ptr, B_ptr,
              &alen, retval);
          /* release access to the data */
          nga_release_block_(&g_A, &i);
          nga_release_block_(&g_B, &i);
        }
      }
    }
  }

  /*convert from C data type to ARMCI type */
  switch(atype) {
    case C_FLOAT: ctype=ARMCI_FLOAT; break;
    case C_DBL: ctype=ARMCI_DOUBLE; break;
    case C_INT: ctype=ARMCI_INT; break;
    case C_LONG: ctype=ARMCI_LONG; break;
    case C_DCPL: ctype=ARMCI_DOUBLE; break;
    default: ga_error("ngai_dot_patch: type not supported",atype);
  }

  if (ga_is_mirrored_(g_a) && ga_is_mirrored_(g_b)) {
    armci_msg_gop_scope(SCOPE_NODE,retval,alen,"+",ctype);
  } else {
#ifdef MPI
    extern ARMCI_Group* ga_get_armci_group_(int);
#endif
    if (a_grp == -1) {
      armci_msg_gop_scope(SCOPE_ALL,retval,alen,"+",ctype);
#ifdef MPI
    } else {
      armci_msg_group_gop_scope(SCOPE_ALL,retval,alen,"+",ctype,
          ga_get_armci_group_((int)a_grp));
#endif
    }
  }

  if(temp_created) ga_destroy_(&g_B);
  GA_POP_NAME;
}


/*\ compute Integer DOT PRODUCT of two patches
 *
 *          . different shapes and distributions allowed but not recommended
 *          . the same number of elements required
\*/
Integer nga_idot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi)
     Integer *g_a, *alo, *ahi;    /* patch of g_a */
     Integer *g_b, *blo, *bhi;    /* patch of g_b */
     char    *t_a, *t_b;        /* transpose operators */
{
    Integer atype, btype, andim, adims[MAXDIM], bndim, bdims[MAXDIM];
    Integer sum = 0.;

    GA_PUSH_NAME("nga_idot_patch");
    
    nga_inquire_internal_(g_a, &atype, &andim, adims);
    nga_inquire_internal_(g_b, &btype, &bndim, bdims);

    if(atype != btype || ((atype != C_INT )&&(atype !=C_LONG))) ga_error(" wrong types ", 0L);

    ngai_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, (void *)(&sum));

    GA_POP_NAME;
    return ((Integer)sum);
}

/*\ compute Double Precision DOT PRODUCT of two patches
 *
 *          . different shapes and distributions allowed but not recommended
 *          . the same number of elements required
\*/
DoublePrecision nga_ddot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi)
     Integer *g_a, *alo, *ahi;    /* patch of g_a */
     Integer *g_b, *blo, *bhi;    /* patch of g_b */
     char    *t_a, *t_b;        /* transpose operators */
{
    Integer atype, btype, andim, adims[MAXDIM], bndim, bdims[MAXDIM];
    DoublePrecision  sum = 0.;
 
#ifdef GA_USE_VAMPIR
    vampir_begin(NGA_DDOT_PATCH,__FILE__,__LINE__);
#endif    

    GA_PUSH_NAME("nga_ddot_patch");
    
    nga_inquire_internal_(g_a, &atype, &andim, adims);
    nga_inquire_internal_(g_b, &btype, &bndim, bdims);

    if(atype != btype || (atype != C_DBL )) ga_error(" wrong types ", 0L);

    ngai_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, (void *)(&sum));

    GA_POP_NAME;
#ifdef GA_USE_VAMPIR
    vampir_end(NGA_DDOT_PATCH,__FILE__,__LINE__);
#endif    
    return (sum);
}

/*\ compute float DOT PRODUCT of two patches
 *
 *          . different shapes and distributions allowed but not recommended
 *          . the same number of elements required
\*/
float nga_fdot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi)
     Integer *g_a, *alo, *ahi;    /* patch of g_a */
     Integer *g_b, *blo, *bhi;    /* patch of g_b */
     char    *t_a, *t_b;        /* transpose operators */
{
    Integer atype, btype, andim, adims[MAXDIM], bndim, bdims[MAXDIM];
    float  sum = 0.;
 
    GA_PUSH_NAME("nga_fdot_patch");

    nga_inquire_internal_(g_a, &atype, &andim, adims);
    nga_inquire_internal_(g_b, &btype, &bndim, bdims);
 
    if(atype != btype || (atype != C_FLOAT )) ga_error(" wrong types ", 0L);
 
    ngai_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, (void *)(&sum));
 
    GA_POP_NAME;
    return (sum);
}                                      

/*\ compute Double Complex DOT PRODUCT of two patches
 *
 *          . different shapes and distributions allowed but not recommended
 *          . the same number of elements required
\*/
DoubleComplex nga_zdot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi)
     Integer *g_a, *alo, *ahi;    /* patch of g_a */
     Integer *g_b, *blo, *bhi;    /* patch of g_b */
     char    *t_a, *t_b;          /* transpose operators */
{
Integer atype, btype, andim, adims[MAXDIM], bndim, bdims[MAXDIM];
DoubleComplex  sum;

#ifdef GA_USE_VAMPIR
   vampir_begin(NGA_ZDOT_PATCH,__FILE__,__LINE__);
#endif    

   GA_PUSH_NAME("nga_zdot_patch");

   ga_inquire_internal_(g_a, &atype, &andim, adims);
   ga_inquire_internal_(g_b, &btype, &bndim, bdims);

   if(atype != btype || (atype != C_DCPL )) ga_error(" wrong types ", 0L);

   ngai_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi,
                  (void *)(&sum));

   GA_POP_NAME;
#ifdef GA_USE_VAMPIR
   vampir_end(NGA_ZDOT_PATCH,__FILE__,__LINE__);
#endif    
   return (sum);
}


/*\ compute DOT PRODUCT of two patches
 *  Fortran interface
\*/
void FATR ngai_dot_patch_(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, retval)
Integer *g_a, *alo, *ahi;    /* patch of g_a */ 
Integer *g_b, *blo, *bhi;    /* patch of g_b */
void *retval; 

#if defined(CRAY) || defined(WIN32)
     _fcd   t_a, t_b;                          /* transpose operators */
{  ngai_dot_patch(g_a, _fcdtocp(t_a), alo, ahi, 
                  g_b, _fcdtocp(t_b), blo, bhi, retval);}
#else 
     char    *t_a, *t_b;                          /* transpose operators */
{  ngai_dot_patch(g_a, t_a, alo, ahi,
                g_b, t_b, blo, bhi, retval);}
#endif

/*\
 *  Set all values in patch to value stored in *val
\*/
ngai_set_patch_value(Integer type, Integer ndim, Integer *loA, Integer *hiA,
                     Integer *ld, void *data_ptr, void *val)
{
  Integer n1dim, i, j, idx;
  Integer bvalue[MAXDIM], bunit[MAXDIM], baseld[MAXDIM];
  /* number of n-element of the first dimension */
  n1dim = 1; for(i=1; i<ndim; i++) n1dim *= (hiA[i] - loA[i] + 1);

  /* calculate the destination indices */
  bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
  /* baseld[0] = ld[0]
   * baseld[1] = ld[0] * ld[1]
   * baseld[2] = ld[0] * ld[1] * ld[2] .....
   */
  baseld[0] = ld[0]; baseld[1] = baseld[0] *ld[1];
  for(i=2; i<ndim; i++) {
    bvalue[i] = 0;
    bunit[i] = bunit[i-1] * (hiA[i-1] - loA[i-1] + 1);
    baseld[i] = baseld[i-1] * ld[i];
  }

  switch (type){
    case C_INT:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }
        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          ((int *)data_ptr)[idx+j] = *(int*)val;
      }
      break;
    case C_DCPL:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++) {
          DoubleComplex tmp = *(DoubleComplex *)val;
          ((DoubleComplex *)data_ptr)[idx+j].real = tmp.real;
          ((DoubleComplex *)data_ptr)[idx+j].imag = tmp.imag;
        }
      }

      break;
    case C_DBL:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++) 
          ((double*)data_ptr)[idx+j] =
            *(double*)val;
      }
      break;
    case C_FLOAT:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }
        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          ((float *)data_ptr)[idx+j] = *(float*)val;
      }
      break;     
    case C_LONG:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }
        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          ((long *)data_ptr)[idx+j] = *(long*)val;
      } 
      break;                          
    default: ga_error(" wrong data type ",type);
  }

}

/*\ FILL IN ARRAY WITH VALUE 
\*/
void FATR nga_fill_patch_(Integer *g_a, Integer *lo, Integer *hi, void* val)
{
  Integer i;
  Integer ndim, dims[MAXDIM], type;
  Integer loA[MAXDIM], hiA[MAXDIM], ld[MAXDIM];
  void *data_ptr;
  Integer num_blocks, nproc;
  Integer me= ga_nodeid_();
  int local_sync_begin,local_sync_end;

#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_FILL_PATCH,__FILE__,__LINE__);
#endif 
  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_(); 

  GA_PUSH_NAME("nga_fill_patch");

  nga_inquire_internal_(g_a,  &type, &ndim, dims);
  num_blocks = ga_total_blocks_(g_a);

  if (num_blocks < 0) { 
    /* get limits of VISIBLE patch */ 
    nga_distribution_(g_a, &me, loA, hiA);

    /*  determine subset of my local patch to access  */
    /*  Output is in loA and hiA */
    if(ngai_patch_intersect(lo, hi, loA, hiA, ndim)){

      /* get data_ptr to corner of patch */
      /* ld are leading dimensions INCLUDING ghost cells */
      nga_access_ptr(g_a, loA, hiA, &data_ptr, ld);

      /* set all values in patch to *val */
      ngai_set_patch_value(type, ndim, loA, hiA, ld, data_ptr, val);

      /* release access to the data */
      nga_release_update_(g_a, loA, hiA);
    }
  } else {
    Integer offset, j, jtmp, chk;
    Integer loS[MAXDIM];
    nproc = ga_nnodes_();
    for (i=me; i<num_blocks; i += nproc) {
      /* get limits of VISIBLE patch */ 
      nga_distribution_(g_a, &i, loA, hiA);

      /* loA is changed by ngai_patch_intersect, so
         save a copy */
      for (j=0; j<ndim; j++) {
        loS[j] = loA[j];
      }

      /*  determine subset of my local patch to access  */
      /*  Output is in loA and hiA */
      if(ngai_patch_intersect(lo, hi, loA, hiA, ndim)){

        /* get data_ptr to corner of patch */
        /* ld are leading dimensions INCLUDING ghost cells */
        nga_access_block_ptr(g_a, &i, &data_ptr, ld);

        /* Check for partial overlap */
        chk = 1;
        for (j=0; j<ndim; j++) {
          if (loS[j] < loA[j]) {
            chk=0;
            break;
          }
        }
        if (!chk) {
          /* Evaluate additional offset for pointer */
          offset = 0;
          jtmp = 1;
          for (j=0; j<ndim-1; j++) {
            offset += (loA[j]-loS[j])*jtmp;
            jtmp *= ld[j];
          }
          offset += (loA[ndim-1]-loS[ndim-1])*jtmp;
          switch (type){
            case C_INT:
              data_ptr = (void*)((int*)data_ptr + offset);
              break;
            case C_DCPL:
              data_ptr = (void*)((double*)data_ptr + 2*offset);
              break;
            case C_DBL:
              data_ptr = (void*)((double*)data_ptr + offset);
              break;
            case C_FLOAT:
              data_ptr = (void*)((float*)data_ptr + offset);
              break;     
            case C_LONG:
              data_ptr = (void*)((long*)data_ptr + offset);
              break;                          
            default: ga_error(" wrong data type ",type);
          }
        }

        /* set all values in patch to *val */
        ngai_set_patch_value(type, ndim, loA, hiA, ld, data_ptr, val);

        /* release access to the data */
        nga_release_update_block_(g_a, &i);
      }
    }
  }
  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_FILL_PATCH,__FILE__,__LINE__);
#endif 
}

ngai_scale_patch_value(Integer type, Integer ndim, Integer *loA, Integer *hiA,
                     Integer *ld, void *src_data_ptr, void *alpha)
{
  Integer n1dim, i, j, idx;
  Integer bvalue[MAXDIM], bunit[MAXDIM], baseld[MAXDIM];
  DoublePrecision tmp1_real, tmp1_imag, tmp2_real, tmp2_imag;
  /* number of n-element of the first dimension */
  n1dim = 1; for(i=1; i<ndim; i++) n1dim *= (hiA[i] - loA[i] + 1);

  /* calculate the destination indices */
  bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
  /* baseld[0] = ld[0]
   * baseld[1] = ld[0] * ld[1]
   * baseld[2] = ld[0] * ld[1] * ld[2] .....
   */
  baseld[0] = ld[0]; baseld[1] = baseld[0] *ld[1];
  for(i=2; i<ndim; i++) {
    bvalue[i] = 0;
    bunit[i] = bunit[i-1] * (hiA[i-1] - loA[i-1] + 1);
    baseld[i] = baseld[i-1] * ld[i];
  }

  /* scale local part of g_a */
  switch(type){
    case C_DBL:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++) 
          ((double*)src_data_ptr)[idx+j]  *=
            *(double*)alpha;                    
      }
      break;
    case C_DCPL:
      for(i=0; i<n1dim; i++) {
        idx = 0;  
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++) {
          tmp1_real =((DoubleComplex *)src_data_ptr)[idx+j].real;
          tmp1_imag =((DoubleComplex *)src_data_ptr)[idx+j].imag;
          tmp2_real = (*(DoubleComplex*)alpha).real;
          tmp2_imag = (*(DoubleComplex*)alpha).imag;

          ((DoubleComplex *)src_data_ptr)[idx+j].real =
            tmp1_real*tmp2_real  - tmp1_imag * tmp2_imag;
          ((DoubleComplex *)src_data_ptr)[idx+j].imag =
            tmp2_imag*tmp1_real  + tmp1_imag * tmp2_real;
        }
      }
      break;
    case C_INT:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          ((int*)src_data_ptr)[idx+j]  *= *(int*)alpha;
      }
      break;
    case C_LONG:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          ((long *)src_data_ptr)[idx+j]  *= *(long*)alpha; 
      }
      break;
    case C_FLOAT:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseld[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiA[0]-loA[0]+1); j++)
          ((float *)src_data_ptr)[idx+j]  *= *(float*)alpha;
      }                                                           
      break;
    default: ga_error(" wrong data type ",type);
  }
}

/*\ SCALE ARRAY 
\*/
void FATR nga_scale_patch_(Integer *g_a, Integer *lo, Integer *hi,
                          void *alpha)
{
    Integer i, j;
    Integer ndim, dims[MAXDIM], type;
    Integer loA[MAXDIM], hiA[MAXDIM];
    Integer ld[MAXDIM];
    void *src_data_ptr;
    Integer num_blocks, nproc;
    Integer idx, n1dim;
    Integer bvalue[MAXDIM], bunit[MAXDIM], baseld[MAXDIM];
    DoublePrecision tmp1_real, tmp1_imag, tmp2_real, tmp2_imag;
    Integer me= ga_nodeid_();
    int local_sync_begin,local_sync_end;

#ifdef GA_USE_VAMPIR
    vampir_begin(NGA_SCALE_PATCH,__FILE__,__LINE__);
#endif 
    local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    if(local_sync_begin)ga_sync_();

    GA_PUSH_NAME("nga_scal_patch");
    
    nga_inquire_internal_(g_a,  &type, &ndim, dims);
    num_blocks = ga_total_blocks_(g_a);

    if (num_blocks < 0) {
      nga_distribution_(g_a, &me, loA, hiA);

      /* determine subset of my patch to access */
      if (ngai_patch_intersect(lo, hi, loA, hiA, ndim)){
        nga_access_ptr(g_a, loA, hiA, &src_data_ptr, ld);

        ngai_scale_patch_value(type, ndim, loA, hiA, ld, src_data_ptr, alpha);

        /* release access to the data */
        nga_release_update_(g_a, loA, hiA); 
      }
    } else {
      Integer offset, j, jtmp, chk;
      Integer loS[MAXDIM];
      nproc = ga_nnodes_();
      for (i=me; i<num_blocks; i += nproc) {
        /* get limits of VISIBLE patch */
        nga_distribution_(g_a, &i, loA, hiA);

        /* loA is changed by ngai_patch_intersect, so
           save a copy */
        for (j=0; j<ndim; j++) {
          loS[j] = loA[j];
        }

        /*  determine subset of my local patch to access  */
        /*  Output is in loA and hiA */
        if(ngai_patch_intersect(lo, hi, loA, hiA, ndim)){

          /* get src_data_ptr to corner of patch */
          /* ld are leading dimensions INCLUDING ghost cells */
          nga_access_block_ptr(g_a, &i, &src_data_ptr, ld);

          /* Check for partial overlap */
          chk = 1;
          for (j=0; j<ndim; j++) {
            if (loS[j] < loA[j]) {
              chk=0;
              break;
            }
          }
          if (!chk) {
            /* Evaluate additional offset for pointer */
            offset = 0;
            jtmp = 1;
            for (j=0; j<ndim-1; j++) {
              offset += (loA[j]-loS[j])*jtmp;
              jtmp *= ld[j];
            }
            offset += (loA[ndim-1]-loS[ndim-1])*jtmp;
            switch (type){
              case C_INT:
                src_data_ptr = (void*)((int*)src_data_ptr + offset);
                break;
              case C_DCPL:
                src_data_ptr = (void*)((double*)src_data_ptr + 2*offset);
                break;
              case C_DBL:
                src_data_ptr = (void*)((double*)src_data_ptr + offset);
                break;
              case C_FLOAT:
                src_data_ptr = (void*)((float*)src_data_ptr + offset);
                break;     
              case C_LONG:
                src_data_ptr = (void*)((long*)src_data_ptr + offset);
                break;                          
              default: ga_error(" wrong data type ",type);
            }
          }

          /* set all values in patch to *val */
          ngai_scale_patch_value(type, ndim, loA, hiA, ld, src_data_ptr, alpha);

          /* release access to the data */
          nga_release_update_block_(g_a, &i);
        }
      }



    }
    GA_POP_NAME;
    if(local_sync_end)ga_sync_();   
#ifdef GA_USE_VAMPIR
    vampir_end(NGA_SCALE_PATCH,__FILE__,__LINE__);
#endif 
}

/*\ Utility function to add patch values together
\*/
void ngai_add_patch_values(Integer type, void* alpha, void *beta,
                           Integer ndim, Integer *loC, Integer *hiC, Integer *ldC,
                           void *A_ptr, void *B_ptr, void *C_ptr)
{
  Integer bvalue[MAXDIM], bunit[MAXDIM], baseldC[MAXDIM];
  Integer idx, n1dim;
  Integer i, j;
  /* compute "local" add */

  /* number of n-element of the first dimension */
  n1dim = 1; for(i=1; i<ndim; i++) n1dim *= (hiC[i] - loC[i] + 1);

  /* calculate the destination indices */
  bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
  /* baseld[0] = ld[0]
   * baseld[1] = ld[0] * ld[1]
   * baseld[2] = ld[0] * ld[1] * ld[2] .....
   */
  baseldC[0] = ldC[0]; baseldC[1] = baseldC[0] *ldC[1];
  for(i=2; i<ndim; i++) {
    bvalue[i] = 0;
    bunit[i] = bunit[i-1] * (hiC[i-1] - loC[i-1] + 1);
    baseldC[i] = baseldC[i-1] * ldC[i];
  }

  switch(type){
    case C_DBL:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseldC[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiC[0]-loC[0]+1); j++)
          ((double*)C_ptr)[idx+j] =
            *(double*)alpha *
            ((double*)A_ptr)[idx+j] +
            *(double*)beta *
            ((double*)B_ptr)[idx+j];
      }
      break;
    case C_DCPL:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseldC[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiC[0]-loC[0]+1); j++) {
          DoubleComplex a = ((DoubleComplex *)A_ptr)[idx+j];
          DoubleComplex b = ((DoubleComplex *)B_ptr)[idx+j];
          DoubleComplex x= *(DoubleComplex*)alpha;
          DoubleComplex y= *(DoubleComplex*)beta;
          ((DoubleComplex *)C_ptr)[idx+j].real = x.real*a.real -
            x.imag*a.imag + y.real*b.real - y.imag*b.imag;
          ((DoubleComplex *)C_ptr)[idx+j].imag = x.real*a.imag +
            x.imag*a.real + y.real*b.imag + y.imag*b.real;
        }
      }
      break;
    case C_INT:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseldC[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiC[0]-loC[0]+1); j++)
          ((int*)C_ptr)[idx+j] = *(int *)alpha *
            ((int*)A_ptr)[idx+j] + *(int*)beta *
            ((int*)B_ptr)[idx+j];
      }
      break;
    case C_FLOAT:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseldC[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiC[0]-loC[0]+1); j++)
          ((float *)C_ptr)[idx+j] = *(float *)alpha *
            ((float *)A_ptr)[idx+j] + *(float *)beta *
            ((float *)B_ptr)[idx+j];
      }
      break;
    case C_LONG:
      for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<ndim; j++) {
          idx += bvalue[j] * baseldC[j-1];
          if(((i+1) % bunit[j]) == 0) bvalue[j]++;
          if(bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
        }

        for(j=0; j<(hiC[0]-loC[0]+1); j++)
          ((long *)C_ptr)[idx+j] = *(long *)alpha *
            ((long *)A_ptr)[idx+j] + *(long *)beta *
            ((long *)B_ptr)[idx+j];
      }
      break;
    default: ga_error(" wrong data type ",type);
  }
}

/*\  SCALED ADDITION of two patches
\*/
void FATR nga_add_patch_(alpha, g_a, alo, ahi, beta,  g_b, blo, bhi,
                         g_c, clo, chi)
Integer *g_a, *alo, *ahi;    /* patch of g_a */
Integer *g_b, *blo, *bhi;    /* patch of g_b */
Integer *g_c, *clo, *chi;    /* patch of g_c */
DoublePrecision *alpha, *beta;
{
  Integer i, j;
  Integer compatible;
  Integer atype, btype, ctype;
  Integer andim, adims[MAXDIM], bndim, bdims[MAXDIM], cndim, cdims[MAXDIM];
  Integer loA[MAXDIM], hiA[MAXDIM], ldA[MAXDIM];
  Integer loB[MAXDIM], hiB[MAXDIM], ldB[MAXDIM];
  Integer loC[MAXDIM], hiC[MAXDIM], ldC[MAXDIM];
  void *A_ptr, *B_ptr, *C_ptr;
  Integer bvalue[MAXDIM], bunit[MAXDIM], baseldC[MAXDIM];
  Integer idx, n1dim;
  Integer atotal, btotal;
  Integer g_A = *g_a, g_B = *g_b;
  Integer me= ga_nodeid_(), A_created=0, B_created=0;
  Integer nproc = ga_nnodes_();
  Integer num_blocks_a, num_blocks_b, num_blocks_c;
  char *tempname = "temp", notrans='n';
  int local_sync_begin,local_sync_end;

#ifdef GA_USE_VAMPIR
  vampir_begin(NGA_ADD_PATCH,__FILE__,__LINE__);
#endif 
  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  GA_PUSH_NAME("nga_add_patch");

  nga_inquire_internal_(g_a, &atype, &andim, adims);
  nga_inquire_internal_(g_b, &btype, &bndim, bdims);
  nga_inquire_internal_(g_c, &ctype, &cndim, cdims);

  if(atype != btype || atype != ctype ) ga_error(" types mismatch ", 0L); 

  /* check if patch indices and dims match */
  for(i=0; i<andim; i++)
    if(alo[i] <= 0 || ahi[i] > adims[i])
      ga_error("g_a indices out of range ", *g_a);
  for(i=0; i<bndim; i++)
    if(blo[i] <= 0 || bhi[i] > bdims[i])
      ga_error("g_b indices out of range ", *g_b);
  for(i=0; i<cndim; i++)
    if(clo[i] <= 0 || chi[i] > cdims[i])
      ga_error("g_b indices out of range ", *g_c);

  /* check if numbers of elements in patches match each other */
  n1dim = 1; for(i=0; i<cndim; i++) n1dim *= (chi[i] - clo[i] + 1);
  atotal = 1; for(i=0; i<andim; i++) atotal *= (ahi[i] - alo[i] + 1);
  btotal = 1; for(i=0; i<bndim; i++) btotal *= (bhi[i] - blo[i] + 1);

  if((atotal != n1dim) || (btotal != n1dim))
    ga_error("  capacities of patches do not match ", 0L);

  num_blocks_a = ga_total_blocks_(g_a);
  num_blocks_b = ga_total_blocks_(g_b);
  num_blocks_c = ga_total_blocks_(g_c);
 
  if (num_blocks_a < 0 && num_blocks_b < 0 && num_blocks_c < 0) {
    /* find out coordinates of patches of g_a, g_b and g_c that I own */
    nga_distribution_(&g_A, &me, loA, hiA);
    nga_distribution_(&g_B, &me, loB, hiB);
    nga_distribution_( g_c, &me, loC, hiC);

    /* test if the local portion of patches matches */
    if(ngai_comp_patch(andim, loA, hiA, cndim, loC, hiC) &&
        ngai_comp_patch(andim, alo, ahi, cndim, clo, chi)) compatible = 1;
    else compatible = 0;
    ga_igop(GA_TYPE_GSM, &compatible, 1, "*");
    if(!compatible) {
      /* either patches or distributions do not match:
       *        - create a temp array that matches distribution of g_c
       *        - do C<= A
       */
      if(*g_b != *g_c) {
        nga_copy_patch(&notrans, g_a, alo, ahi, g_c, clo, chi);
        andim = cndim;
        g_A = *g_c;
        nga_distribution_(&g_A, &me, loA, hiA);
      }
      else {
        if (!ga_duplicate(g_c, &g_A, tempname))
          ga_error("ga_dadd_patch: dup failed", 0L);
        nga_copy_patch(&notrans, g_a, alo, ahi, &g_A, clo, chi);
        andim = cndim;
        A_created = 1;
        nga_distribution_(&g_A, &me, loA, hiA);
      }
    }

    /* test if the local portion of patches matches */
    if(ngai_comp_patch(bndim, loB, hiB, cndim, loC, hiC) &&
        ngai_comp_patch(bndim, blo, bhi, cndim, clo, chi)) compatible = 1;
    else compatible = 0;
    ga_igop(GA_TYPE_GSM, &compatible, 1, "*");
    if(!compatible) {
      /* either patches or distributions do not match:
       *        - create a temp array that matches distribution of g_c
       *        - copy & reshape patch of g_b into g_B
       */
      if (!ga_duplicate(g_c, &g_B, tempname))
        ga_error("ga_dadd_patch: dup failed", 0L);
      nga_copy_patch(&notrans, g_b, blo, bhi, &g_B, clo, chi);
      bndim = cndim;
      B_created = 1;
      nga_distribution_(&g_B, &me, loB, hiB);
    }        

    if(andim > bndim) cndim = bndim;
    if(andim < bndim) cndim = andim;

    if(!ngai_comp_patch(andim, loA, hiA, cndim, loC, hiC))
      ga_error(" A patch mismatch ", g_A); 
    if(!ngai_comp_patch(bndim, loB, hiB, cndim, loC, hiC))
      ga_error(" B patch mismatch ", g_B);

    /*  determine subsets of my patches to access  */
    if (ngai_patch_intersect(clo, chi, loC, hiC, cndim)){
      nga_access_ptr(&g_A, loC, hiC, &A_ptr, ldA);
      nga_access_ptr(&g_B, loC, hiC, &B_ptr, ldB);
      nga_access_ptr( g_c, loC, hiC, &C_ptr, ldC);

      ngai_add_patch_values(atype, alpha, beta, cndim,
          loC, hiC, ldC, A_ptr, B_ptr, C_ptr);

      /* release access to the data */
      nga_release_       (&g_A, loC, hiC);
      nga_release_       (&g_B, loC, hiC); 
      nga_release_update_( g_c, loC, hiC); 
    }
  } else {
    /* create copies of arrays A and B that are identically distributed
       as C*/
    if (!ga_duplicate(g_c, &g_A, tempname))
      ga_error("ga_dadd_patch: dup failed", 0L);
    nga_copy_patch(&notrans, g_a, alo, ahi, &g_A, clo, chi);
    andim = cndim;
    A_created = 1;

    if (!ga_duplicate(g_c, &g_B, tempname))
      ga_error("ga_dadd_patch: dup failed", 0L);
    nga_copy_patch(&notrans, g_b, blo, bhi, &g_B, clo, chi);
    bndim = cndim;
    B_created = 1;

    /* C is normally distributed so just add copies together for regular
       arrays */
    if (num_blocks_c < 0) {
      nga_distribution_( g_c, &me, loC, hiC);
      if(andim > bndim) cndim = bndim;
      if(andim < bndim) cndim = andim;
      if (ngai_patch_intersect(clo, chi, loC, hiC, cndim)){
        nga_access_ptr(&g_A, loC, hiC, &A_ptr, ldA);
        nga_access_ptr(&g_B, loC, hiC, &B_ptr, ldB);
        nga_access_ptr( g_c, loC, hiC, &C_ptr, ldC);

        ngai_add_patch_values(atype, alpha, beta, cndim,
            loC, hiC, ldC, A_ptr, B_ptr, C_ptr);

        /* release access to the data */
        nga_release_       (&g_A, loC, hiC);
        nga_release_       (&g_B, loC, hiC); 
        nga_release_update_( g_c, loC, hiC); 
      }
    } else {
      Integer idx, lod[MAXDIM], hid[MAXDIM];
      Integer offset, jtot, last;
      for (idx = me; idx < num_blocks_c; idx += nproc) {
        nga_distribution_(g_c, &idx, loC, hiC);
        /* make temporary copies of loC and hiC since ngai_patch_intersect
           destroys original versions */
        for (j=0; j<cndim; j++) {
          lod[j] = loC[j];
          hid[j] = hiC[j];
        }
        if (ngai_patch_intersect(clo, chi, loC, hiC, cndim)) {
          nga_access_block_ptr(&g_A, &idx, &A_ptr, ldA);
          nga_access_block_ptr(&g_B, &idx, &B_ptr, ldB);
          nga_access_block_ptr( g_c, &idx, &C_ptr, ldC);

          /* evaluate offsets for system */
          offset = 0;
          last = cndim - 1;
          jtot = 1;
          for (j=0; j<last; j++) {
            offset += (loC[j] - lod[j])*jtot;
            jtot = ldC[j];
          }
          offset += (loC[last]-lod[last])*jtot;

          switch(ctype) {
            case C_DBL:
              A_ptr = (void*)((double*)(A_ptr) + offset);
              B_ptr = (void*)((double*)(B_ptr) + offset);
              C_ptr = (void*)((double*)(C_ptr) + offset);
              break;
            case C_INT:
              A_ptr = (void*)((int*)(A_ptr) + offset);
              B_ptr = (void*)((int*)(B_ptr) + offset);
              C_ptr = (void*)((int*)(C_ptr) + offset);
              break;
            case C_DCPL:
              A_ptr = (void*)((DoubleComplex*)(A_ptr) + offset);
              B_ptr = (void*)((DoubleComplex*)(B_ptr) + offset);
              C_ptr = (void*)((DoubleComplex*)(C_ptr) + offset);
              break;
            case C_FLOAT:
              A_ptr = (void*)((float*)(A_ptr) + offset);
              B_ptr = (void*)((float*)(B_ptr) + offset);
              C_ptr = (void*)((float*)(C_ptr) + offset);
              break;
            case C_LONG:
              A_ptr = (void*)((long*)(A_ptr) + offset);
              B_ptr = (void*)((long*)(B_ptr) + offset);
              C_ptr = (void*)((long*)(C_ptr) + offset);
              break;
            default:
              break;
          }
          ngai_add_patch_values(atype, alpha, beta, cndim,
              loC, hiC, ldC, A_ptr, B_ptr, C_ptr);

          /* release access to the data */
          nga_release_block_       (&g_A, &idx);
          nga_release_block_       (&g_B, &idx); 
          nga_release_update_block_( g_c, &idx); 
        }
      }
    }
  }

  if(A_created) ga_destroy_(&g_A);
  if(B_created) ga_destroy_(&g_B);

  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
#ifdef GA_USE_VAMPIR
  vampir_end(NGA_ADD_PATCH,__FILE__,__LINE__);
#endif 
}

void FATR nga_zero_patch_(Integer *g_a, Integer *lo, Integer *hi)
{
    Integer ndim, dims[MAXDIM], type;
    int ival = 0;
    long lval = 0; 
    double dval = 0.0;
    DoubleComplex cval;
    float fval = 0.0;
    void *valptr;
    int local_sync_begin,local_sync_end;
    
#ifdef GA_USE_VAMPIR
    vampir_begin(NGA_ZERO_PATCH,__FILE__,__LINE__);
#endif 

    local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
    _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
    if(local_sync_begin)ga_sync_();

    GA_PUSH_NAME("nga_zero_patch");
    
    nga_inquire_internal_(g_a,  &type, &ndim, dims);
    
    switch (type){
        case C_INT:
            valptr = (void *)(&ival);
            break;
        case C_DBL:
            valptr = (void *)(&dval);
            break;
        case C_DCPL:
        {
            cval.real = 0.0; cval.imag = 0.0;
            valptr = (void *)(&cval);
            break;
        }
        case C_FLOAT:
            valptr = (void *)(&fval);
            break;      
       case C_LONG:
            valptr = (void *)(&lval);
            break; 
        default: ga_error(" wrong data type ",type);
    }
    nga_fill_patch_(g_a, lo, hi, valptr);
    
    GA_POP_NAME;
    if(local_sync_end)ga_sync_();
#ifdef GA_USE_VAMPIR
    vampir_end(NGA_ZERO_PATCH,__FILE__,__LINE__);
#endif 
}

/*************************************************************
 *   2-dim patch operations                                  *
 *************************************************************/

/*\ COPY A PATCH AND POSSIBLY RESHAPE
 *
 *  . the element capacities of two patches must be identical
 *  . copy by column order - Fortran convention
\*/
void ga_copy_patch(char *trans, Integer *g_a, Integer *ailo, Integer *aihi,
                   Integer *ajlo, Integer *ajhi, Integer *g_b, Integer *bilo,
                   Integer *bihi, Integer *bjlo, Integer *bjhi)
{
    Integer alo[2], ahi[2], blo[2], bhi[2];

    alo[0] = *ailo; alo[1] = *ajlo;
    ahi[0] = *aihi; ahi[1] = *ajhi;
    blo[0] = *bilo; blo[1] = *bjlo;
    bhi[0] = *bihi; bhi[1] = *bjhi;

#ifdef GA_USE_VAMPIR
    vampir_begin(GA_COPY_PATCH,__FILE__,__LINE__);
#endif 
    nga_copy_patch(trans, g_a, alo, ahi, g_b, blo, bhi);
#ifdef GA_USE_VAMPIR
    vampir_end(GA_COPY_PATCH,__FILE__,__LINE__);
#endif 
}


/*\ COPY A PATCH AND POSSIBLY RESHAPE
 *  Fortran interface
\*/
void FATR ga_copy_patch_(trans, g_a, ailo, aihi, ajlo, ajhi,
                    g_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;
#if defined(CRAY) || defined(WIN32)
     _fcd    trans;
{ga_copy_patch(_fcdtocp(trans),g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi);}
#else 
     char*   trans;
{  ga_copy_patch(trans,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi); }
#endif


/*\ generic dot product routine
\*/
void gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                   g_b, t_b, bilo, bihi, bjlo, bjhi, retval)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     char    *t_a, *t_b;                          /* transpose operators */
     DoublePrecision *retval;
{
    Integer alo[2], ahi[2], blo[2], bhi[2];

    alo[0] = *ailo; alo[1] = *ajlo;
    ahi[0] = *aihi; ahi[1] = *ajhi;
    blo[0] = *bilo; blo[1] = *bjlo;
    bhi[0] = *bihi; bhi[1] = *bjhi;

    ngai_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, (void *)retval);
}



/*\ compute Double Precision DOT PRODUCT of two patches
 *
 *          . different shapes and distributions allowed but not recommended
 *          . the same number of elements required
\*/
DoublePrecision ga_ddot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                              g_b, t_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     char    *t_a, *t_b;                          /* transpose operators */
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
DoublePrecision  sum = 0.;

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_DDOT_PATCH,__FILE__,__LINE__);
#endif

   GA_PUSH_NAME("ga_ddot_patch");

   ga_inquire_internal_(g_a, &atype, &adim1, &adim2);
   ga_inquire_internal_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != C_DBL )) ga_error(" wrong types ", 0L);

   gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                 g_b, t_b, bilo, bihi, bjlo, bjhi, &sum);

   GA_POP_NAME;
#ifdef GA_USE_VAMPIR
   vampir_end(GA_DDOT_PATCH,__FILE__,__LINE__);
#endif
   return (sum);
}


/*\ compute Double Complex DOT PRODUCT of two patches
 *
 *          . different shapes and distributions allowed but not recommended
 *          . the same number of elements required
\*/
DoubleComplex ga_zdot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                            g_b, t_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     char    *t_a, *t_b;                          /* transpose operators */
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
DoubleComplex  sum;

#ifdef GA_USE_VAMPIR
   vampir_begin(GA_ZDOT_PATCH,__FILE__,__LINE__);
#endif

   GA_PUSH_NAME("ga_zdot_patch");

   ga_inquire_internal_(g_a, &atype, &adim1, &adim2);
   ga_inquire_internal_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != C_DCPL )) ga_error(" wrong types ", 0L);

   gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                 g_b, t_b, bilo, bihi, bjlo, bjhi, (DoublePrecision*)&sum);

   GA_POP_NAME;
#ifdef GA_USE_VAMPIR
   vampir_end(GA_ZDOT_PATCH,__FILE__,__LINE__);
#endif
   return (sum);
}

/*\ compute float DOT PRODUCT of two patches
 *
 *          . different shapes and distributions allowed but not recommended
 *          . the same number of elements required
\*/
float ga_fdot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                              g_b, t_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     char    *t_a, *t_b;                          /* transpose operators */
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
float  sum = 0.;

   GA_PUSH_NAME("ga_fdot_patch");
 
   ga_inquire_internal_(g_a, &atype, &adim1, &adim2);
   ga_inquire_internal_(g_b, &btype, &bdim1, &bdim2);
 
   if(atype != btype || (atype != C_FLOAT )) ga_error(" wrong types ", 0L);
 
   gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                 g_b, t_b, bilo, bihi, bjlo, bjhi, (DoublePrecision*)&sum);
 
   GA_POP_NAME;
   return (sum);
}                   


/*\ compute DOT PRODUCT of two patches
 *  Fortran interface
\*/
void FATR gai_dot_patch_(g_a, t_a, ailo, aihi, ajlo, ajhi,
                    g_b, t_b, bilo, bihi, bjlo, bjhi, retval)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     DoublePrecision *retval;

#if defined(CRAY) || defined(WIN32)
     _fcd   t_a, t_b;                          /* transpose operators */
{  gai_dot_patch(g_a, _fcdtocp(t_a), ailo, aihi, ajlo, ajhi,
                 g_b, _fcdtocp(t_b), bilo, bihi, bjlo, bjhi, retval);}
#else 
     char    *t_a, *t_b;                          /* transpose operators */
{ gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                g_b, t_b, bilo, bihi, bjlo, bjhi, retval);}
#endif




/*\ FILL IN ARRAY WITH VALUE 
\*/
void FATR ga_fill_patch_(g_a, ilo, ihi, jlo, jhi, val)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
     Void    *val;
{
    Integer lo[2], hi[2];

    lo[0] = *ilo; lo[1] = *jlo;
    hi[0] = *ihi; hi[1] = *jhi;

#ifdef GA_USE_VAMPIR
    vampir_begin(GA_FILL_PATCH,__FILE__,__LINE__);
#endif
    nga_fill_patch_(g_a, lo, hi, val);
#ifdef GA_USE_VAMPIR
    vampir_end(GA_FILL_PATCH,__FILE__,__LINE__);
#endif
}



/*\ SCALE ARRAY 
\*/
void FATR ga_scale_patch_(g_a, ilo, ihi, jlo, jhi, alpha)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
     DoublePrecision     *alpha;
{
    Integer lo[2], hi[2];

    lo[0] = *ilo; lo[1] = *jlo;
    hi[0] = *ihi; hi[1] = *jhi;

#ifdef GA_USE_VAMPIR
    vampir_begin(GA_SCALE_PATCH,__FILE__,__LINE__);
#endif
    nga_scale_patch_(g_a, lo, hi, (void *)alpha);
#ifdef GA_USE_VAMPIR
    vampir_end(GA_SCALE_PATCH,__FILE__,__LINE__);
#endif
}


/*\  SCALED ADDITION of two patches
\*/
void FATR ga_add_patch_(alpha, g_a, ailo, aihi, ajlo, ajhi,
                    beta,  g_b, bilo, bihi, bjlo, bjhi,
                           g_c, cilo, cihi, cjlo, cjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     DoublePrecision      *alpha, *beta;
{
    Integer alo[2], ahi[2], blo[2], bhi[2], clo[2], chi[2];

    alo[0] = *ailo; alo[1] = *ajlo;
    ahi[0] = *aihi; ahi[1] = *ajhi;
    blo[0] = *bilo; blo[1] = *bjlo;
    bhi[0] = *bihi; bhi[1] = *bjhi;
    clo[0] = *cilo; clo[1] = *cjlo;
    chi[0] = *cihi; chi[1] = *cjhi;
    
#ifdef GA_USE_VAMPIR
    vampir_begin(GA_ADD_PATCH,__FILE__,__LINE__);
#endif
    nga_add_patch_(alpha, g_a, alo, ahi, beta, g_b, blo, bhi, g_c, clo, chi);
#ifdef GA_USE_VAMPIR
    vampir_end(GA_ADD_PATCH,__FILE__,__LINE__);
#endif
}


