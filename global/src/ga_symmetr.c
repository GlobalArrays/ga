
/**
 * Symmetrizes matrix A:  A := .5 * (A+A`)
 * diag(A) remains unchanged
 * 
 */

#include "global.h"
#include "globalp.h"
#include "macdecls.h"

void FATR 
gai_add(Integer *lo, Integer *hi, Void *a, Void *b, DoublePrecision alpha,
	Integer type, Integer nelem, Integer ndim) {

  Integer i, j, m=0;
  Integer nrow, ncol, indexA=0, indexB=0;
  DoublePrecision *A = (DoublePrecision *)a, *B = (DoublePrecision*)b;
  Integer offset1=1, offset2=1;

  nrow = hi[ndim-2] - lo[ndim-2] + 1;
  ncol = hi[ndim-1] - lo[ndim-1] + 1;
  
  for(i=0; i<ndim-2; i++) {
    offset1 *= hi[i] - lo[i] + 1;
    offset2 *= hi[i] - lo[i] + 1;
  }
  offset1 *= nrow;
  
  for(j=0; j<offset2; ++j,indexA=j,indexB=j,m=0) {
    for(i=0; i<nrow*ncol; i++, indexA += offset1, indexB += offset2) {
      if(indexA >= nelem) indexA = j + ++m*offset2;
      A[indexA] = alpha *(A[indexA] + B[indexB]);
    }
  }
}

void FATR 
ga_symmetrize_(Integer *g_a) {
  
  DoublePrecision alpha = 0.5;
  Integer i, me = ga_nodeid_();
  Integer alo[GA_MAX_DIM], ahi[GA_MAX_DIM], lda[GA_MAX_DIM], nelem=1;
  Integer blo[GA_MAX_DIM], bhi[GA_MAX_DIM], ldb[GA_MAX_DIM];
  Integer ndim, dims[GA_MAX_DIM], type;
  Logical have_data;
  Integer g_b; /* temporary global array (b = A') */
  Void *a_ptr, *b_ptr;
  Integer bindex;
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  GA_PUSH_NAME("nga_copy_patch");

  
  nga_inquire_internal_(g_a, &type, &ndim, dims);
  
  if (dims[ndim-1] != dims[ndim-2]) 
    ga_error("ga_sym: can only sym square matrix", 0L);
  
  /* Find the local distribution */
  nga_distribution_(g_a, &me, alo, ahi);
 
 
  have_data = ahi[0]>0;
  for(i=1; i<ndim; i++) have_data = have_data && ahi[i]>0;
  
  if(have_data) {
    nga_access_ptr(g_a, alo, ahi, &a_ptr, lda); 
    
    for(i=0; i<ndim; i++) nelem *= ahi[i]-alo[i] +1;
    if(!MA_push_get(MT_F_DBL, nelem, "v", &g_b, &bindex) ||
       !MA_get_pointer(g_b, &b_ptr)) 
      ga_error(" MA Failed: insufficient memory ", nelem);
    
    
    for(i=0; i<ndim-2; i++) {bhi[i]=ahi[i]; blo[i]=alo[i]; }
    
    /* switch rows and cols */
    blo[ndim-1]=alo[ndim-2];
    bhi[ndim-1]=ahi[ndim-2];
    blo[ndim-2]=alo[ndim-1];
    bhi[ndim-2]=ahi[ndim-1];

    for (i=0; i < ndim-1; i++) 
      ldb[i] = bhi[i] - blo[i] + 1; 
    nga_get_(g_a, blo, bhi, b_ptr, ldb);
  }
  ga_sync_(); 

  if(have_data) {
    gai_add(alo, ahi, a_ptr, b_ptr, alpha, type, nelem, ndim);
    nga_release_update_(g_a, alo, ahi);
    if (!MA_pop_stack(g_b)) ga_error("MA_pop_stack failed",0);
  }
  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
}
