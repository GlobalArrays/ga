
/**
 * Symmetrizes matrix A:  A := .5 * (A+A`)
 * diag(A) remains unchanged
 * 
 */

#include "global.h"
#include "globalp.h"
#include "macdecls.h"

#  define COPYINDEX_F2C(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[n-i-1]=(int)(farr)[i] -1;}

void FATR 
gai_add(Integer *lo, Integer *hi, Void *a, Void *b, DoublePrecision alpha,
	Integer type, Integer nelem, Integer ndim) {

  Integer alo[GA_MAX_DIM], ahi[GA_MAX_DIM];
  Integer f, i, j=0, k=0, m=0;
  Integer offset = 1; /* 2-d offset */
  Integer nrow, ncol, indexA=0;
  DoublePrecision *A = (DoublePrecision *)a, *B = (DoublePrecision*)b;
  
  COPYINDEX_F2C(lo, alo, ndim);
  COPYINDEX_F2C(hi, ahi, ndim); 
 
  nrow = ahi[ndim-2]-alo[ndim-2]+1;  
  ncol = ahi[ndim-1]-alo[ndim-1]+1;
  offset = nrow*ncol;

  for(f=0; f<nelem; f+=offset) {
    j=0; k=0; m=0;
    for(i=0; i<nrow*ncol; i++) {
	indexA=k+j*nrow;
	if(indexA>=nrow*ncol) {
	  j=0; k++; 
	  indexA=k+j*ncol;
	}
	A[f+indexA] = alpha *(A[f+indexA] + B[f+i]);
	j++;
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
  
  ga_sync_();
  GA_PUSH_NAME("nga_copy_patch");
  
  nga_inquire_internal_(g_a, &type, &ndim, dims);
  
  if (dims[ndim-1] != dims[ndim-2]) 
    ga_error("ga_sym: can only sym square matrix", 0L);
  
  /* Find the local distribution */
  nga_distribution_(g_a, &me, alo, ahi);
 
 
  have_data = ahi[0]>0;
  for(i=1; i<ndim; i++) have_data = have_data && ahi[i]>0;
  
  /*ga_print_(g_a);*/
  
  if(have_data) {
    nga_access_ptr(g_a, alo, ahi, &a_ptr, lda); 
    
    for(i=0; i<ndim; i++) nelem *= ahi[i]-alo[i] +1;
    if(!MA_push_get(MT_F_DBL, nelem, "v", &g_b, &bindex) ||
       !MA_get_pointer(g_b, &b_ptr)) 
      ga_error(" MA Failed: insufficient memory ", nelem);
    
    
    for(i=2; i<ndim; i++) {bhi[i]=ahi[i]; blo[i]=alo[i]; ldb[i]=lda[i]; }
    
    /* switch rows and cols */
    blo[0]=alo[1];
    bhi[0]=ahi[1];
    blo[1]=alo[0];
    bhi[1]=ahi[0];

    for (i=0; i < ndim-1; i++) 
      ldb[i] = bhi[i] - blo[i] + 1;/* as I switched rows and cols */
    nga_get_(g_a, blo, bhi, b_ptr, ldb);
  }
  ga_sync_(); // why ? check

  if(have_data) {
    gai_add(alo, ahi, a_ptr, b_ptr, alpha, type, nelem, ndim);
    nga_release_update_(g_a, alo, ahi);
    if (!MA_pop_stack(g_b)) ga_error("MA_pop_stack failed",0);
  }

  GA_POP_NAME;
  ga_sync_();
}

