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

#include "global.h"
#include "globalp.h"
#include <math.h>

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
static logical ngai_patch_intersect(Integer *lo, Integer *hi,
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
static logical ngai_comp_patch(Integer andim, Integer *alo, Integer *ahi,
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
static logical ngai_test_shape(Integer *alo, Integer *ahi, Integer *blo,
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
    Integer ld[MAXDIM], ald[MAXDIM], bld[MAXDIM];
    Integer lod[MAXDIM], hid[MAXDIM];
    Integer src_hdl, src_idx, dst_hdl, dst_idx, vhandle, vindex;
    void *src_data_ptr, *tmp_ptr;
    Integer *src_idx_ptr, *dst_idx_ptr;
    Integer bvalue[MAXDIM], bunit[MAXDIM];
    Integer factor_idx1[MAXDIM], factor_idx2[MAXDIM], factor_data[MAXDIM];
    Integer base;
    Integer me= ga_nodeid_();
    
    ga_sync_();
    
    GA_PUSH_NAME("nga_copy_patch");
    
    nga_inquire_(g_a, &atype, &andim, adims);
    nga_inquire_(g_b, &btype, &bndim, bdims);
    
    if(*g_a == *g_b)
        /* they are the same patch */
        if(ngai_comp_patch(andim, alo, ahi, bndim, blo, bhi)) return;
        /* they are in the same GA, but not the same patch */
        else
            ga_error("arrays have to be different ", 0L);

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
    
    /* now find out cordinates of a patch of g_a that I own */
    nga_distribution_(g_a, &me, los, his);
    
    /* copy my share of data */
    if(ngai_patch_intersect(alo, ahi, los, his, andim)){
        nga_access_ptr(g_a, los, his, &src_data_ptr, ld); 
        
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
            ngai_dest_indices(andim, los, alo, ald, bndim, lod, blo, bld);
            ngai_dest_indices(andim, his, alo, ald, bndim, hid, blo, bld);
            nga_put_(g_b, lod, hid, src_data_ptr, ld);
        }
        /*** due to generality of this transformation scatter is required ***/
        else{
            if(!MA_push_get(atype, nelem, "v", &vhandle, &vindex) ||
               !MA_get_pointer(vhandle, &tmp_ptr))
                ga_error(" MA failed-v ", 0L);
            if(!MA_push_get(MT_F_INT, (andim*nelem), "si", &src_hdl, &src_idx)
               || !MA_get_pointer(src_hdl, &src_idx_ptr))
                ga_error(" MA failed-si ", 0L);
            if(!MA_push_get(MT_F_INT, (bndim*nelem), "di", &dst_hdl, &dst_idx)
               || !MA_get_pointer(dst_hdl, &dst_idx_ptr))
                ga_error(" MA failed-di ", 0L);
                
            /* calculate the destination indices */

            /* given los and his, find indices for each elements
             * bvalue: starting index in each dimension
             * bunit: stride in each dimension
             */
            for(i=0; i<andim; i++) {
                bvalue[i] = los[i];
                if(i == 0) bunit[i] = 1;
                else bunit[i] = bunit[i-1] * (his[i-1] - los[i-1] + 1);
            }

            /* source indices */
            for(i=0; i<nelem; i++) {
                for(j=0; j<andim; j++){
                    src_idx_ptr[i*andim+j] = bvalue[j];
                    /* if the next element is the first element in
                     * one dimension, increment the index by 1
                     */
                    if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                    /* if the index becomes larger than the upper
                     * bound in one dimension, reset it.
                     */
                    if(bvalue[j] > his[j]) bvalue[j] = los[j];
                }
            }

            /* index factor: reshaping without transpose */
            factor_idx1[0] = 1;
            for(j=1; j<andim; j++) 
                factor_idx1[j] = factor_idx1[j-1] * ald[j-1];

            /* index factor: reshaping with transpose */
            factor_idx2[andim-1] = 1;
            for(j=(andim-1)-1; j>=0; j--)
                factor_idx2[j] = factor_idx2[j+1] * ald[j+1];

            /* data factor */
            factor_data[0] = 1;
            for(j=1; j<andim; j++) 
                factor_data[j] = factor_data[j-1] * ld[j-1];
            
            /* destination indices */
            for(i=0; i<nelem; i++) {
                /* linearize the n-dimensional indices to one dimension */
                idx = 0;
                if (*trans == 'n' || *trans == 'N')
                    for(j=0; j<andim; j++) 
                        idx += (src_idx_ptr[i*andim+j] - alo[j]) *
                            factor_idx1[j];
                else
                    /* if the patch needs to be transposed, reverse
                     * the indices: (i, j, ...) -> (..., j, i)
                     */
                    for(j=(andim-1); j>=0; j--) 
                        idx += (src_idx_ptr[i*andim+j] - alo[j]) *
                            factor_idx2[j];

                /* convert the one dimensional index to n-dimensional
                 * indices of destination
                 */
                for(j=0; j<bndim; j++) {
                    dst_idx_ptr[i*bndim+j] = idx % bld[j] + blo[j];
                    idx /= bld[j];
                }
                
                /* move the data block to create a new block */
                /* linearize the data indices */
                idx = 0;
                for(j=0; j<andim; j++) 
                    idx += (src_idx_ptr[i*andim+j]) * factor_data[j];

                /* adjust the postion
                 * base: starting address of the first element */
                idx -= base;

                /* move the element to the temporary location */
                switch(atype) {
                    case MT_F_DBL: ((DoublePrecision *)tmp_ptr)[i] =
                                       ((DoublePrecision *)src_data_ptr)[idx]; 
                    break;
                    case MT_F_INT: ((Integer *)tmp_ptr)[i] =
                                       ((Integer *)src_data_ptr)[idx];
                    break;
                    case MT_F_DCPL:((DoubleComplex *)tmp_ptr)[i] =
                                       ((DoubleComplex *)src_data_ptr)[idx];
                }
            }

            nga_release_(g_a, los, his);
            nga_scatter_(g_b, tmp_ptr, dst_idx_ptr, &nelem);
            if (!MA_pop_stack(dst_hdl) || !MA_pop_stack(src_hdl) ||
                !MA_pop_stack(vhandle)) ga_error("MA_pop_stack failed",0);
        }
    }
    GA_POP_NAME;
    ga_sync_();
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
    Integer idx, n1dim;
    Integer atotal, btotal;
    Integer isum;
    DoublePrecision dsum;
    DoubleComplex zsum;
    Integer me= ga_nodeid_(), temp_created=0;
    Integer type = GA_TYPE_GSM, len = 1;
    char *tempname = "temp", transp, transp_a, transp_b;

    ga_sync_();
    GA_PUSH_NAME("ngai_dot_patch");

    nga_inquire_(g_a, &atype, &andim, adims);
    nga_inquire_(g_b, &btype, &bndim, bdims);
    
    if(atype != btype ) ga_error(" type mismatch ", 0L);
    if((atype != MT_F_INT ) && (atype != MT_F_DBL ) && (atype != MT_F_DCPL))
        ga_error(" wrong type", 0L);
    
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

    isum = 0; dsum = 0.; zsum.real = 0.; zsum.imag = 0.;
    
    /*  determine subsets of my patches to access  */
    if(ngai_patch_intersect(alo, ahi, loA, hiA, andim)){
        nga_access_ptr(&g_A, loA, hiA, &A_ptr, ldA);
        nga_access_ptr(&g_B, loA, hiA, &B_ptr, ldB);

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
            case MT_F_INT:
                for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<andim; j++) {
                        idx += bvalue[j] * baseldA[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
                    }
                    
                    for(j=0; j<(hiA[0]-loA[0]+1); j++)
                        isum += ((Integer *)A_ptr)[idx+j] *
                            ((Integer *)B_ptr)[idx+j];
                }
                break;
            case MT_F_DCPL:
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
                break;
            case  MT_F_DBL:
                for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<andim; j++) {
                        idx += bvalue[j] * baseldA[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
                    }
                    
                    for(j=0; j<(hiA[0]-loA[0]+1); j++)
                        dsum += ((DoublePrecision *)A_ptr)[idx+j] *
                            ((DoublePrecision *)B_ptr)[idx+j];
                }
        }

        /* release access to the data */
        nga_release_(&g_A, loA, hiA);
        nga_release_(&g_B, loA, hiA);
    }

    ga_sync_();

    /* the return value */
    switch (atype){
        case MT_F_INT:
            ga_igop(type, &isum, 1, "+");
            *((Integer *)retval) += isum;
            break;
        case  MT_F_DBL:
            ga_dgop(type, &dsum, 1, "+");
            *((DoublePrecision *)retval) = dsum;
            break;
        case MT_F_DCPL:
            ga_dgop(type, &zsum.real, 1, "+");
            ga_dgop(type, &zsum.imag, 1, "+");
            (*((DoubleComplex *)retval)).real = zsum.real;
            (*((DoubleComplex *)retval)).imag = zsum.imag;
    }
    
    if(temp_created) ga_destroy_(&g_B);
    GA_POP_NAME;
    ga_sync_();
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
    
    ga_sync_();
    GA_PUSH_NAME("nga_idot_patch");
    
    ga_inquire_(g_a, &atype, &andim, adims);
    ga_inquire_(g_b, &btype, &bndim, bdims);

    if(atype != btype || (atype != MT_F_INT )) ga_error(" wrong types ", 0L);

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
    
    ga_sync_();
    GA_PUSH_NAME("nga_ddot_patch");
    
    ga_inquire_(g_a, &atype, &andim, adims);
    ga_inquire_(g_b, &btype, &bndim, bdims);

    if(atype != btype || (atype != MT_F_DBL )) ga_error(" wrong types ", 0L);

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

   ga_sync_();
   GA_PUSH_NAME("nga_zdot_patch");

   ga_inquire_(g_a, &atype, &andim, adims);
   ga_inquire_(g_b, &btype, &bndim, bdims);

   if(atype != btype || (atype != MT_F_DCPL )) ga_error(" wrong types ", 0L);

   ngai_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi,
                  (void *)(&sum));

   GA_POP_NAME;
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

/*\ FILL IN ARRAY WITH VALUE 
\*/
void FATR nga_fill_patch_(Integer *g_a, Integer *lo, Integer *hi, void* val)
{
    Integer i, j;
    Integer ndim, dims[MAXDIM], type;
    Integer loA[MAXDIM], hiA[MAXDIM], ld[MAXDIM];
    void *data_ptr;
    Integer idx, factor, n1dim;
    Integer bvalue[MAXDIM], bunit[MAXDIM], baseld[MAXDIM];
    Integer me= ga_nodeid_();
    
    ga_sync_();
    GA_PUSH_NAME("nga_fill_patch");
    
    nga_inquire_(g_a,  &type, &ndim, dims);
    
    nga_distribution_(g_a, &me, loA, hiA);
    
    /*  determine subset of my patch to access  */
    if(ngai_patch_intersect(lo, hi, loA, hiA, ndim)){
        nga_access_ptr(g_a, loA, hiA, &data_ptr, ld);
 
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
            case MT_F_INT:
                for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<ndim; j++) {
                        idx += bvalue[j] * baseld[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
                    }
                    for(j=0; j<(hiA[0]-loA[0]+1); j++)
                        ((Integer *)data_ptr)[idx+j] = *(Integer*)val;
                }
                break;
            case MT_F_DCPL:
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
            case MT_F_DBL:
                for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<ndim; j++) {
                        idx += bvalue[j] * baseld[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
                    }
                    
                    for(j=0; j<(hiA[0]-loA[0]+1); j++) 
                        ((DoublePrecision *)data_ptr)[idx+j] =
                            *(DoublePrecision *)val;
                }
                break;
            default: ga_error(" wrong data type ",type);
        }
        
        /* release access to the data */
        nga_release_update_(g_a, loA, hiA);
    }
    GA_POP_NAME;
    ga_sync_();
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
    Integer idx, factor, n1dim;
    Integer bvalue[MAXDIM], bunit[MAXDIM], baseld[MAXDIM];
    DoublePrecision tmp1_real, tmp1_imag, tmp2_real, tmp2_imag;
    Integer me= ga_nodeid_();

    ga_sync_();
    GA_PUSH_NAME("nga_scal_patch");
    
    nga_inquire_(g_a,  &type, &ndim, dims);
    nga_distribution_(g_a, &me, loA, hiA);
    
    /* determine subset of my patch to access */
    if (ngai_patch_intersect(lo, hi, loA, hiA, ndim)){
        nga_access_ptr(g_a, loA, hiA, &src_data_ptr, ld);

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
            case MT_F_DBL:
                for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<ndim; j++) {
                        idx += bvalue[j] * baseld[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
                    }
                    
                    for(j=0; j<(hiA[0]-loA[0]+1); j++) 
                        ((DoublePrecision *)src_data_ptr)[idx+j]  *=
                            *(DoublePrecision*)alpha;                    
                }
                break;
            case MT_F_DCPL:
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
            case MT_F_INT:
                for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<ndim; j++) {
                        idx += bvalue[j] * baseld[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
                    }
                    
                    for(j=0; j<(hiA[0]-loA[0]+1); j++)
                        ((Integer *)src_data_ptr)[idx+j]  *= *(Integer*)alpha;
                }
        }

        /* release access to the data */
        nga_release_update_(g_a, loA, hiA); 
    }
    GA_POP_NAME;
    ga_sync_();
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
    Integer me= ga_nodeid_(), B_created=0, A_created=0;
    char *tempname = "temp", notrans='n';

    ga_sync_();
    GA_PUSH_NAME("nga_add_patch");

    nga_inquire_(g_a, &atype, &andim, adims);
    nga_inquire_(g_b, &btype, &bndim, bdims);
    nga_inquire_(g_c, &ctype, &cndim, cdims);

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
        nga_copy_patch(&notrans, g_a, alo, ahi, g_c, clo, chi);
        andim = cndim;
        g_A = *g_c;
        A_created = 1;
        nga_distribution_(&g_A, &me, loA, hiA);
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
        
        /* compute "local" add */

        /* number of n-element of the first dimension */
        n1dim = 1; for(i=1; i<cndim; i++) n1dim *= (hiC[i] - loC[i] + 1);

        /* calculate the destination indices */
        bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
        /* baseld[0] = ld[0]
         * baseld[1] = ld[0] * ld[1]
         * baseld[2] = ld[0] * ld[1] * ld[2] .....
         */
        baseldC[0] = ldC[0]; baseldC[1] = baseldC[0] *ldC[1];
        for(i=2; i<cndim; i++) {
            bvalue[i] = 0;
            bunit[i] = bunit[i-1] * (hiC[i-1] - loC[i-1] + 1);
            baseldC[i] = baseldC[i-1] * ldC[i];
        }
        
        switch(atype){
            case MT_F_DBL:
                for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<cndim; j++) {
                        idx += bvalue[j] * baseldC[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
                    }
                    
                    for(j=0; j<(hiC[0]-loC[0]+1); j++)
                        ((DoublePrecision*)C_ptr)[idx+j] =
                            *(DoublePrecision*)alpha *
                            ((DoublePrecision*)A_ptr)[idx+j] +
                            *(DoublePrecision*)beta *
                            ((DoublePrecision*)B_ptr)[idx+j];
                }
                break;
            case MT_F_DCPL:
                for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<cndim; j++) {
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
            case MT_F_INT:
                for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<cndim; j++) {
                        idx += bvalue[j] * baseldC[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
                    }
                    
                    for(j=0; j<(hiC[0]-loC[0]+1); j++)
                        ((Integer *)C_ptr)[idx+j] = *(Integer *)alpha *
                            ((Integer *)A_ptr)[idx+j] + *(Integer *)beta *
                            ((Integer *)B_ptr)[idx+j];
                }
                break;
        }
        
        /* release access to the data */
        nga_release_       (&g_A, loC, hiC);
        nga_release_       (&g_B, loC, hiC); 
        nga_release_update_( g_c, loC, hiC); 

    }
    
    if(B_created) ga_destroy_(&g_B);
    
    GA_POP_NAME;
    ga_sync_();
}

