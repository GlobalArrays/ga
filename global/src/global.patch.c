/*$Id: global.patch.c,v 1.10 1995-03-29 23:40:34 d3h325 Exp $*/
#include "global.h"
#include "globalp.h"
#include "macommon.h"

#ifdef KSR
#  define dgemm_ sgemm_
#endif

#ifdef CRAY_T3D
#      include <fortran.h>
#      define cptofcd(fcd)  _cptofcd((fcd),1)
#else
#      define cptofcd(fcd) (fcd)
#endif


#define DEST_INDICES(is,js, ilos,jlos, lds, id,jd, ilod, jlod, ldd) \
{ \
    int _index_;\
    _index_ = (lds)*((js)-(jlos)) + (is)-(ilos);\
    id = (_index_)%(ldd) + (ilod);\
    jd = (_index_)/(ldd) + (jlod);\
}


/*\ check if dimensions of two patches are divisible 
\*/
logical patches_conforming(ailo, aihi, ajlo, ajhi, bilo, bihi, bjlo, bjhi)
     Integer *ailo, *aihi, *ajlo, *ajhi;
     Integer *bilo, *bihi, *bjlo, *bjhi;
{
Integer mismatch;
Integer adim1, bdim1, adim2, bdim2;
     adim1 = *aihi - *ailo +1;
     adim2 = *ajhi - *ajlo +1;
     bdim1 = *bihi - *bilo +1;
     bdim2 = *bjhi - *bjlo +1;
     mismatch  = (adim1<bdim1) ? bdim1%adim1 : adim1%bdim1; 
     mismatch += (adim2<bdim2) ? bdim2%adim2 : adim2%bdim2; 
     if(mismatch)return(FALSE);
     else return(TRUE);
} 



/*\ check if patches are identical 
\*/
static logical comp_patch(ilo, ihi, jlo, jhi, ilop, ihip, jlop, jhip)
     Integer ilo, ihi, jlo, jhi;
     Integer ilop, ihip, jlop, jhip;
{
   if(ihip != ihi || ilop != ilo || jhip != jhi || jlop != jlo) return(FALSE);
   else return(TRUE); 
}


static logical patch_intersect(ilo, ihi, jlo, jhi, ilop, ihip, jlop, jhip)
     Integer *ilo, *ihi, *jlo, *jhi;
     Integer *ilop, *ihip, *jlop, *jhip;
{
     /* check consistency of patch coordinates */
     if( *ihi < *ilo || *jhi < *jlo)     return FALSE; /* inconsistent */
     if( *ihip < *ilop || *jhip < *jlop) return FALSE; /* inconsistent */

     /* find the intersection and update (ilop: ihip, jlop: jhip) */
     if( *ihi < *ilop || *ihip < *ilo) return FALSE; /* don't intersect */
     if( *jhi < *jlop || *jhip < *jlo) return FALSE; /* don't intersect */
     *ilop = MAX(*ilo,*ilop);
     *ihip = MIN(*ihi,*ihip);
     *jlop = MAX(*jlo,*jlop);
     *jhip = MIN(*jhi,*jhip);

     return TRUE;
}


/*\ COMPARE DISTRIBUTIONS of two global arrays
\*/
logical ga_compare_distr_(g_a, g_b)
     Integer *g_a, *g_b;
{
Integer me= ga_nodeid_();
Integer iloA, ihiA, jloA, jhiA;
Integer iloB, ihiB, jloB, jhiB;
DoublePrecision mismatch;
Integer type = GA_TYPE_GSM, len = 1;

   ga_sync_();
   GA_PUSH_NAME("ga_compare_distr");

   ga_distribution_(g_a, &me, &iloA, &ihiA, &jloA, &jhiA);
   ga_distribution_(g_b, &me, &iloB, &ihiB, &jloB, &jhiB);

   mismatch = comp_patch(iloA, ihiA, jloA, jhiA, iloB, ihiB, jloB, jhiB)?0. :1.;
   ga_dgop_(&type, &mismatch, &len, "+",1);
   ga_sync_();
   GA_POP_NAME;

   if(mismatch) return (FALSE);
   else return(TRUE); 
}



/*\ COPY A PATCH AND POSSIBLY RESHAPE
 *
 *  . the element capacities of two patches must be identical
 *  . copy by column order - Fortran convention
\*/
void ga_copy_patch(trans, g_a, ailo, aihi, ajlo, ajhi,
                   g_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;
     char    *trans;
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
Integer ilos, ihis, jlos, jhis;
Integer ilod, ihid, jlod, jhid;
Integer me= ga_nodeid_(), index, ld, i,j;
Integer ihandle, jhandle, vhandle, iindex, jindex, vindex, nelem, base, ii, jj; 

   ga_sync_();

   GA_PUSH_NAME("ga_copy_patch");
   if(*g_a == *g_b) ga_error(" arrays have to different ", 0L);

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype ) ga_error(" array type mismatch ", 0L);

   /* check if patch indices and dims match */
   if (*ailo <= 0 || *aihi > adim1 || *ajlo <= 0 || *ajhi > adim2)
       ga_error(" g_a indices out of range ", 0L);
   if (*bilo <= 0 || *bihi > bdim1 || *bjlo <= 0 || *bjhi > bdim2)
       ga_error(" g_b indices out of range ", 0L);

   /* check if numbers of elements in two patches match each other */
   if ((*bihi - *bilo + 1) * (*bjhi - *bjlo + 1) !=
       (*aihi - *ailo + 1) * (*ajhi - *ajlo + 1))
       ga_error(" capacities two of patches do not match ", 0L);

   /* now find out cordinates of a patch of g_a that I own */
   ga_distribution_(g_a, &me, &ilos, &ihis, &jlos, &jhis);

   /* copy my share of data */
   if(patch_intersect(ailo, aihi, ajlo, ajhi, &ilos, &ihis, &jlos, &jhis)){
      ga_access_(g_a, &ilos, &ihis, &jlos, &jhis, &index, &ld);
      nelem = (ihis-ilos+1)*(jhis-jlos+1);
      index --;     /* fortran to C conversion */

      if((*trans == 'n' || *trans == 'N') && (*ailo- *aihi ==  *bilo- *bihi)){ 
         /*** straight copy possible if there's no reshaping or transpose ***/
         
         /* find source[ilo:ihi, jlo:jhi] --> destination[ilo:ihi, jlo:jhi] */
         DEST_INDICES(ilos, jlos, *ailo, *ajlo, (*aihi - *ailo +1),
                      ilod, jlod, *bilo, *bjlo, (*bihi - *bilo +1) );
         DEST_INDICES(ihis, jhis, *ailo, *ajlo, (*aihi - *ailo +1),
                      ihid, jhid, *bilo, *bjlo, (*bihi - *bilo +1) );

         ga_put_(g_b, &ilod, &ihid, &jlod, &jhid, DBL_MB+index, &ld); 

      }else{
        /*** due to generality of this transformation scatter is required ***/

         if(!MA_push_get(MT_F_INT, nelem, "i", &ihandle, &iindex))
            ga_error(" MA failed-i ", 0L);
         if(!MA_push_get(MT_F_INT, nelem, "j", &jhandle, &jindex))
            ga_error(" MA failed-j ", 0L);
         if(!MA_push_get(atype, nelem, "v", &vhandle, &vindex))
            ga_error(" MA failed-v ", 0L);

         base = 0;
         if(atype == MT_F_DBL ){
           if (*trans == 'n' || *trans == 'N')  
             for(j = jlos, jj=0; j <= jhis; j++, jj++)
               for(i = ilos, ii =0; i <= ihis; i++, ii++){
                   DEST_INDICES(i, j, *ailo, *ajlo, (*aihi- *ailo +1), 
                             INT_MB[base+iindex], INT_MB[base+jindex],
                             *bilo, *bjlo, (*bihi - *bilo +1) );
                   DBL_MB[base+vindex] = DBL_MB[index+ ld*jj + ii];
                   base++;
               }
           else
             for(j = jlos, jj=0; j <= jhis; j++, jj++)
               for(i = ilos, ii =0; i <= ihis; i++, ii++){
                   DEST_INDICES(j, i, *ajlo, *ailo, (*ajhi - *ajlo +1),
                                INT_MB[base+iindex], INT_MB[base+jindex],
                                *bilo, *bjlo, (*bihi - *bilo +1) );
                   DBL_MB[base+vindex] = DBL_MB[index+ ld*jj + ii];
                   base++;
               }
         }else{
           if (*trans == 'n' || *trans == 'N')
             for(j = jlos, jj=0; j <= jhis; j++, jj++)
               for(i = ilos, ii =0; i <= ihis; i++, ii++){
                   DEST_INDICES(i, j, *ailo, *ajlo, (*aihi- *ailo +1),
                             INT_MB[base+iindex], INT_MB[base+jindex],
                             *bilo, *bjlo, (*bihi - *bilo +1) );
                   INT_MB[base+vindex] = INT_MB[index+ ld*jj + ii];
                   base++;
               }
           else
             for(j = jlos, jj=0; j <= jhis; j++, jj++)
               for(i = ilos, ii =0; i <= ihis; i++, ii++){
                   DEST_INDICES(j, i, *ajlo, *ailo, (*ajhi - *ajlo +1),
                                INT_MB[base+iindex], INT_MB[base+jindex],
                                *bilo, *bjlo, (*bihi - *bilo +1) );
                   INT_MB[base+vindex] = INT_MB[index+ ld*jj + ii];
                   base++;
               }
         }

         ga_release_(g_a, &ilos, &ihis, &jlos, &jhis);
         ga_scatter_(g_b, DBL_MB+vindex, INT_MB+iindex, INT_MB+jindex, &nelem);
         MA_pop_stack(vhandle);
         MA_pop_stack(jhandle);
         MA_pop_stack(ihandle);
      }
  }
  GA_POP_NAME;
  ga_sync_();
}


/*\ COPY A PATCH AND POSSIBLY RESHAPE
 *  Fortran interface
\*/
void ga_copy_patch_(trans, g_a, ailo, aihi, ajlo, ajhi,
                    g_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;
#ifdef CRAY_T3D
     _fcd    trans;
{ga_copy_patch(_fcdtocp(trans),g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi);}
#else 
     char*   trans;
{  ga_copy_patch(trans,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi); }
#endif



/*\ compute DOT PRODUCT of two patches
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
Integer iloA, ihiA, jloA, jhiA, indexA, ldA;
Integer iloB, ihiB, jloB, jhiB, indexB, ldB;
Integer g_A = *g_a, g_B = *g_b;
Integer me= ga_nodeid_(), i, j, temp_created=0;
Integer type = GA_TYPE_GSM, len = 1;
char *tempname = "temp", transp, transp_a, transp_b;
DoublePrecision  sum = 0.;

   ga_sync_();
   GA_PUSH_NAME("ga_ddot_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != MT_F_DBL )) ga_error(" wrong types ", 0L); 

  /* check if patch indices and g_a dims match */ 
   if (*ailo <= 0 || *aihi > adim1 || *ajlo <= 0 || *ajhi > adim2) 
      ga_error(" g_a indices out of range ", *g_a);

   /* check if patch indices and g_b dims match */
   if (*bilo <= 0 || *bihi > bdim1 || *bjlo <= 0 || *bjhi > bdim2)
       ga_error(" g_b indices out of range ", *g_b);

   /* check if numbers of elements in two patches match each other */
   if ((*bihi - *bilo + 1) * (*bjhi - *bjlo + 1) !=
       (*aihi - *ailo + 1) * (*ajhi - *ajlo + 1))
       ga_error(" capacities of two patches do not match ", 0L);

   /* is transpose operation required ? */
   /* -- only if for one array transpose operation requested*/
   transp_a = (*t_a == 'n' || *t_a =='N')? 'n' : 't'; 
   transp_b = (*t_b == 'n' || *t_b =='N')? 'n' : 't'; 
   transp   = (transp_a == transp_b)? 'n' : 't'; 

   /* compare patches and distributions of g_a and g_b */
   if( !(comp_patch(*ailo, *aihi, *ajlo, *ajhi, *bilo, *bihi, *bjlo, *bjhi) &&
         ga_compare_distr_(g_a, g_b) && (transp=='n') ) ){

         /* either patches or distributions do not match:
          *        - create a temp array that matches distribution of g_a
          *        - copy & reshape patch of g_b into g_B
          */
         ga_duplicate_(g_a, &g_B, tempname, sizeof(tempname));
         ga_copy_patch(&transp, g_b, bilo, bihi, bjlo, bjhi,
                               &g_B, ailo, aihi, ajlo, ajhi);  
         temp_created = 1;
   }

   /* since patches and distributions of g_A and g_B match each other dot them*/
 
   /* find out coordinates of patches of g_A and g_B that I own */
   ga_distribution_(&g_A, &me, &iloA, &ihiA, &jloA, &jhiA);
   ga_distribution_(&g_B, &me, &iloB, &ihiB, &jloB, &jhiB);

   if( ! comp_patch(iloA, ihiA, jloA, jhiA, iloB, ihiB, jloB, jhiB))
         ga_error(" patches mismatch ",0); 

   /*  determine subsets of my patches to access  */
   if (patch_intersect(ailo, aihi, ajlo, ajhi, &iloA, &ihiA, &jloA, &jhiA)){
       ga_access_(&g_A, &iloA, &ihiA, &jloA, &jhiA, &indexA, &ldA);
       ga_access_(&g_B, &iloA, &ihiA, &jloA, &jhiA, &indexB, &ldB);
       indexA --; indexB--;   /* Fortran to C correction of starting address */

      /* compute "local" contribution to the dot product */
       for(j=0; j < jhiA - jloA+1; j++)
          for(i=0; i< ihiA - iloA +1; i++)
             sum += DBL_MB[indexA +j*ldA + i]  *
                    DBL_MB[indexB +j*ldB + i];

       /* release access to the data */
        ga_release_(&g_A, &iloA, &ihiA, &jloA, &jhiA);
        ga_release_(&g_B, &iloA, &ihiA, &jloA, &jhiA); 
   }

   ga_dgop_(&type, &sum, &len, "+",1);
   ga_sync_();
   if(temp_created) ga_destroy_(&g_B);

   GA_POP_NAME;
   return (sum);
}



/*\ compute DOT PRODUCT of two patches
 *  Fortran interface
\*/
DoublePrecision ga_ddot_patch_(g_a, t_a, ailo, aihi, ajlo, ajhi,
                               g_b, t_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */

#ifdef CRAY_T3D
     _fcd   t_a, t_b;                          /* transpose operators */
{ return ga_ddot_patch(g_a, _fcdtocp(t_a), ailo, aihi, ajlo, ajhi,
                       g_b, _fcdtocp(t_b), bilo, bihi, bjlo, bjhi);}
#else 
     char    *t_a, *t_b;                          /* transpose operators */
{ return ga_ddot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                       g_b, t_b, bilo, bihi, bjlo, bjhi);}
#endif


/*\ FILL IN ARRAY WITH VALUE  (integer version) 
\*/
void ga_ifill_patch_(g_a, ilo, ihi, jlo, jhi, val)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi, *val;
{
Integer dim1, dim2, type;
Integer iloA, ihiA, jloA, jhiA, index, ld;
Integer me= ga_nodeid_(), i, j;

   ga_sync_();
   GA_PUSH_NAME("ga_ifill_patch");

   ga_inquire_(g_a,  &type, &dim1, &dim2);
   if(type != MT_F_INT) ga_error(" wrong array type ", 0L);
   ga_distribution_(g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

   /*  determine subset of my patch to access  */
   if (patch_intersect(ilo, ihi, jlo, jhi, &iloA, &ihiA, &jloA, &jhiA)){
       ga_access_(g_a, &iloA, &ihiA, &jloA, &jhiA, &index, &ld);
       index --;  /* Fortran to C correction of starting address */

       for(j=0; j<jhiA-jloA+1; j++)
            for(i=0; i<ihiA-iloA+1; i++)
                 INT_MB[index +j*ld + i ]  = *val;

       /* release access to the data */
        ga_release_update_(g_a, &iloA, &ihiA, &jloA, &jhiA);
   }

   GA_POP_NAME;
   ga_sync_();
}



/*\ FILL IN ARRAY WITH VALUE  (DP version)
\*/
void ga_dfill_patch_(g_a, ilo, ihi, jlo, jhi, val)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi; 
     DoublePrecision     *val;
{
Integer dim1, dim2, type;
Integer iloA, ihiA, jloA, jhiA, index, ld;
Integer me= ga_nodeid_(), i, j;

   ga_sync_();
   GA_PUSH_NAME("ga_dfill_patch");

   ga_inquire_(g_a,  &type, &dim1, &dim2);
   if(type != MT_F_DBL) ga_error("wrong array type ", *g_a);
   ga_distribution_(g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

   /*  determine subset of my patch to access  */
   if (patch_intersect(ilo, ihi, jlo, jhi, &iloA, &ihiA, &jloA, &jhiA)){
       ga_access_(g_a, &iloA, &ihiA, &jloA, &jhiA, &index, &ld);
       index --;  /* Fortran to C correction of starting address */

       for(j=0; j<jhiA-jloA+1; j++)
            for(i=0; i<ihiA-iloA+1; i++)
                 DBL_MB[index +j*ld + i ]  = *val;

       /* release access to the data */
        ga_release_update_(g_a, &iloA, &ihiA, &jloA, &jhiA);
   }
   GA_POP_NAME;
   ga_sync_();
}



/*\ SCALE ARRAY 
\*/
void ga_dscal_patch_(g_a, ilo, ihi, jlo, jhi, alpha)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
     DoublePrecision     *alpha;
{
Integer dim1, dim2, type;
Integer iloA, ihiA, jloA, jhiA, index, ld, i, j;
Integer me= ga_nodeid_();

   ga_sync_();
   GA_PUSH_NAME("ga_dscal_patch");

   ga_inquire_(g_a,  &type, &dim1, &dim2);
   if(type != MT_F_DBL) ga_error("wrong array type ", *g_a);
   ga_distribution_(g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

   /*  determine subset of my patch to access  */
   if (patch_intersect(ilo, ihi, jlo, jhi, &iloA, &ihiA, &jloA, &jhiA)){
       ga_access_(g_a, &iloA, &ihiA, &jloA, &jhiA, &index, &ld);
       index --;  /* Fortran to C correction of starting address */

       for(j=0; j<jhiA-jloA+1; j++)
            for(i=0; i<ihiA-iloA+1; i++)
                 DBL_MB[index +j*ld + i ]  *= *alpha;

       /* release access to the data */
        ga_release_update_(g_a, &iloA, &ihiA, &jloA, &jhiA);
   }
   GA_POP_NAME;
   ga_sync_();
}


/*\  SCALED ADDITION of two patches
\*/
void ga_dadd_patch_(alpha, g_a, ailo, aihi, ajlo, ajhi,
                    beta,  g_b, bilo, bihi, bjlo, bjhi,
                           g_c, cilo, cihi, cjlo, cjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     DoublePrecision      *alpha, *beta;
{
Integer atype, btype, ctype, adim1, adim2, bdim1, bdim2, cdim1, cdim2;
Integer iloA, ihiA, jloA, jhiA, indexA, ldA;
Integer iloB, ihiB, jloB, jhiB, indexB, ldB;
Integer iloC, ihiC, jloC, jhiC, indexC, ldC;
Integer g_A = *g_a, g_B = *g_b;
Integer me= ga_nodeid_(), i, j, B_created=0, A_created=0;
Integer nelem;
char *tempname = "temp", notrans='n';

   ga_sync_();
   GA_PUSH_NAME("ga_dadd_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);
   ga_inquire_(g_c, &ctype, &cdim1, &cdim2);

   if(atype != btype || atype != ctype || atype != MT_F_DBL)
               ga_error(" types mismatch ", 0L); 

  /* check if patch indices and dims match */ 
   if (*ailo <= 0 || *aihi > adim1 || *ajlo <= 0 || *ajhi > adim2) 
       ga_error("  g_a indices out of range ", *g_a);
   if (*bilo <= 0 || *bihi > bdim1 || *bjlo <= 0 || *bjhi > bdim2)
       ga_error("  g_b indices out of range ", *g_b);
   if (*cilo <= 0 || *cihi > cdim1 || *cjlo <= 0 || *cjhi > cdim2)
       ga_error("  g_c indices out of range ", *g_c);

   /* check if numbers of elements in patches match each other */
   nelem = (*cihi - *cilo + 1) * (*cjhi - *cjlo + 1);
   if ((*bihi - *bilo + 1) * (*bjhi - *bjlo + 1) != nelem ||
       (*aihi - *ailo + 1) * (*ajhi - *ajlo + 1) != nelem )
       ga_error("  capacities of patches do not match ", 0L);

   /* compare patches and distributions of g_a and g_c */
   if( !(comp_patch(*ailo, *aihi, *ajlo, *ajhi, *cilo, *cihi, *cjlo, *cjhi) &&
         ga_compare_distr_(g_a, g_c) ) ){

         /* either patches or distributions do not match:
          *        - create a temp array that matches distribution of g_c
          *        - copy & reshape patch of g_a into g_A
          */
         ga_duplicate_(g_c, &g_A, tempname, sizeof(tempname));
         ga_copy_patch(&notrans, g_a, ailo, aihi, ajlo, ajhi,
                                &g_A, cilo, cihi, cjlo, cjhi);  
         A_created = 1;
   }

   /* compare patches and distributions of g_b and g_c */
   if( !(comp_patch(*bilo, *bihi, *bjlo, *bjhi, *cilo, *cihi, *cjlo, *cjhi) &&
         ga_compare_distr_(g_b, g_c) ) ){

         /* either patches or distributions do not match:
          *        - create a temp array that matches distribution of g_c
          *        - copy & reshape patch of g_b into g_B
          */
         ga_duplicate_(g_c, &g_B, tempname, sizeof(tempname));
         ga_copy_patch(&notrans, g_b, bilo, bihi, bjlo, bjhi,
                                &g_B, cilo, cihi, cjlo, cjhi);  
         B_created = 1;
   }

   /* since patches and distributions of g_A and g_B match g_c, add them*/
 
   /* find out coordinates of patches of g_A and g_B that I own */
   ga_distribution_(&g_A, &me, &iloA, &ihiA, &jloA, &jhiA);
   ga_distribution_(&g_B, &me, &iloB, &ihiB, &jloB, &jhiB);
   ga_distribution_( g_c, &me, &iloC, &ihiC, &jloC, &jhiC);

   if( ! comp_patch(iloA, ihiA, jloA, jhiA, iloC, ihiC, jloC, jhiC))
         ga_error(" A patch mismatch ",g_A); 
   if( ! comp_patch(iloC, ihiC, jloC, jhiC, iloB, ihiB, jloB, jhiB))
         ga_error(" B patch mismatch ",g_B); 

   /*  determine subsets of my patches to access  */
   if (patch_intersect(cilo, cihi, cjlo, cjhi, &iloC, &ihiC, &jloC, &jhiC)){
       ga_access_(&g_A, &iloC, &ihiC, &jloC, &jhiC, &indexA, &ldA);
       ga_access_(&g_B, &iloC, &ihiC, &jloC, &jhiC, &indexB, &ldB);
       ga_access_( g_c, &iloC, &ihiC, &jloC, &jhiC, &indexC, &ldC);

       indexA--; indexB--; indexC--;    /* Fortran to C correction of indices*/

      /* compute "local" add */
       for(j=0; j < jhiC - jloC+1; j++)
          for(i=0; i< ihiC - iloC +1; i++)
             DBL_MB[indexC +j*ldC + i] = *alpha * DBL_MB[indexA +j*ldA + i]  +
                                         *beta  * DBL_MB[indexB +j*ldB + i];

       /* release access to the data */
        ga_release_       (&g_A, &iloC, &ihiC, &jloC, &jhiC);
        ga_release_       (&g_B, &iloC, &ihiC, &jloC, &jhiC); 
        ga_release_update_( g_c, &iloC, &ihiC, &jloC, &jhiC); 
   }

   if(A_created) ga_destroy_(&g_A);
   if(B_created) ga_destroy_(&g_B);

   GA_POP_NAME;
   ga_sync_();
}


/*\ MATRIX MULTIPLICATION for patches 
 *  
 *  C[lo:hi,lo:hi] = alpha*op(A)[lo:hi,lo:hi] * op(B)[lo:hi,lo:hi]        
 *                 + beta *C[lo:hi,lo:hi]
 *
 *  where:
 *          op(A) = A or A' depending on the transpose flag
 *  [lo:hi,lo:hi] - patch indices _after_ op() operator was applied
 *
\*/
void ga_matmul_patch(transa, transb, alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi)

     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     DoublePrecision      *alpha, *beta;
     char    *transa, *transb;
{
#define Ichunk 128
#define Jchunk 128
#define Kchunk 128
DoublePrecision a[Ichunk*Kchunk], b[Kchunk*Jchunk], c[Ichunk*Jchunk];
Integer atype, btype, ctype, adim1, adim2, bdim1, bdim2, cdim1, cdim2;
Integer me= ga_nodeid_(), nproc=ga_nnodes_();
Integer i, ijk = 0, i0, i1, j0, j1;
Integer ilo, ihi, idim, jlo, jhi, jdim, klo, khi, kdim;
Integer n, m, k, adim, bdim, cdim;
DoublePrecision ONE = 1.;

   ga_sync_();
   GA_PUSH_NAME("ga_matmul_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);
   ga_inquire_(g_c, &ctype, &cdim1, &cdim2);

   if(atype != btype || atype != ctype || atype != MT_F_DBL)
               ga_error(" types mismatch ", 0L);

  /* check if patch indices and dims match */
   if (*transa == 'n' || *transa == 'N'){
      if (*ailo <= 0 || *aihi > adim1 || *ajlo <= 0 || *ajhi > adim2)
         ga_error("  g_a indices out of range ", *g_a);
   }else
      if (*ailo <= 0 || *aihi > adim2 || *ajlo <= 0 || *ajhi > adim1)
         ga_error("  g_a indices out of range ", *g_a);

   if (*transb == 'n' || *transb == 'N'){
      if (*bilo <= 0 || *bihi > bdim1 || *bjlo <= 0 || *bjhi > bdim2)
          ga_error("  g_b indices out of range ", *g_b);
   }else
      if (*bilo <= 0 || *bihi > bdim2 || *bjlo <= 0 || *bjhi > bdim1)
          ga_error("  g_b indices out of range ", *g_b);

   if (*cilo <= 0 || *cihi > cdim1 || *cjlo <= 0 || *cjhi > cdim2)
       ga_error("  g_c indices out of range ", *g_c);

  /* verify if patch dimensions are consistent */
   m = *aihi - *ailo +1;
   n = *bjhi - *bjlo +1;
   k = *ajhi - *ajlo +1;
   if( (*cihi - *cilo +1) != m) ga_error(" a & c dims error",m);
   if( (*cjhi - *cjlo +1) != n) ga_error(" b & c dims error",n);
   if( (*bihi - *bilo +1) != k) ga_error(" a & b dims error",k);

   if(*beta) ga_dscal_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);
   else      ga_dfill_patch_(g_c, cilo, cihi, cjlo, cjhi, beta);
  
   for(jlo = 0; jlo < n; jlo += Jchunk){ /* loop through columns of g_c patch */
       jhi = MIN(n-1, jlo+Jchunk-1);
       jdim= jhi - jlo +1;
       for(ilo = 0; ilo < m; ilo += Ichunk){ /*loop through rows of g_c patch */
           ihi = MIN(m-1, ilo+Ichunk-1);
           idim= cdim = ihi - ilo +1;
           for(klo = 0; klo < k; klo += Kchunk){    /* loop cols of g_a patch */
                                                    /* loop rows of g_b patch */
               if(ijk%nproc == me){
                  for (i = 0; i < idim*jdim; i++) c[i]=0.;
                  khi = MIN(k-1, klo+Kchunk-1);
                  kdim= khi - klo +1;
                  if (*transa == 'n' || *transa == 'N'){ 
                     adim = idim;
                     i0= *ailo+ilo; i1= *ailo+ihi;   
                     j0= *ajlo+klo; j1= *ajlo+khi;
                     ga_get_(g_a, &i0, &i1, &j0, &j1, a, &idim);
                  }else{
                     adim = kdim;
                     i0= *ajlo+klo; i1= *ajlo+khi;   
                     j0= *ailo+ilo; j1= *ailo+ihi;
                     ga_get_(g_a, &i0, &i1, &j0, &j1, a, &kdim);
                  }
                  if (*transb == 'n' || *transb == 'N'){ 
                     bdim = kdim;
                     i0= *bilo+klo; i1= *bilo+khi;   
                     j0= *bjlo+jlo; j1= *bjlo+jhi;
                     ga_get_(g_b, &i0, &i1, &j0, &j1, b, &kdim);
                  }else{
                     bdim = jdim;
                     i0= *bjlo+jlo; i1= *bjlo+jhi;   
                     j0= *bilo+klo; j1= *bilo+khi;
                     ga_get_(g_b, &i0, &i1, &j0, &j1, b, &jdim);
                  }
#                 ifdef CRAY_T3D
                    SGEMM(cptofcd(transa), cptofcd(transb), &idim, &jdim, &kdim,
                          alpha, a, &adim, b, &bdim, &ONE, c, &cdim);
#                 else
                    dgemm_(transa, transb, &idim, &jdim, &kdim,
                           alpha, a, &adim, b, &bdim, &ONE, c, &cdim, 1, 1);
#                 endif
                  i0= *cilo+ilo; i1= *cilo+ihi;   j0= *cjlo+jlo; j1= *cjlo+jhi;
                  ga_acc_(g_c, &i0, &i1, &j0, &j1, c, &cdim, &ONE);
               }
               ijk++;
          }
      }
   }
 
   GA_POP_NAME;
   ga_sync_();
}


/*\ MATRIX MULTIPLICATION for patches 
 *  Fortran interface
\*/
void ga_matmul_patch_(transa, transb, alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi)

     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     DoublePrecision      *alpha, *beta;

#ifdef CRAY_T3D
     _fcd   transa, transb;
{    ga_matmul_patch(_fcdtocp(transa), _fcdtocp(transb), alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi);}
#else
     char    *transa, *transb;
{    ga_matmul_patch (transa, transb, alpha, beta,
                      g_a, ailo, aihi, ajlo, ajhi,
                      g_b, bilo, bihi, bjlo, bjhi,
                      g_c, cilo, cihi, cjlo, cjhi);}
#endif

