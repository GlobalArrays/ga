/*$Id: global.patch.c,v 1.23 1999-11-18 19:52:41 d3h325 Exp $*/
#include "global.h"
#include "globalp.h"
#include <math.h>


#if defined(CRAY) || defined(WIN32)
#   define cptofcd(fcd)  _cptofcd((fcd),1)
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
 *  now implemented elsewhere using internal GA API
\*/
logical FATR ga_compare_distr_disabled(g_a, g_b)
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
   ga_dgop(type, &mismatch, len, "+");
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
char*   base_addr;
Integer byte_index;

#        define ITERATOR_2D   for(j = jlos, jj=0; j <= jhis; j++, jj++)\
                         for(i = ilos, ii =0; i <= ihis; i++, ii++)
#        define COPY_2D(ADDR_BASE) ADDR_BASE[base+vindex]= \
                                   ADDR_BASE[index+ ld*jj + ii]
   ga_sync_();

   GA_PUSH_NAME("ga_copy_patch");
/*   if(*g_a == *g_b) ga_error(" arrays have to be different ", 0L);*/

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

   /* set the address of base for each datatype */
   switch(atype){
         case  MT_F_DBL: base_addr = (char*) DBL_MB; break;
         case  MT_F_INT: base_addr = (char*) INT_MB; break;
         case  MT_F_DCPL: base_addr = (char*) DCPL_MB;
   }

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

         byte_index = index *  GAsizeofM(atype); 
         ga_put_(g_b, &ilod, &ihid, &jlod, &jhid, base_addr+byte_index, &ld); 

      }else{
        /*** due to generality of this transformation scatter is required ***/

         if(!MA_push_get(MT_F_INT, nelem, "i", &ihandle, &iindex))
            ga_error(" MA failed-i ", 0L);
         if(!MA_push_get(MT_F_INT, nelem, "j", &jhandle, &jindex))
            ga_error(" MA failed-j ", 0L);
         if(!MA_push_get(atype, nelem, "v", &vhandle, &vindex))
            ga_error(" MA failed-v ", 0L);

         base = 0;

#        define  COPY_2D_TRANSP(ADDR_BASE)\
           if (*trans == 'n' || *trans == 'N')\
              ITERATOR_2D {\
                   DEST_INDICES(i, j, *ailo, *ajlo, *aihi- *ailo+1,\
                        INT_MB[base+iindex],\
                        INT_MB[base+jindex], *bilo, *bjlo, (*bihi - *bilo +1));\
                   COPY_2D(ADDR_BASE);\
                   base++; }\
           else\
              ITERATOR_2D {\
                   DEST_INDICES(j,i, *ajlo, *ailo, *ajhi - *ajlo +1,\
                        INT_MB[base+iindex],\
                        INT_MB[base+jindex], *bilo, *bjlo, (*bihi - *bilo +1));\
                   COPY_2D(ADDR_BASE);\
                   base++; }

         switch(atype){
         case MT_F_DBL:  COPY_2D_TRANSP(DBL_MB); break; 
         case MT_F_INT:  COPY_2D_TRANSP(INT_MB); break; 
         case MT_F_DCPL: COPY_2D_TRANSP(DCPL_MB);
         }
#        undef ITERATOR_2D
#        undef COPY_2D
#        undef COPY_2D_TRANSP

         ga_release_(g_a, &ilos, &ihis, &jlos, &jhis);
         byte_index = vindex *  GAsizeofM(atype); 
         ga_scatter_(g_b, base_addr+byte_index, 
                     INT_MB+iindex, INT_MB+jindex, &nelem);
         if (!MA_pop_stack(vhandle) || !MA_pop_stack(jhandle) ||
             !MA_pop_stack(ihandle)) ga_error("MA_pop_stack failed",0);
      }
  }
  GA_POP_NAME;
  ga_sync_();
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
Integer atype, btype, adim1, adim2, bdim1, bdim2;
Integer iloA, ihiA, jloA, jhiA, indexA, ldA;
Integer iloB, ihiB, jloB, jhiB, indexB, ldB;
Integer g_A = *g_a, g_B = *g_b;
Integer me= ga_nodeid_(), i, j, temp_created=0;
Integer type = GA_TYPE_GSM, len = 1;
char *tempname = "temp", transp, transp_a, transp_b;
DoublePrecision  sum[2];

   ga_sync_();
   GA_PUSH_NAME("gai_dot_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype ) ga_error(" type mismatch ", 0L);
   if((atype != MT_F_DBL ) && (atype != MT_F_DCPL)) ga_error(" wrong type", 0L);

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
         if (!ga_duplicate(g_a, &g_B, tempname)) ga_error("duplicate failed",0L);

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

   sum[0] = 0.; sum[1] = 0.;

   /*  determine subsets of my patches to access  */
   if (patch_intersect(ailo, aihi, ajlo, ajhi, &iloA, &ihiA, &jloA, &jhiA)){
       ga_access_(&g_A, &iloA, &ihiA, &jloA, &jhiA, &indexA, &ldA);
       ga_access_(&g_B, &iloA, &ihiA, &jloA, &jhiA, &indexB, &ldB);
       indexA --; indexB--;   /* Fortran to C correction of starting address */

      /* compute "local" contribution to the dot product */
       if(atype == MT_F_DCPL)
          for(j=0; j < jhiA - jloA+1; j++) for(i=0; i< ihiA - iloA +1; i++){
                  DoubleComplex a = DCPL_MB[indexA +j*ldA + i];
                  DoubleComplex b = DCPL_MB[indexB +j*ldB + i];
                  sum[0] += a.real*b.real  - b.imag * a.imag;
                  sum[1] += a.imag*b.real  + b.imag * a.real;

          }
       else
          for(j=0; j < jhiA - jloA+1; j++) for(i=0; i< ihiA - iloA +1; i++)
              sum[0] += DBL_MB[indexA +j*ldA + i] * DBL_MB[indexB +j*ldB + i];

       /* release access to the data */
        ga_release_(&g_A, &iloA, &ihiA, &jloA, &jhiA);
        ga_release_(&g_B, &iloA, &ihiA, &jloA, &jhiA);
   }

   len = (atype == MT_F_DCPL) ? 2 : 1; 
   ga_dgop(type, sum, len, "+");
   for(i=0;i<len;i++) retval[i]=sum[i];

   if(temp_created) ga_destroy_(&g_B);

   GA_POP_NAME;
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

   ga_sync_();
   GA_PUSH_NAME("ga_ddot_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != MT_F_DBL )) ga_error(" wrong types ", 0L);

   gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                 g_b, t_b, bilo, bihi, bjlo, bjhi, &sum);

   GA_POP_NAME;
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

   ga_sync_();
   GA_PUSH_NAME("ga_zdot_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != MT_F_DCPL )) ga_error(" wrong types ", 0L);

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
Integer dim1, dim2, type;
Integer iloA, ihiA, jloA, jhiA, index, ld;
Integer me= ga_nodeid_(), i, j;

   ga_sync_();
   GA_PUSH_NAME("ga_fill_patch");

   ga_inquire_(g_a,  &type, &dim1, &dim2);

   ga_distribution_(g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

   /*  determine subset of my patch to access  */
   if (patch_intersect(ilo, ihi, jlo, jhi, &iloA, &ihiA, &jloA, &jhiA)){
       ga_access_(g_a, &iloA, &ihiA, &jloA, &jhiA, &index, &ld);
       index --;  /* Fortran to C correction of starting address */

      switch (type){
        case MT_F_INT:
           for(j=0; j<jhiA-jloA+1; j++)
              for(i=0; i<ihiA-iloA+1; i++)
                   INT_MB[index +j*ld + i ]  = *(Integer*)val;
           break;
        case MT_F_DCPL:
           for(j=0; j<jhiA-jloA+1; j++)
              for(i=0; i<ihiA-iloA+1; i++){
                   DCPL_MB[index +j*ld + i].real  = (*(DoubleComplex*)val).real;
                   DCPL_MB[index +j*ld + i].imag  = (*(DoubleComplex*)val).imag;
              }
           break;
        case MT_F_DBL:
           for(j=0; j<jhiA-jloA+1; j++)
              for(i=0; i<ihiA-iloA+1; i++)
                   DBL_MB[index +j*ld + i]  = *(DoublePrecision*)val;
           break;
        default: ga_error(" wrong data type ",type);
      }

      /* release access to the data */
       ga_release_update_(g_a, &iloA, &ihiA, &jloA, &jhiA);
   }
   GA_POP_NAME;
   ga_sync_();
}



/*\ SCALE ARRAY 
\*/
void FATR ga_scale_patch_(g_a, ilo, ihi, jlo, jhi, alpha)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
     DoublePrecision     *alpha;
{
Integer dim1, dim2, type;
Integer iloA, ihiA, jloA, jhiA, index, ld, i, j;
Integer me= ga_nodeid_();

   ga_sync_();
   GA_PUSH_NAME("ga_scal_patch");

   ga_inquire_(g_a,  &type, &dim1, &dim2);
   ga_distribution_(g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

   /*  determine subset of my patch to access  */
   if (patch_intersect(ilo, ihi, jlo, jhi, &iloA, &ihiA, &jloA, &jhiA)){
       ga_access_(g_a, &iloA, &ihiA, &jloA, &jhiA, &index, &ld);
       index --;  /* Fortran to C correction of starting address */

      /* scale local part of g_a */
      switch(type){
         case MT_F_DBL:
              for(j=0; j<jhiA-jloA+1; j++)
                 for(i=0; i<ihiA-iloA+1; i++)
                     DBL_MB[index +j*ld + i]  *= *(DoublePrecision*)alpha;
              break;
         case MT_F_DCPL:
              for(j=0; j<jhiA-jloA+1; j++)
                 for(i=0; i<ihiA-iloA+1; i++){
                     DoubleComplex elem = DCPL_MB[index +j*ld + i];
                     DoubleComplex scale= *(DoubleComplex*)alpha;
                     DCPL_MB[index +j*ld + i].real =
                             scale.real*elem.real  - elem.imag * scale.imag;
                     DCPL_MB[index +j*ld + i].imag =
                             scale.imag*elem.real  + elem.imag * scale.real;
                 }
              break;
         case MT_F_INT:
              for(j=0; j<jhiA-jloA+1; j++)
                 for(i=0; i<ihiA-iloA+1; i++)
                     INT_MB[index +j*ld + i]  *= *(Integer*)alpha;
      }

      /* release access to the data */
      ga_release_update_(g_a, &iloA, &ihiA, &jloA, &jhiA);

   }
   GA_POP_NAME;
   ga_sync_();
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
Integer atype, btype, ctype, adim1, adim2, bdim1, bdim2, cdim1, cdim2;
Integer iloA, ihiA, jloA, jhiA, indexA, ldA;
Integer iloB, ihiB, jloB, jhiB, indexB, ldB;
Integer iloC, ihiC, jloC, jhiC, indexC, ldC;
Integer g_A = *g_a, g_B = *g_b;
Integer me= ga_nodeid_(), i, j, B_created=0, A_created=0;
Integer nelem;
char *tempname = "temp", notrans='n';

   ga_sync_();
   GA_PUSH_NAME("ga_add_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);
   ga_inquire_(g_c, &ctype, &cdim1, &cdim2);

   if(atype != btype || atype != ctype ) ga_error(" types mismatch ", 0L); 

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
         if (!ga_duplicate(g_c, &g_A, tempname))
              ga_error("ga_dadd_patch: duplicate failed", 0L);
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
         if (!ga_duplicate(g_c, &g_B, tempname))
            ga_error("ga_dadd_patch: dup failed", 0L);
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
       switch(atype){
         case MT_F_DBL:
              for(j=0; j < jhiC - jloC+1; j++)
                 for(i=0; i< ihiC - iloC +1; i++)
                      DBL_MB[indexC +j*ldC + i]  =
                         *(DoublePrecision*)alpha * DBL_MB[indexA +j*ldA + i] +
                         *(DoublePrecision*)beta  * DBL_MB[indexB +j*ldB + i];
              break;
         case MT_F_DCPL:
              for(j=0; j < jhiC - jloC+1; j++)
                 for(i=0; i< ihiC - iloC +1; i++){
                     DoubleComplex a = DCPL_MB[indexA +j*ldA + i];
                     DoubleComplex b = DCPL_MB[indexB +j*ldB + i];
                     DoubleComplex x= *(DoubleComplex*)alpha;
                     DoubleComplex y= *(DoubleComplex*)beta;
                     /* c = x*a + y*b */
                     DCPL_MB[indexC +j*ldC + i].real = x.real*a.real -
                             x.imag*a.imag + y.real*b.real - y.imag*b.imag;
                     DCPL_MB[indexC +j*ldC + i].imag = x.real*a.imag +
                             x.imag*a.real + y.real*b.imag + y.imag*b.real;
                  }
              break;
         case MT_F_INT:
              for(j=0; j < jhiC - jloC+1; j++)
                 for(i=0; i< ihiC - iloC +1; i++)
                      INT_MB[indexC +j*ldC + i]  =
                         *(Integer*)alpha * INT_MB[indexA +j*ldA + i] +
                         *(Integer*)beta  * INT_MB[indexB +j*ldB + i];
       }

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

