/* $Id: global.alg.c,v 1.14 1999-07-28 00:27:05 d3h325 Exp $ */
/*************************************************************************\
 Purpose:   File global.alg.c contains a set of linear algebra routines 
            that operate on global arrays in the SPMD mode. 

 Remarks:   Only global array kernel routines that encapsulate (hopefuly)
            all the architecture dependent operations are used to reference
            the distributed data.  

 Developed: 01.16.94 by Jarek Nieplocha
 Modified:  04.28.94 -- changed base arrays addressing
 Modified:  06.11.96 -- added support for complex datatype
\************************************************************************/

 
#include <stdio.h>
#include "global.h"
#include "globalp.h"


/*\ COPY ONE GLOBAL ARRAY INTO ANOTHER
\*/
void FATR ga_copy_(g_a, g_b)
     Integer *g_a, *g_b;
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
Integer ilo, ihi, jlo, jhi;
Integer me= ga_nodeid_(), index, ld;

   ga_sync_();

   ga_check_handle(g_a, "ga_copy");
   ga_check_handle(g_b, "ga_copy");

   if(*g_a == *g_b) ga_error("ga_copy: arrays have to be different ", 0L);

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != MT_F_DBL && atype != MT_F_INT &&
                         atype != MT_F_DCPL))
               ga_error("ga_copy: wrong types ", 0L);

   if(adim1 != bdim1 || adim2!=bdim2 )
               ga_error("ga_copy: arrays not conformant", 0L);

   ga_distribution_(g_a, &me, &ilo, &ihi, &jlo, &jhi);

   if (  ihi>0 && jhi>0 ){                                   
      ga_access_(g_a, &ilo, &ihi, &jlo, &jhi,  &index, &ld);
      switch (atype){
        case MT_F_DBL:
           ga_put_(g_b, &ilo, &ihi, &jlo, &jhi, DBL_MB+index-1, &ld); break;
        case MT_F_DCPL:
           ga_put_(g_b, &ilo, &ihi, &jlo, &jhi, DCPL_MB+index-1, &ld); break;
        case MT_F_INT:
           ga_put_(g_b, &ilo, &ihi, &jlo, &jhi, INT_MB+index-1, &ld);
      }
      ga_release_(g_a, &ilo, &ihi, &jlo, &jhi);
   }

   ga_sync_();
}


void FATR ga_zero_(g_a)
     Integer *g_a;
{
Integer ilo,ihi,jlo,jhi,ld,me,type,dim1,dim2;
register Integer i,j;
Integer index;

   ga_sync_();

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_zero");
   GA_PUSH_NAME("ga_zero");

   ga_inquire_(g_a, &type, &dim1, &dim2);
   ga_distribution_(g_a, &me, &ilo, &ihi, &jlo, &jhi);

   if (DBL_MB == (DoublePrecision*)0 || INT_MB == (Integer*)0 )
                  ga_error("null pointer for base array",0L);
   if (DCPL_MB == (DoubleComplex*)0 )
                  ga_error("null pointer for base array",0L);

   if (  ihi>0 && jhi>0 ){
      ga_access_(g_a, &ilo, &ihi, &jlo, &jhi,  &index, &ld);
      index --;  /* Fortran to C correction of starting address */ 

/*      fprintf(stderr,"%d: ld=%d idx=%d\n",me, ld, index);*/
      switch (type){
        case MT_F_INT:
           for(j=0; j<jhi-jlo+1; j++)
              for(i=0; i<ihi-ilo+1; i++)
                   INT_MB[index +j*ld + i ]  = 0; 
           break;
        case MT_F_DCPL:
           for(j=0; j<jhi-jlo+1; j++)
              for(i=0; i<ihi-ilo+1; i++){
                   DCPL_MB[index +j*ld + i].real  = 0.; 
                   DCPL_MB[index +j*ld + i].imag  = 0.; 
              }
           break;
        case MT_F_DBL:
           for(j=0; j<jhi-jlo+1; j++)
              for(i=0; i<ihi-ilo+1; i++)
                   DBL_MB[index +j*ld + i]  = 0.; 
           break;
        default: ga_error(" wrong data type ",type);
      }

      /* release access to the data */
      ga_release_update_(g_a, &ilo, &ihi, &jlo, &jhi);
   } 


   GA_POP_NAME;
   ga_sync_();
}


DoublePrecision FATR ga_ddot_(g_a, g_b)
        Integer *g_a, *g_b;
{
Integer  atype, adim1, adim2, btype, bdim1, bdim2, ald, bld;
Integer  ailo,aihi, ajlo, ajhi, bilo, bihi, bjlo, bjhi;
register Integer i,j;
Integer  type,len, me;
DoublePrecision  sum;
Integer     index_a, index_b;

   ga_sync_();

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_ddot");
   ga_check_handle(g_b, "ga_ddot");

   GA_PUSH_NAME("ga_ddot");
   ga_inquire_(g_a,  &atype, &adim1, &adim2);
   ga_inquire_(g_b,  &btype, &bdim1, &bdim2);

   if(atype != btype || atype != MT_F_DBL)
        ga_error("types not correct", 0L);

   if (adim1!=bdim1 || adim2 != bdim2)
            ga_error("arrays not conformant", 0L);

   if (DBL_MB == (DoublePrecision*)0 )
                  ga_error(" null pointer for base array",0L);

   ga_distribution_(g_a, &me, &ailo, &aihi, &ajlo, &ajhi);
   ga_distribution_(g_b, &me, &bilo, &bihi, &bjlo, &bjhi);

   if (ailo!=bilo || aihi != bihi || ajlo!=bjlo || ajhi != bjhi){
         /*
         fprintf(stderr,"\nme =%d: %d-%d %d-%d vs %d-%d %d-%d dim:%dx%d\n",me,
                ailo,aihi, ajlo, ajhi, bilo, bihi, bjlo, bjhi,adim1,adim2);
         */
         ga_error("distributions not identical",0L);
   }

   sum = 0.;
   if (  aihi>0 && ajhi>0 ){
       ga_access_(g_a, &ailo, &aihi, &ajlo, &ajhi,  &index_a, &ald);
       if(g_a == g_b){
          index_b = index_a; bld =ald;
       }else
       ga_access_(g_b, &bilo, &bihi, &bjlo, &bjhi,  &index_b, &bld);

       index_a --;  /* Fortran to C correction of starting address */ 
       index_b --;  /* Fortran to C correction of starting address */ 

   
       /* compute "local" contribution to the dot product */
       for(j=0; j<ajhi-ajlo+1; j++)
          for(i=0; i<aihi-ailo+1; i++)
             sum += DBL_MB[index_a +j*ald + i]  *
                    DBL_MB[index_b +j*bld + i];
   
       /* release access to the data */
       ga_release_(g_a, &ailo, &aihi, &ajlo, &ajhi);
       ga_release_(g_b, &bilo, &bihi, &bjlo, &bjhi);
   }

   type = GA_TYPE_GSM; len =1;
   ga_dgop(type, &sum, len, "+"); 
    
   GA_POP_NAME;

   return (sum);
}
 

Integer FATR ga_idot_(g_a, g_b)
        Integer *g_a, *g_b;
{
Integer  atype, adim1, adim2, btype, bdim1, bdim2, ald, bld;
Integer  ailo,aihi, ajlo, ajhi, bilo, bihi, bjlo, bjhi;
register Integer i,j;
Integer  type,len, me;
Integer  sum;
Integer     index_a, index_b;

   ga_sync_();

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_idot");
   ga_check_handle(g_b, "ga_idot");

   GA_PUSH_NAME("ga_idot");
   ga_inquire_(g_a,  &atype, &adim1, &adim2);
   ga_inquire_(g_b,  &btype, &bdim1, &bdim2);

   if(atype != btype || atype != MT_F_INT) ga_error("type not correct", 0L);

   if (adim1!=bdim1 || adim2 != bdim2) ga_error("arrays not conformant", 0L);

   if (INT_MB == (Integer*)0 )ga_error(" null pointer for base array",0L);

   ga_distribution_(g_a, &me, &ailo, &aihi, &ajlo, &ajhi);
   ga_distribution_(g_b, &me, &bilo, &bihi, &bjlo, &bjhi);

   if (ailo!=bilo || aihi != bihi || ajlo!=bjlo || ajhi != bjhi){
         ga_error("distributions not identical",0L);
   }

   sum = 0.;
   if (  aihi>0 && ajhi>0 ){
       ga_access_(g_a, &ailo, &aihi, &ajlo, &ajhi,  &index_a, &ald);
       if(g_a == g_b){
          index_b = index_a; bld =ald;
       }else
       ga_access_(g_b, &bilo, &bihi, &bjlo, &bjhi,  &index_b, &bld);

       index_a --;  /* Fortran to C correction of starting address */ 
       index_b --;  /* Fortran to C correction of starting address */ 

       /* compute "local" contribution to the dot product */
       for(j=0; j<ajhi-ajlo+1; j++)
          for(i=0; i<aihi-ailo+1; i++)
             sum += INT_MB[index_a +j*ald + i]  *
                    INT_MB[index_b +j*bld + i];
   
       /* release access to the data */
       ga_release_(g_a, &ailo, &aihi, &ajlo, &ajhi);
       ga_release_(g_b, &bilo, &bihi, &bjlo, &bjhi);
   }

   type = GA_TYPE_GSM; len =1;
   ga_igop(type, &sum, len, "+"); 
    
   GA_POP_NAME;

   return (sum);
}


#if defined(CRAY) || defined(WIN32)
# define gai_zdot_ GAI_ZDOT
#endif
/*DoubleComplex ga_zdot_(g_a, g_b)*/
void FATR gai_zdot_(g_a, g_b, retval)
        Integer *g_a, *g_b;
        DoubleComplex *retval;
{
Integer  atype, adim1, adim2, btype, bdim1, bdim2, ald, bld;
Integer  ailo,aihi, ajlo, ajhi, bilo, bihi, bjlo, bjhi;
register Integer i,j;
Integer  type,len, me;
DoubleComplex  sum;
Integer     index_a, index_b;

   ga_sync_();

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_ddot");
   ga_check_handle(g_b, "ga_ddot");

   GA_PUSH_NAME("ga_ddot");
   ga_inquire_(g_a,  &atype, &adim1, &adim2);
   ga_inquire_(g_b,  &btype, &bdim1, &bdim2);

   if(atype != btype || atype != MT_F_DCPL)
        ga_error("types not correct", 0L);

   if (adim1!=bdim1 || adim2 != bdim2)
            ga_error("arrays not conformant", 0L);

   if (DCPL_MB == (DoubleComplex*)0 )
                  ga_error(" null pointer for base array",0L);

   ga_distribution_(g_a, &me, &ailo, &aihi, &ajlo, &ajhi);
   ga_distribution_(g_b, &me, &bilo, &bihi, &bjlo, &bjhi);

   if (ailo!=bilo || aihi != bihi || ajlo!=bjlo || ajhi != bjhi){
         /*
         fprintf(stderr,"\nme =%d: %d-%d %d-%d vs %d-%d %d-%d dim:%dx%d\n",me,
                ailo,aihi, ajlo, ajhi, bilo, bihi, bjlo, bjhi,adim1,adim2);
         */
         ga_error("distributions not identical",0L);
   }

   sum.real = 0.;
   sum.imag = 0.;
   if (  aihi>0 && ajhi>0 ){
       ga_access_(g_a, &ailo, &aihi, &ajlo, &ajhi,  &index_a, &ald);
       if(g_a == g_b){
          index_b = index_a; bld =ald;
       }else
       ga_access_(g_b, &bilo, &bihi, &bjlo, &bjhi,  &index_b, &bld);

       index_a --;  /* Fortran to C correction of starting address */
       index_b --;  /* Fortran to C correction of starting address */


       /* compute "local" contribution to the dot product */
       for(j=0; j<ajhi-ajlo+1; j++)
          for(i=0; i<aihi-ailo+1; i++){
                  DoubleComplex a = DCPL_MB[index_a +j*ald + i];
                  DoubleComplex b = DCPL_MB[index_b +j*bld + i];
                  sum.real += a.real*b.real  - b.imag * a.imag;
                  sum.imag += a.imag*b.real  + b.imag * a.real;

       }
       /* release access to the data */
       ga_release_(g_a, &ailo, &aihi, &ajlo, &ajhi);
       ga_release_(g_b, &bilo, &bihi, &bjlo, &bjhi);
   }

   type = GA_TYPE_GSM; len =2; /* take advantage of DoubleComplex layout */
   ga_dgop(type, (DoublePrecision*)&sum, len, "+"); 

   GA_POP_NAME;
/*   ga_sync_();*/

   *retval = sum;
}


/*\ DoubleComplex ga_zdot - C version
\*/
DoubleComplex ga_zdot(Integer *g_a, Integer *g_b)
{
DoubleComplex sum;
        gai_zdot_(g_a, g_b, &sum);
        return sum;
}
  
 
void FATR ga_scale_(g_a, alpha)
        Integer *g_a;
        void *alpha;
{
Integer type, dim1, dim2, ld,   ilo,ihi, jlo, jhi,me;
register Integer i,j;
Integer index;

   ga_sync_();

   me = ga_nodeid_();

   GA_PUSH_NAME("ga_scale");
   ga_inquire_(g_a, &type, &dim1, &dim2);

   if (DBL_MB == (DoublePrecision*)0 || INT_MB == (Integer*)0)
                  ga_error("null pointer for base array", 0L);
   if (DCPL_MB == (DoubleComplex*)0 )
                  ga_error("null pointer for base array",0L);

   ga_distribution_(g_a, &me, &ilo, &ihi, &jlo, &jhi);

   if (  ihi>0 && jhi>0 ){
      ga_access_(g_a, &ilo, &ihi, &jlo, &jhi,  &index, &ld);
      index --;  /* Fortran to C correction of starting address */ 

  
      /* scale local part of g_a */
      switch(type){
         case MT_F_DBL:
              for(j=0; j<jhi-jlo+1; j++)
                 for(i=0; i<ihi-ilo+1; i++)
                     DBL_MB[index +j*ld + i]  *= *(DoublePrecision*)alpha;
              break;
         case MT_F_DCPL:
              for(j=0; j<jhi-jlo+1; j++)
                 for(i=0; i<ihi-ilo+1; i++){
                     DoubleComplex elem = DCPL_MB[index +j*ld + i];
                     DoubleComplex scale= *(DoubleComplex*)alpha;
                     DCPL_MB[index +j*ld + i].real = 
                             scale.real*elem.real  - elem.imag * scale.imag; 
                     DCPL_MB[index +j*ld + i].imag = 
                             scale.imag*elem.real  + elem.imag * scale.real;
                 }
              break;
         case MT_F_INT:
              for(j=0; j<jhi-jlo+1; j++)
                 for(i=0; i<ihi-ilo+1; i++)
                     INT_MB[index +j*ld + i]  *= *(Integer*)alpha;
      }

      /* release access to the data */
      ga_release_update_(g_a, &ilo, &ihi, &jlo, &jhi);
   }

   GA_POP_NAME;
   ga_sync_();
}




void FATR ga_add_(alpha, g_a, beta, g_b,g_c)
        Integer *g_a, *g_b, *g_c;
        Void *alpha, *beta;
{
Integer atype, adim1, adim2, ald;
Integer btype, bdim1, bdim2, bld;
Integer ctype, cdim1, cdim2, cld;
Integer ailo,aihi, ajlo, ajhi;
Integer bihi, bilo, bjlo, bjhi;
Integer cihi, cilo, cjlo, cjhi;
register Integer i,j;
Integer me;
Integer index_a, index_b, index_c;


   ga_sync_();

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_dadd");
   ga_check_handle(g_b, "ga_dadd");
   ga_check_handle(g_c, "ga_dadd");

   GA_PUSH_NAME("ga_add");
   ga_inquire_(g_a,  &atype, &adim1, &adim2);
   ga_inquire_(g_b,  &btype, &bdim1, &bdim2);
   ga_inquire_(g_c,  &ctype, &cdim1, &cdim2);

   if(atype != btype || atype != ctype) ga_error("type mismatch ", 0L);

   if (adim1!=bdim1 || adim2 != bdim2 || adim1!=cdim1 || adim2 != cdim2)
            ga_error("arrays not conformant", 0L);

   if (DBL_MB == (DoublePrecision*)0 || INT_MB == (Integer*)0)
                  ga_error(": null pointer for base array",0L);
   if (DCPL_MB == (DoubleComplex*)0 )
                  ga_error("null pointer for base array",0L);

   ga_distribution_(g_a, &me, &ailo, &aihi, &ajlo, &ajhi);
   ga_distribution_(g_b, &me, &bilo, &bihi, &bjlo, &bjhi);
   ga_distribution_(g_c, &me, &cilo, &cihi, &cjlo, &cjhi);

   if (ailo!=bilo || aihi != bihi || ajlo!=bjlo || ajhi != bjhi ||
       ailo!=cilo || aihi != cihi || ajlo!=cjlo || ajhi != cjhi)
             ga_error("distributions not identical",0L);

   if (  aihi>0 && ajhi>0 ){

       ga_access_(g_a, &ailo, &aihi, &ajlo, &ajhi,  &index_a, &ald);
       if(g_a == g_b){
          index_b = index_a; bld =ald; 
       }else
       ga_access_(g_b, &bilo, &bihi, &bjlo, &bjhi,  &index_b, &bld);
       if(g_a == g_c){
          index_c = index_a; cld =ald; 
       }else if(g_b == g_c){
          index_c = index_b; cld =bld; 
       }else
       ga_access_(g_c, &cilo, &cihi, &cjlo, &cjhi,  &index_c, &cld);

       index_a --;  /* Fortran to C correction of starting address */ 
       index_b --;  /* Fortran to C correction of starting address */ 
       index_c --;  /* Fortran to C correction of starting address */ 

       /* operation on the "local" piece of data */
       switch(atype){
         case MT_F_DBL:
              for(j=0; j<ajhi-ajlo+1; j++)
                  for(i=0; i<aihi-ailo+1; i++)
                      DBL_MB[index_c +j*cld + i]  =
                         *(DoublePrecision*)alpha * DBL_MB[index_a +j*ald + i] +
                         *(DoublePrecision*)beta  * DBL_MB[index_b +j*bld + i];
              break;
         case MT_F_DCPL:
              for(j=0; j<ajhi-ajlo+1; j++)
                  for(i=0; i<aihi-ailo+1; i++){
                     DoubleComplex a = DCPL_MB[index_a +j*ald + i];
                     DoubleComplex b = DCPL_MB[index_b +j*bld + i];
                     DoubleComplex x= *(DoubleComplex*)alpha;
                     DoubleComplex y= *(DoubleComplex*)beta;
                     /* c = x*a + y*b */
                     DCPL_MB[index_c +j*cld + i].real = x.real*a.real - 
                             x.imag*a.imag + y.real*b.real - y.imag*b.imag;
                     DCPL_MB[index_c +j*cld + i].imag = x.real*a.imag + 
                             x.imag*a.real + y.real*b.imag + y.imag*b.real;
                  }
              break;
         case MT_F_INT:
              for(j=0; j<ajhi-ajlo+1; j++)
                  for(i=0; i<aihi-ailo+1; i++)
                      INT_MB[index_c +j*cld + i]  =
                         *(Integer*)alpha * INT_MB[index_a +j*ald + i] +
                         *(Integer*)beta  * INT_MB[index_b +j*bld + i];
       }

       /* release access to the data */
       ga_release_update_(g_c, &cilo, &cihi, &cjlo, &cjhi);
       ga_release_(g_b, &bilo, &bihi, &bjlo, &bjhi);
       ga_release_(g_a, &ailo, &aihi, &ajlo, &ajhi);
   }


   GA_POP_NAME;
   ga_sync_();
}
