/*************************************************************************\
 Purpose:   File global.alg.c contains a set of linear algebra routines 
            that operate on global arrays in the SPMD mode. 

 Remarks:   Only global array kernel routines that encapsulate (hopefuly)
            all the architecture dependent operations are used to reference
            the distributed data.  

 Developed: 01.16.94 by Jarek Nieplocha
 Modified:  04.28.94 -- changed base arrays addressing
\************************************************************************/

 
#include <stdio.h>
#include "global.h"
#include "globalp.h"
#include "macommon.h"



void ga_zero_(g_a)
     Integer *g_a;
{
Integer ilo,ihi,jlo,jhi,ld,me,type,dim1,dim2;
register Integer i,j;
Integer index;

   ga_sync_();

#ifdef GA_TRACE
       trace_stime_();
#endif

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_zero");
   GA_PUSH_NAME("ga_zero");

   ga_inquire_(g_a, &type, &dim1, &dim2);
   ga_distribution_(g_a, &me, &ilo, &ihi, &jlo, &jhi);

   if (DBL_MB == (DoublePrecision*)0 || INT_MB == (Integer*)0)
                  ga_error("null pointer for base array",0L);

   if (  ihi>0 && jhi>0 ){
      ga_access_(g_a, &ilo, &ihi, &jlo, &jhi,  &index, &ld);
      index --;  /* Fortran to C correction of starting address */ 

      if(type==MT_F_DBL){
         for(j=0; j<jhi-jlo+1; j++)
            for(i=0; i<ihi-ilo+1; i++)
                 DBL_MB[index +j*ld + i]  = 0.; 
      }else if(type==MT_F_INT){
         for(j=0; j<jhi-jlo+1; j++)
            for(i=0; i<ihi-ilo+1; i++)
                 INT_MB[index +j*ld + i ]  = 0; 
      }else ga_error(" wrong data type ",0L);

      /* release access to the data */
      ga_release_update_(g_a, &ilo, &ihi, &jlo, &jhi);
   } 

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_ZER;
   trace_genrec_(g_a, &ilo, &ihi, &jlo, &jhi, &op_code);
#endif

   GA_POP_NAME;
   ga_sync_();
}




DoublePrecision ga_ddot_(g_a, g_b)
        Integer *g_a, *g_b;
{
Integer  atype, adim1, adim2, btype, bdim1, bdim2, ald, bld;
Integer  ailo,aihi, ajlo, ajhi, bilo, bihi, bjlo, bjhi;
register Integer i,j;
Integer  type,len, me;
DoublePrecision  sum;
Integer     index_a, index_b;

   ga_sync_();

#ifdef GA_TRACE
       trace_stime_();
#endif

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

   if (DBL_MB == (DoublePrecision*)0 || INT_MB == (Integer*)0)
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
   ga_dgop_(&type, &sum, &len, "+",1); 
    
#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_DDT;
   trace_genrec_(g_a, &ailo, &aihi, &ajlo, &ajhi, &op_code);
   if(g_a != g_b) trace_genrec_(g_b, &bilo, &bihi, &bjlo, &bjhi, &op_code);
#endif

   GA_POP_NAME;
   ga_sync_();

   return (sum);
}
 

  
 
void ga_dscal_(g_a, alpha)
        Integer *g_a;
        DoublePrecision *alpha;
{
Integer type, dim1, dim2, ld,   ilo,ihi, jlo, jhi,me;
register Integer i,j;
Integer index;

   ga_sync_();

#ifdef GA_TRACE
       trace_stime_();
#endif

   me = ga_nodeid_();

   GA_PUSH_NAME("ga_dscal");
   ga_inquire_(g_a, &type, &dim1, &dim2);

   if(type != MT_F_DBL)
        ga_error("type not correct", 0L);

   if (DBL_MB == (DoublePrecision*)0 || INT_MB == (Integer*)0)
                  ga_error("null pointer for base array", 0L);

   ga_distribution_(g_a, &me, &ilo, &ihi, &jlo, &jhi);

   if (  ihi>0 && jhi>0 ){
      ga_access_(g_a, &ilo, &ihi, &jlo, &jhi,  &index, &ld);
      index --;  /* Fortran to C correction of starting address */ 

  
      /* scale local part of g_a */
      for(j=0; j<jhi-jlo+1; j++)
         for(i=0; i<ihi-ilo+1; i++)
             DBL_MB[index +j*ld + i]  *= *alpha; 

      /* release access to the data */
      ga_release_update_(g_a, &ilo, &ihi, &jlo, &jhi);
   }

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_DSC;
   trace_genrec_(g_a, &ilo, &ihi, &jlo, &jhi, &op_code);
#endif

   GA_POP_NAME;
   ga_sync_();
}




void ga_dadd_(alpha, g_a, beta, g_b,g_c)
        Integer *g_a, *g_b, *g_c;
        DoublePrecision *alpha, *beta;
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

#ifdef GA_TRACE
       trace_stime_();
#endif

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_dadd");
   ga_check_handle(g_b, "ga_dadd");
   ga_check_handle(g_c, "ga_dadd");

   GA_PUSH_NAME("ga_dadd");
   ga_inquire_(g_a,  &atype, &adim1, &adim2);
   ga_inquire_(g_b,  &btype, &bdim1, &bdim2);
   ga_inquire_(g_c,  &ctype, &cdim1, &cdim2);

   if(atype != btype || atype != ctype || atype != MT_F_DBL)
        ga_error("types not correct", 0L);

   if (adim1!=bdim1 || adim2 != bdim2 || adim1!=cdim1 || adim2 != cdim2)
            ga_error("arrays not conformant", 0L);

   if (DBL_MB == (DoublePrecision*)0 || INT_MB == (Integer*)0)
                  ga_error(": null pointer for base array",0L);

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
       for(j=0; j<ajhi-ajlo+1; j++)
          for(i=0; i<aihi-ailo+1; i++){
              DBL_MB[index_c +j*cld + i]  =
                 *alpha * DBL_MB[index_a +j*ald + i]  +
                 *beta  * DBL_MB[index_b +j*bld + i];
       }

       /* release access to the data */
       ga_release_update_(g_c, &cilo, &cihi, &cjlo, &cjhi);
       ga_release_(g_b, &bilo, &bihi, &bjlo, &bjhi);
       ga_release_(g_a, &ailo, &aihi, &ajlo, &ajhi);
   }

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_ADD;
   trace_genrec_(g_a, &ailo, &aihi, &ajlo, &ajhi, &op_code);
   if(g_a != g_b) trace_genrec_(g_b, &bilo, &bihi, &bjlo, &bjhi, &op_code);
   if(g_a != g_c && g_c != g_b) 
                  trace_genrec_(g_c, &cilo, &cihi, &cjlo, &cjhi, &op_code);
#endif

   GA_POP_NAME;
   ga_sync_();
}
