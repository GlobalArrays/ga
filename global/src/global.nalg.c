/*************************************************************************\
 Purpose:   File global.nalg.c contains a set of linear algebra routines 
            that operate on n-dim global arrays in the SPMD mode. 

 Date: 10.22.98
 Author: Jarek Nieplocha
\************************************************************************/

 
#include <stdio.h>
#include "global.h"
#include "globalp.h"


/* work arrays used in all routines */
static Integer dims[MAXDIM], ld[MAXDIM-1];
static Integer lo[MAXDIM],hi[MAXDIM];

#define GET_ELEMS(ndim,lo,hi,ld,pelems){\
int _i;\
      for(_i=0, *pelems = hi[ndim-1]-lo[ndim-1]+1; _i< ndim-1;_i++) {\
         if(ld[_i] != (hi[_i]-lo[_i]+1)) ga_error("layout problem",_i);\
         *pelems *= hi[_i]-lo[_i]+1;\
      }\
}


void FATR ga_zero_(Integer *g_a)
{
Integer ndim, type, me, index, elems;
register Integer i;

   ga_sync_();

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_zero");
   GA_PUSH_NAME("ga_zero");

   nga_inquire_(g_a, &type, &ndim, dims);
   nga_distribution_(g_a, &me, lo, hi);

   if (DBL_MB == (DoublePrecision*)0 || INT_MB == (Integer*)0 ||
       DCPL_MB == (DoubleComplex*)0) ga_error("null pointer for base array",0L);

   if ( lo[0]> 0 ){ /* base index is 1: we get 0 if no elements stored on p */

      nga_access_(g_a, lo, hi, &index, ld);
      GET_ELEMS(ndim,lo,hi,ld,&elems);

      index --;  /* Fortran to C correction of starting address */ 
      
      switch (type){
        case MT_F_INT:
           for(i=0;i<elems;i++) INT_MB[index+ i ]  = 0;
           break;
        case MT_F_DCPL:
           for(i=0;i<elems;i++)DCPL_MB[index+i].real=DCPL_MB[index+i].imag = 0.;
           break;
        case MT_F_DBL:
           for(i=0;i<elems;i++) DBL_MB[index+ i ]  = 0;
           break;
        default: ga_error(" wrong data type ",type);
      }

      /* release access to the data */
      nga_release_update_(g_a, lo, hi);
   } 


   GA_POP_NAME;
   ga_sync_();
}


void gai_dot(int Type, Integer *g_a, Integer *g_b, void *value)
{
Integer  ndim, type, me, elems=0, elemsb=0;
Integer  sum =0, index_a, index_b; 
register Integer i;
Integer isum=0;
DoubleComplex zsum ={0.,0.};

   ga_sync_();
   me = ga_nodeid_();
   ga_check_handle(g_a, "ga_dot");
   ga_check_handle(g_b, "ga_dot");

   GA_PUSH_NAME("ga_dot");

   if(ga_compare_distr_(g_a,g_b) == FALSE)
         ga_error("distributions not identical",0L);

   nga_inquire_(g_a,  &type, &ndim, dims);
   if(type != Type) ga_error("type not correct", *g_a);
   nga_distribution_(g_a, &me, lo, hi);
   if(lo[0]>0){
      nga_access_(g_a, lo, hi, &index_a, ld);
      GET_ELEMS(ndim,lo,hi,ld,&elems);
   }

   if(*g_a == *g_b){
     index_b = index_a;
     elemsb = elems;
   }else {  
     nga_inquire_(g_b,  &type, &ndim, dims);
     if(type != Type) ga_error("type not correct", *g_b);
     nga_distribution_(g_b, &me, lo, hi);
     if(lo[0]>0){
        nga_access_(g_b, lo, hi, &index_b, ld);
        GET_ELEMS(ndim,lo,hi,ld,&elemsb);
     }
   }

   if(elems!= elemsb)ga_error("inconsistent number of elements",elems-elemsb); 

      index_a --;  /* Fortran to C correction of starting address */ 
      index_b --;  /* Fortran to C correction of starting address */ 

      /* compute "local" contribution to the dot product */
      switch (type){

        case MT_F_INT:
           for(i=0;i<elems;i++) 
                 isum += INT_MB[index_a + i]  * INT_MB[index_b + i];
           *(Integer*)value = isum; 
           break;

        case MT_F_DCPL:
           for(i=0;i<elems;i++){
               DoubleComplex a = DCPL_MB[index_a + i];
               DoubleComplex b = DCPL_MB[index_b + i];
               zsum.real += a.real*b.real  - b.imag * a.imag;
               zsum.imag += a.imag*b.real  + b.imag * a.real;
           }
           *(DoubleComplex*)value = zsum; 
           break;

        case MT_F_DBL:
           for(i=0;i<elems;i++) 
                 zsum.real += DBL_MB[index_a + i]  * DBL_MB[index_b + i];
           *(DoublePrecision*)value = zsum.real; 
           break;
        default: ga_error(" wrong data type ",type);
      }
   
      /* release access to the data */
      if(elems>0){
         nga_release_(g_a, lo, hi);
         if(*g_a != *g_b)nga_release_(g_b, lo, hi);
      }


   if(Type == MT_F_INT)ga_igop((Integer)GA_TYPE_GSM,(Integer*)value, 1, "+");
   else if(Type == MT_F_DBL) 
     ga_dgop((Integer)GA_TYPE_GSM, (DoublePrecision*)value, 1, "+"); 
   else
     ga_dgop((Integer)GA_TYPE_GSM, (DoublePrecision*)value, 2, "+"); 
    
   GA_POP_NAME;

}


Integer FATR ga_idot_(g_a, g_b)
        Integer *g_a, *g_b;
{
Integer sum;
        gai_dot(MT_F_INT, g_a, g_b, &sum);
        return sum;
}


DoublePrecision FATR ga_ddot_(g_a, g_b)
        Integer *g_a, *g_b;
{
DoublePrecision sum;
        gai_dot(MT_F_DBL, g_a, g_b, &sum);
        return sum;
}


/*\ DoubleComplex ga_zdot - C version
\*/ 
DoubleComplex ga_zdot(Integer *g_a, Integer *g_b)
{
DoubleComplex sum;
        gai_dot(MT_F_DCPL, g_a, g_b, &sum);
        return sum;
}


#if defined(CRAY) || defined(WIN32)
# define gai_zdot_ GAI_ZDOT
#endif
void FATR gai_zdot_(g_a, g_b, retval)
        Integer *g_a, *g_b;
        DoubleComplex *retval;  
{
     gai_dot(MT_F_DCPL, g_a, g_b, retval);
}


 
void FATR ga_scale_(Integer *g_a, void* alpha)
{
Integer ndim, type, me, index, elems;
register Integer i;

   ga_sync_();

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_zero");
   GA_PUSH_NAME("ga_zero");

   nga_inquire_(g_a, &type, &ndim, dims);
   nga_distribution_(g_a, &me, lo, hi);

   if (DBL_MB == (DoublePrecision*)0 || INT_MB == (Integer*)0 ||
       DCPL_MB == (DoubleComplex*)0) ga_error("null pointer for base array",0L);

   if ( lo[0]> 0 ){ /* base index is 1: we get 0 if no elements stored on p */

      nga_access_(g_a, lo, hi, &index, ld);
      GET_ELEMS(ndim,lo,hi,ld,&elems);

      index --;  /* Fortran to C correction of starting address */

      switch (type){
        case MT_F_INT:
           for(i=0;i<elems;i++) INT_MB[index+ i ]  *= *(Integer*)alpha;
           break;
        case MT_F_DCPL:
           for(i=0;i<elems;i++){
                DoubleComplex elem = DCPL_MB[index + i];
                DoubleComplex scale= *(DoubleComplex*)alpha;
                DCPL_MB[index  + i].real =
                        scale.real*elem.real  - elem.imag * scale.imag;
                DCPL_MB[index  + i].imag =
                        scale.imag*elem.real  + elem.imag * scale.real;
           }
           break;
        case MT_F_DBL:
           for(i=0;i<elems;i++) DBL_MB[index+ i] *= *(DoublePrecision*)alpha;
           break;
        default: ga_error(" wrong data type ",type);
      }

      /* release access to the data */
      nga_release_update_(g_a, lo, hi);
   }

   GA_POP_NAME;
   ga_sync_();
}




void FATR ga_add_(void *alpha, Integer* g_a, 
                  void* beta, Integer* g_b, Integer* g_c)
{
Integer  ndim, type, typeC, me, elems=0, elemsb=0, elemsa=0;
register Integer i;
Integer index_a, index_b, index_c;


   ga_sync_();

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_add");
   ga_check_handle(g_b, "ga_add");
   ga_check_handle(g_c, "ga_add");

   GA_PUSH_NAME("ga_add");

   if(ga_compare_distr_(g_a,g_b) == FALSE)
         ga_error("distributions not identical",0L);
   if(ga_compare_distr_(g_a,g_c) == FALSE)
         ga_error("distributions not identical",1L);

   nga_inquire_(g_c,  &typeC, &ndim, dims);
   nga_distribution_(g_c, &me, lo, hi);
   if (  lo[0]>0 ){
     nga_access_(g_c, lo, hi, &index_c, ld);
     GET_ELEMS(ndim,lo,hi,ld,&elems);
   }

   if(*g_a == *g_c){
     index_a = index_c;
     elemsa = elems;
   }else { 
     nga_inquire_(g_a,  &type, &ndim, dims);
     if(type != typeC) ga_error("types not consistent", *g_a);
     nga_distribution_(g_a, &me, lo, hi);
     if (  lo[0]>0 ){
       nga_access_(g_a, lo, hi, &index_a, ld);
       GET_ELEMS(ndim,lo,hi,ld,&elemsa);
     }
   }

   if(*g_b == *g_c){
     index_b = index_c;
     elemsb = elems;
   }else {
     nga_inquire_(g_b,  &type, &ndim, dims);
     if(type != typeC) ga_error("types not consistent", *g_b);
     nga_distribution_(g_b, &me, lo, hi);
     if (  lo[0]>0 ){
       nga_access_(g_b, lo, hi, &index_b, ld);
       GET_ELEMS(ndim,lo,hi,ld,&elemsb);
     }
   }

   if(elems!= elemsb)ga_error("inconsistent number of elements a",elems-elemsb);
   if(elems!= elemsa)ga_error("inconsistent number of elements b",elems-elemsa);

   if (  lo[0]>0 ){

       index_a --;  /* Fortran to C correction of starting address */ 
       index_b --;  /* Fortran to C correction of starting address */ 
       index_c --;  /* Fortran to C correction of starting address */ 

       /* operation on the "local" piece of data */
       switch(type){
         case MT_F_DBL:
                  for(i=0; i<elems; i++)
                      DBL_MB[index_c + i]  =
                         *(DoublePrecision*)alpha * DBL_MB[index_a + i] +
                         *(DoublePrecision*)beta  * DBL_MB[index_b + i];
              break;
         case MT_F_DCPL:
                  for(i=0; i<elems; i++){
                     DoubleComplex a = DCPL_MB[index_a + i];
                     DoubleComplex b = DCPL_MB[index_b + i];
                     DoubleComplex x= *(DoubleComplex*)alpha;
                     DoubleComplex y= *(DoubleComplex*)beta;
                     /* c = x*a + y*b */
                     DCPL_MB[index_c + i].real = x.real*a.real - 
                             x.imag*a.imag + y.real*b.real - y.imag*b.imag;
                     DCPL_MB[index_c + i].imag = x.real*a.imag + 
                             x.imag*a.real + y.real*b.imag + y.imag*b.real;
                  }
              break;
         case MT_F_INT:
                  for(i=0; i<elems; i++)
                      INT_MB[index_c + i]  =
                         *(Integer*)alpha * INT_MB[index_a + i] +
                         *(Integer*)beta  * INT_MB[index_b + i];
       }

       /* release access to the data */
       nga_release_update_(g_c, lo, hi);
       if(*g_c != *g_a)nga_release_(g_a, lo, hi);
       if(*g_c != *g_b)nga_release_(g_b, lo, hi);
   }


   GA_POP_NAME;
   ga_sync_();
}
