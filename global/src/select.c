#include <stdio.h>
#include "global.h"
#include "globalp.h"
#include "../../armci/src/message.h"
  

#define GET_ELEMS(ndim,lo,hi,ld,pelems){\
int _i;\
      for(_i=0, *pelems = hi[ndim-1]-lo[ndim-1]+1; _i< ndim-1;_i++) {\
         if(ld[_i] != (hi[_i]-lo[_i]+1)) ga_error("layout problem",_i);\
         *pelems *= hi[_i]-lo[_i]+1;\
      }\
}


void FATR nga_select_elem_(Integer *g_a, char* op, void* val, Integer *subscript)
{
Integer ndim, type, me, elems, ind=0, i;
Integer lo[MAXDIM],hi[MAXDIM],dims[MAXDIM],ld[MAXDIM-1];
struct  info_t{ 
        union val_t {double dval; long lval;}v; 
        Integer subscr[MAXDIM]; 
        DoubleComplex extra;} info;
int     participate=0;

   ga_sync_();

   me = ga_nodeid_();

   ga_check_handle(g_a, "ga_select_elem");
   GA_PUSH_NAME("ga_elem_op");

   if (strncmp(op,"min",3) == 0);
   else if (strncmp(op,"max",3) == 0);
   else ga_error("operator not recognized",0);

   nga_inquire_(g_a, &type, &ndim, dims);
   nga_distribution_(g_a, &me, lo, hi);

   if ( lo[0]> 0 ){ /* base index is 1: we get 0 if no elements stored on p */

      /******************* calculate local result ************************/
      void    *ptr;
      nga_access_ptr(g_a, lo, hi, &ptr, ld);
      GET_ELEMS(ndim,lo,hi,ld,&elems);
      participate =1;

      switch (type){
        Integer *ia,ival;
        DoublePrecision *da,dval;
        DoubleComplex *ca;

        case MT_F_INT:
           ia = (Integer*)ptr;
           ival = *ia;
          
           if (strncmp(op,"min",3) == 0)
              for(i=0;i<elems;i++){ if(ival > ia[i]) {ival=ia[i];ind=i; } } 
           else
              for(i=0;i<elems;i++){ if(ival < ia[i]) {ival=ia[i];ind=i; } }

           info.v.lval = (long) ival;
           break;

        case MT_F_DCPL:
           ca = (DoubleComplex*)ptr;
           dval=ca->real*ca->real + ca->imag*ca->imag;
           if (strncmp(op,"min",3) == 0)
              for(i=0;i<elems;i++, ca+=sizeof(DoubleComplex) ){
                  DoublePrecision tmp = ca->real*ca->real + ca->imag*ca->imag; 
                  if(dval > tmp){dval = tmp; ind = i;}
              }
           else
              for(i=0;i<elems;i++, ca+=sizeof(DoubleComplex) ){
                  DoublePrecision tmp = ca->real*ca->real + ca->imag*ca->imag; 
                  if(dval < tmp){dval = tmp; ind = i;}
              }
           
           info.v.dval = dval; /* use abs value  for comparison*/
           info.extra = ((DoubleComplex*)ptr)[ind]; /* append the actual val */
           break;

        case MT_F_DBL:
           da = (DoublePrecision*)ptr;
           dval = *da;
           if (strncmp(op,"min",3) == 0)
              for(i=0;i<elems;i++){ if(dval > da[i]) {dval=da[i];ind=i; } }
           else
              for(i=0;i<elems;i++){ if(dval < da[i]) {dval=da[i];ind=i; } }

           info.v.dval = dval; 
           break;

        default: ga_error(" wrong data type ",type);
      }

      /* release access to the data */
      nga_release_(g_a, lo, hi);

      /* determine element subscript in the ndim-array */
      for(i = 0; i < ndim; i++){
          int elems = (int)( hi[i]-lo[i]+1);
          info.subscr[i] = ind%elems + lo[i] ;
          ind /= elems;
      }
   } 

   /* calculate global result */
   if(type==MT_F_INT){ 
      int size = sizeof(double) + sizeof(Integer)*(int)ndim;
      armci_msg_sel(&info,size,op,ARMCI_LONG,participate);
      *(Integer*)val = info.v.lval;
   }else if(type==MT_F_DBL){
      int size = sizeof(double) + sizeof(Integer)*(int)ndim;
      armci_msg_sel(&info,size,op,ARMCI_DOUBLE,participate);
      *(DoublePrecision*)val = info.v.dval;
   }else{
      int size = sizeof(info); /* for simplicity we send entire info */
      armci_msg_sel(&info,size,op,ARMCI_DOUBLE,participate);
      *(DoubleComplex*)val = info.extra;
   }

   for(i = 0; i < ndim; i++) subscript[i]=info.subscr[i];

   GA_POP_NAME;
}
