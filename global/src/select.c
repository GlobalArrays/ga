#include <stdio.h>
#include "global.h"
#include "globalp.h"
#include "../../armci/src/message.h"
  
#if defined(CRAY)
#  include <fortran.h>
#endif

#define GET_ELEMS(ndim,lo,hi,ld,pelems){\
int _i;\
      for(_i=0, *pelems = hi[ndim-1]-lo[ndim-1]+1; _i< ndim-1;_i++) {\
         if(ld[_i] != (hi[_i]-lo[_i]+1)) ga_error("layout problem",_i);\
         *pelems *= hi[_i]-lo[_i]+1;\
      }\
}


/* note that there is no FATR - on windows and cray we call this though a wrapper below */
void nga_select_elem_(Integer *g_a, char* op, void* val, Integer *subscript)
{
Integer ndim, type, me, elems, ind=0, i;
Integer lo[MAXDIM],hi[MAXDIM],dims[MAXDIM],ld[MAXDIM-1];
struct  info_t{ 
        union val_t {double dval; long lval; float fval;}v; 
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

   nga_inquire_internal_(g_a, &type, &ndim, dims);
   nga_distribution_(g_a, &me, lo, hi);

   if ( lo[0]> 0 ){ /* base index is 1: we get 0 if no elements stored on p */

      /******************* calculate local result ************************/
      void    *ptr;
      nga_access_ptr(g_a, lo, hi, &ptr, ld);
      GET_ELEMS(ndim,lo,hi,ld,&elems);
      participate =1;

      switch (type){
        int *ia,ival;
        double *da,dval;
        DoubleComplex *ca;
        float *fa,fval;
        long *la,lval;
        case C_INT:
           ia = (int*)ptr;
           ival = *ia;
          
           if (strncmp(op,"min",3) == 0)
              for(i=0;i<elems;i++){ if(ival > ia[i]) {ival=ia[i];ind=i; } } 
           else
              for(i=0;i<elems;i++){ if(ival < ia[i]) {ival=ia[i];ind=i; } }

           info.v.lval = (long) ival;
	   break;

        case C_DCPL:
           ca = (DoubleComplex*)ptr;
           dval=ca->real*ca->real + ca->imag*ca->imag;
           if (strncmp(op,"min",3) == 0)
              for(i=0;i<elems;i++, ca+=1 ){
                  DoublePrecision tmp = ca->real*ca->real + ca->imag*ca->imag; 
                  if(dval > tmp){dval = tmp; ind = i;}
              }
           else
              for(i=0;i<elems;i++, ca+=1 ){
                  DoublePrecision tmp = ca->real*ca->real + ca->imag*ca->imag; 
                  if(dval < tmp){dval = tmp; ind = i;}
              }
           
           info.v.dval = dval; /* use abs value  for comparison*/
           info.extra = ((DoubleComplex*)ptr)[ind]; /* append the actual val */
           break;

        case C_DBL:
           da = (double*)ptr;
           dval = *da;
           if (strncmp(op,"min",3) == 0)
              for(i=0;i<elems;i++){ if(dval > da[i]) {dval=da[i];ind=i; } }
           else
              for(i=0;i<elems;i++){ if(dval < da[i]) {dval=da[i];ind=i; } }

           info.v.dval = dval; 
           break;

        case C_FLOAT:
           fa = (float*)ptr;
           fval = *fa;
 
           if (strncmp(op,"min",3) == 0)
              for(i=0;i<elems;i++){ if(fval > fa[i]) {fval=fa[i];ind=i; } }
           else
              for(i=0;i<elems;i++){ if(fval < fa[i]) {fval=fa[i];ind=i; } }
 
           info.v.fval = fval;
           break;
        case C_LONG:
           la = (long*)ptr;
           lval = *la;
 
           if (strncmp(op,"min",3) == 0)
              for(i=0;i<elems;i++){ if(lval > la[i]) {lval=la[i];ind=i; } }
           else
              for(i=0;i<elems;i++){ if(lval < la[i]) {lval=la[i];ind=i; } }
 
           info.v.lval = lval;
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
   if(type==C_INT || type==C_LONG){ 
      int size = sizeof(double) + sizeof(Integer)*(int)ndim;
      armci_msg_sel(&info,size,op,ARMCI_LONG,participate);
      *(Integer*)val = info.v.lval;
   }else if(type==C_DBL){
      int size = sizeof(double) + sizeof(Integer)*(int)ndim;
      armci_msg_sel(&info,size,op,ARMCI_DOUBLE,participate);
      *(DoublePrecision*)val = info.v.dval;
   }else if(type==C_FLOAT){
      int size = sizeof(double) + sizeof(Integer)*ndim;
      armci_msg_sel(&info,size,op,ARMCI_DOUBLE,participate);
      *(float*)val = info.v.fval;       
   }else{
      int size = sizeof(info); /* for simplicity we send entire info */
      armci_msg_sel(&info,size,op,ARMCI_DOUBLE,participate);
      *(DoubleComplex*)val = info.extra;
   }

   for(i = 0; i < ndim; i++) subscript[i]=info.subscr[i];
   GA_POP_NAME;
}

#if defined(CRAY) || defined(WIN32)  
void FATR NGA_SELECT_ELEM(Integer *g_a, _fcd op, void* val, Integer *subscript)
{
     nga_select_elem_(g_a,_fcdtocp(op), val, subscript);
}
#endif
