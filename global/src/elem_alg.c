/**************************************************************
File: elem.alg.c

Elementwise operations on patches and whole arrays

Author: Limin Zhang, Ph.D.
	Mathematics Department
        Columbia Basin College
        Pasco, WA 99301
        Limin.Zhang@cbc2.org

Mentor: Jarek Naplocha, Ph.D.
  	Environmental Molecular Science Laboratory
        Pacific Northwest National Laboratory
	Richland, WA 99352

Date: 1/18/2002
    
Purpose:
      to design and implement some interfaces between TAO and
      global arrays.
**************************************************************/

#include "global.h"
#include "globalp.h"
#include <math.h>

#ifndef GA_INFINITY_I
#define GA_INFINITY_I 100000
#endif

#ifndef GA_NEGATIVE_INFINITY_I
#define GA_NEGATIVE_INFINITY_I -100000
#endif

#ifndef GA_INFINITY_L
#define GA_INFINITY_L 100000
#endif

#ifndef GA_NEGATIVE_INFINITY_L
#define GA_NEGATIVE_INFINITY_L -100000
#endif

#ifndef GA_INFINITY
#define GA_INFINITY 1.0e20
#endif

#ifndef GA_NEGATIVE_INFINITY
#define GA_NEGATIVE_INFINITY -1.0e20
#endif

#define OP_ABS 0
#define OP_ADD_CONST 1
#define OP_RECIP 2
#define OP_ELEM_MULT 3
#define OP_ELEM_DIV 4
#define OP_ELEM_MAX 5
#define OP_ELEM_MIN 6
#define OP_STEPMAX 7
#define OP_STEPMAX2 8
#define OP_FILL 100 /*The OP_FILL is not currently in use */

int debug_gai_oper_elem = 1;

static void do_stepmax2(void *ptr, int nelem, int type)
/*look at elements one by one and replace the positive infinity with negative infinity */ 
{
    int i;
    switch (type){
         int *ia;
         double *da;
         float *fa;
         DoubleComplex *ca,val;
	 long *la;

  	 case C_DBL:
                /*Only double data type will be handled for TAO/GA project*/ 
              da = (double *) ptr;
              for(i=0;i<nelem;i++)
                  if(da[i]>=GA_INFINITY) da[i]=-GA_INFINITY;
              break;
         case C_INT:
         case C_DCPL:
         case C_FLOAT:
 	case C_LONG:
         default: ga_error("do_stepmax2:wrong data type",type);
    }
}
static void do_stepmax(void *ptr, int nelem, int type)
/*look at elements one by one and replace the positive with negative infinity */ 
{
    int i;
    switch (type){
         int *ia;
         double *da;
         float *fa;
         DoubleComplex *ca,val;
	 long *la;

  	 case C_DBL:
                /*Only double data type will be handled for TAO/GA project*/ 
              da = (double *) ptr;
              for(i=0;i<nelem;i++)
                  if(da[i]>0) da[i]=-GA_INFINITY;
              break;
         case C_INT:
         case C_DCPL:
         case C_FLOAT:
 	case C_LONG:
         default: ga_error("do_stepmax:wrong data type",type);
    }
}


static void do_abs(void *ptr, int nelem, int type)
{
    int i;
    switch (type){
         int *ia;
         double *da;
         float *fa;
         DoubleComplex *ca,val;
	 long *la;

         case C_INT:
              ia = (int *)ptr; 
              for(i=0;i<nelem;i++)
                  ia[i]= ABS(ia[i]);
              break; 
         case C_DCPL:
              ca = (DoubleComplex *) ptr;
              for(i=0;i<nelem;i++){
                  val = ca[i];
                  ca[i].real = sqrt(val.real * val.real + val.imag *val.imag);
                  ca[i].imag = 0.0;
              }
              break;
  	 case C_DBL:
              da = (double *) ptr;
              for(i=0;i<nelem;i++)
                  da[i]= ABS(da[i]);
              break;
         case C_FLOAT:
              fa = (float *)ptr;
              for(i=0;i<nelem;i++)
                  fa[i]= ABS(fa[i]);
              break;
 	case C_LONG:
              la = (long *)ptr;
              for(i=0;i<nelem;i++)
                  la[i]= ABS(la[i]);
              break;

         default: ga_error("wrong data type",type);
    }
} 

static void do_recip(void *ptr, int nelem, int type)
{
    int i;
    switch (type){
         int *ia;
         double *da, temp;
         float *fa;
         DoubleComplex *ca,val;
         long *la; 

         case C_INT:
              ia = (int *)ptr;
              for(i=0;i<nelem;i++)
                  if(ia[i]!=0) ia[i]= 1/ia[i];
                     else
 		  ia[i] = GA_INFINITY_I;
                  //ga_error("zero value at index",i);  
              break;
         case C_DCPL:
              ca = (DoubleComplex *) ptr;
              for(i=0;i<nelem;i++){
                  temp = ca[i].real*ca[i].real + ca[i].imag*ca[i].imag;
                  if( temp!=0.0){
                   ca[i].real =ca[i].real/temp;
                   ca[i].imag =-ca[i].imag/temp;
                  }
                  else{
 		     //ga_error("zero value at index",i);
 		     ca[i].real = GA_INFINITY;
 		     ca[i].imag = GA_INFINITY;
                 }
              }
              break;
         case C_DBL:
              da = (double *) ptr;
              for(i=0;i<nelem;i++)
                  if(da[i]!=0.0) da[i]= (double)1/da[i];
  		     else
		  //ga_error("zero value at index",i);
 		  da[i] = GA_INFINITY;
              break;
         case C_FLOAT:
              fa = (float *)ptr;
              for(i=0;i<nelem;i++)
                  if(fa[i]!=0.0) fa[i]= (float)1/fa[i];
                     else
		  //ga_error("zero value at index",i);		
 		  fa[i] = GA_INFINITY;
              break;
	case C_LONG:
              la = (long *)ptr;
              for(i=0;i<nelem;i++)
                  if(la[i]!=0.0) la[i]= (long)1/la[i];
                     else
                  //ga_error("zero value at index",i);
 		  la[i] = GA_INFINITY_I;
              break;


         default: ga_error("wrong data type",type);
    }
} 

static void do_add_const(void *ptr, int nelem, int type, void *alpha)
{
    int i;
    switch (type){
         int *ia;
         double *da;
         float *fa;
         DoubleComplex *ca,val;
	 long *la;

         case C_INT:
              ia = (int *)ptr;
              for(i=0;i<nelem;i++)
                  ia[i] += *(int *)alpha;
              break;
         case C_DCPL:
              ca = (DoubleComplex *) ptr;
              for(i=0;i<nelem;i++){
                  val = *(DoubleComplex*)alpha;
                  ca[i].real += val.real;
                  ca[i].imag += val.imag;
              }
              break;
         case C_DBL:
              da = (double *) ptr;
              for(i=0;i<nelem;i++)
                  da[i] += *(double*)alpha;
              break;
         case C_FLOAT:
              fa = (float *)ptr;
              for(i=0;i<nelem;i++)
                  fa[i] += *(float*)alpha;
              break;
	 case C_LONG:
              la = (long *)ptr;
              for(i=0;i<nelem;i++)
                  la[i] += *(long *)alpha;
              break;

         default: ga_error("wrong data type",type);
    }
} 

/*
void do_fill(void *ptr, int nelem, int type, void *alpha)
{
    int i;
    switch (type){
         int *ia;
         double *da;
         float *fa;
         DoubleComplex *ca,val;
         long *la;

         case C_INT:
              ia = (int *)ptr;
              for(i=0;i<nelem;i++)
                  ia[i] = *(int *)alpha;
              break;
         case C_DCPL:
              ca = (DoubleComplex *) ptr;
              for(i=0;i<nelem;i++){
                  val = *(DoubleComplex*)alpha;
                  ca[i].real = val.real;
                  ca[i].imag = val.imag;
              }
              break;
         case C_DBL:
              da = (double *) ptr;
              for(i=0;i<nelem;i++)
                  da[i] = *(double*)alpha;
              break;
         case C_FLOAT:
              fa = (float *)ptr;
              for(i=0;i<nelem;i++)
                  fa[i] = *(float*)alpha;
              break;
	 case C_LONG:
              la = (long *)ptr;
              for(i=0;i<nelem;i++)
                  la[i] = *(long *)alpha;
              break;

         default: ga_error("wrong data type",type);
    }
} 
*/

/*
Input Parameters

int *g_a -- the global array handle
int *lo, *hi--the integer arrays that define the patch of the global array
void *scalar -- the pointer that points to the data to pass. When it is NULL, no scalar will be passed.
int op -- the operations to handle. For example op can be  

OP_ABS for pointwise taking absolute function 
OP_ADD_CONSTANT 2 for pointwise adding the same constant
OP_RECIP for pointwise taking reciprocal
OP_FILL for pointwise filling value 

Output Parameters

None

*/

static void FATR gai_oper_elem(Integer *g_a, Integer *lo, Integer *hi, void *scalar, Integer op)
{

    Integer i, j;    
    Integer ndim, dims[MAXDIM], type;
    Integer loA[MAXDIM], hiA[MAXDIM], ld[MAXDIM];
    void *temp, *data_ptr;
    Integer idx, n1dim;
    Integer bvalue[MAXDIM], bunit[MAXDIM], baseld[MAXDIM];
    Integer me= ga_nodeid_();
    
    ga_sync_();
    ga_check_handle(g_a, "gai_oper_elem");

    GA_PUSH_NAME("gai_oper_elem");
    
    nga_inquire_internal_(g_a,  &type, &ndim, dims);
    
    /* get limits of VISIBLE patch */
    nga_distribution_(g_a, &me, loA, hiA);
    
    /*  determine subset of my local patch to access  */
    /*  Output is in loA and hiA */
    if(ngai_patch_intersect(lo, hi, loA, hiA, ndim)){
    
        /* get data_ptr to corner of patch */
        /* ld are leading dimensions INCLUDING ghost cells */
        nga_access_ptr(g_a, loA, hiA, &data_ptr, ld);
        
        /* number of n-element of the first dimension */
        n1dim = 1; for(i=1; i<ndim; i++) n1dim *= (hiA[i] - loA[i] + 1);
        
        /* calculate the destination indices */
        bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
        /* baseld[0] = ld[0]
         * baseld[1] = ld[0] * ld[1]
         * baseld[2] = ld[0] * ld[1] * ld[2] ....
         */
        baseld[0] = ld[0]; baseld[1] = baseld[0] *ld[1];
        for(i=2; i<ndim; i++) {
            bvalue[i] = 0;
            bunit[i] = bunit[i-1] * (hiA[i-1] - loA[i-1] + 1);
            baseld[i] = baseld[i-1] * ld[i];
        }

       for(i=0; i<n1dim; i++) {
                idx = 0;
                for(j=1; j<ndim; j++) {
                    idx += bvalue[j] * baseld[j-1];
                    if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                    if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
                }

   		switch(type){
			case C_INT: 
		           temp=((int*)data_ptr)+idx; 
			   break;
			case C_DCPL: 
		           temp=((DoubleComplex*)data_ptr)+idx; 
			    break;
			case C_DBL: 
		           temp=((double*)data_ptr)+idx; 
			    break;
		        case C_FLOAT:
                           temp=((float*)data_ptr)+idx;
                            break;
 		     	case C_LONG:
                           temp=((long *)data_ptr)+idx;
                            break;
			default: ga_error("wrong data type.",type);	
	
               }

                    switch(op){
	              case OP_ABS:
		           do_abs(temp,hiA[0] -loA[0] +1, type); break;
                           break;
		      case OP_ADD_CONST:
		           do_add_const(temp,hiA[0] -loA[0] +1, type, scalar); 
			   break;
	              case OP_RECIP:
		           do_recip(temp,hiA[0] -loA[0] +1, type); break;
                           break;
		      default: ga_error("bad operation",op);
                    }
        }

        /* release access to the data */
        nga_release_update_(g_a, loA, hiA);
     }
    GA_POP_NAME;
    ga_sync_();
}





void ga_abs_value_patch_(Integer *g_a, Integer *lo, Integer *hi)
{
    gai_oper_elem(g_a, lo, hi, NULL, OP_ABS);
}

void ga_recip_patch_(Integer *g_a, Integer *lo, Integer *hi)
{
    gai_oper_elem(g_a, lo, hi, NULL, OP_RECIP);

}

void ga_add_constant_patch_(Integer *g_a, Integer *lo, Integer *hi, void *alpha)
{
    gai_oper_elem(g_a, lo, hi, alpha, OP_ADD_CONST);

}

void ga_abs_value_(Integer *g_a)
{
   Integer type, ndim;
   Integer lo[MAXDIM],hi[MAXDIM];

    nga_inquire_internal_(g_a,  &type, &ndim, hi);
    while(ndim){
        lo[ndim-1]=1;
        ndim--;
    }
    gai_oper_elem(g_a, lo, hi, NULL, OP_ABS);
}

void ga_add_constant_(Integer *g_a, void *alpha)
{
   Integer type, ndim;
   Integer lo[MAXDIM],hi[MAXDIM];

    nga_inquire_internal_(g_a,  &type, &ndim, hi);
    while(ndim){
        lo[ndim-1]=1;
        ndim--;
    }
    gai_oper_elem(g_a, lo, hi, alpha, OP_ADD_CONST);
}

void ga_recip_(Integer *g_a)
{        
   Integer type, ndim;
   Integer lo[MAXDIM],hi[MAXDIM];
        
    nga_inquire_internal_(g_a,  &type, &ndim, hi);
    while(ndim){
        lo[ndim-1]=1; 
        ndim--;
    }
    gai_oper_elem(g_a, lo, hi, NULL, OP_RECIP);
}    



static void do_multiply(void *pA, void *pB, void *pC, Integer nelems, Integer type){
  Integer i;
  
  switch(type){
    double aReal, aImag, bReal, bImag, cReal, cImag;
    
  case C_DBL:
    for(i = 0; i<nelems; i++)
      ((double*)pC)[i]= ((double*)pA)[i]*((double*)pB)[i]; 
    break;
  case C_DCPL:
    for(i = 0; i<nelems; i++) {
      aReal = ((DoubleComplex*)pA)[i].real; 
      bReal = ((DoubleComplex*)pB)[i].real; 
      aImag = ((DoubleComplex*)pA)[i].imag; 
      bImag = ((DoubleComplex*)pB)[i].imag; 
      ((DoubleComplex*)pC)[i].real = aReal*bReal-aImag*bImag;
      ((DoubleComplex*)pC)[i].imag = aReal*bImag+aImag*bReal;
    }
    break;
  case C_INT:
    for(i = 0; i<nelems; i++)
      ((int*)pC)[i] = ((int*)pA)[i]* ((int*)pB)[i];
    break;
  case C_FLOAT:
    for(i = 0; i<nelems; i++)
      ((float*)pC)[i]=  ((float*)pA)[i]*((float*)pB)[i];
    break;
  case C_LONG:
    for(i = 0; i<nelems; i++)
      ((long *)pC)[i]= ((long *)pA)[i]* ((long *)pB)[i];
    break;
    
  default: ga_error(" wrong data type ",type);
  }
}


static void do_divide(void *pA, void *pB, void *pC, Integer nelems, Integer type){
  Integer i;
  double aReal, aImag, bReal, bImag, cReal, cImag;
  double temp;

  switch(type){
  
  case C_DBL:
    for(i = 0; i<nelems; i++) {
      if(((double*)pB)[i]!=0.0)
	((double*)pC)[i]=  ((double*)pA)[i]/((double*)pB)[i];
      else{
	if(((double*)pA)[i]>=0)
	  ((double*)pC)[i]=  GA_INFINITY;
	else
	  ((double*)pC)[i]=  GA_NEGATIVE_INFINITY;
	//ga_error("zero divisor ",((double*)pB)[i]);
      }
    }
    break;
  case C_DCPL:
    for(i = 0; i<nelems; i++) {
      aReal = ((DoubleComplex*)pA)[i].real;
      bReal = ((DoubleComplex*)pB)[i].real;
      aImag = ((DoubleComplex*)pA)[i].imag;
      bImag = ((DoubleComplex*)pB)[i].imag;
      temp = bReal*bReal+bImag*bImag;
      if(temp!=0.0){
	((DoubleComplex*)pC)[i].real
	  =(aReal*bReal+aImag*bImag)/temp;
	((DoubleComplex*)pC)[i].imag
	  =(aImag*bReal-aReal*bImag)/temp;
      }
      else{
	((DoubleComplex*)pC)[i].real=GA_INFINITY;
	((DoubleComplex*)pC)[i].imag=GA_INFINITY;
	//ga_error("zero divisor ",temp);
      }
    }
    break;
  case C_INT:
    for(i = 0; i<nelems; i++){
      if(((int*)pB)[i]!=0)
	((int*)pC)[i] = ((int*)pA)[i]/((int*)pB)[i];
      else{
	if(((int*)pA)[i]>=0)
	  ((int*)pC)[i]=GA_INFINITY_I;
	else
	  ((int*)pC)[i]=GA_NEGATIVE_INFINITY_I;
	//ga_error("zero divisor ",((int*)pB)[i]);
      } 
    }
    break;
  case C_FLOAT:
    for(i = 0; i<nelems; i++){
      if(((float*)pB)[i]!=0.0) 
	((float*)pC)[i]=  ((float*)pA)[i]/((float*)pB)[i];
      else{
	if(((float*)pA)[i]>=0)
	  ((float*)pC)[i]= GA_INFINITY;
	else
	  ((float*)pC)[i]= GA_NEGATIVE_INFINITY;
	//ga_error("zero divisor ",((float*)pB)[i]);
      }
    }
    break;
  case C_LONG:
    for(i = 0; i<nelems; i++){
      if(((long *)pB)[i]!=0)
	((long *)pC)[i]=  ((long *)pA)[i]/((long *)pB)[i];
      else{
	if(((long *)pA)[i]>=0)
	  ((long *)pC)[i] = GA_INFINITY_L;
	else
	  ((long *)pC)[i] = GA_NEGATIVE_INFINITY_L;
	//ga_error("zero divisor ",((long*)pB)[i]);
      }
    }
    break;		
  default: ga_error(" wrong data type ",type);
  }
}
 




static void do_maximum(void *pA, void *pB, void *pC, Integer nelems, Integer type){
  Integer i;
  double aReal, aImag, bReal, bImag, cReal, cImag, temp1, temp2;

  switch(type){
    
  case C_DBL:
    for(i = 0; i<nelems; i++)
      ((double*)pC)[i] = MAX(((double*)pA)[i],((double*)pB)[i]);
    break;
  case C_DCPL:
    for(i = 0; i<nelems; i++) {
      aReal = ((DoubleComplex*)pA)[i].real;
      bReal = ((DoubleComplex*)pB)[i].real;
      aImag = ((DoubleComplex*)pA)[i].imag;
      bImag = ((DoubleComplex*)pB)[i].imag;
      temp1 = aReal*aReal+aImag*aImag;
      temp2 = bReal*bReal+bImag*bImag;
      if(temp1>temp2){
	((DoubleComplex*)pC)[i].real=((DoubleComplex*)pA)[i].real;
	((DoubleComplex*)pC)[i].imag=((DoubleComplex*)pA)[i].imag;
      }
      else{
	((DoubleComplex*)pC)[i].real=((DoubleComplex*)pB)[i].real;
	((DoubleComplex*)pC)[i].imag=((DoubleComplex*)pB)[i].imag;
      }
    }
    break;
  case C_INT:
    for(i = 0; i<nelems; i++)
      ((int*)pC)[i] =MAX(((int*)pA)[i],((int*)pB)[i]);
    break;
  case C_FLOAT:
    for(i = 0; i<nelems; i++)
      ((float*)pC)[i]=MAX(((float*)pA)[i],((float*)pB)[i]);
    break;
    
  case C_LONG:
    for(i = 0; i<nelems; i++)
      ((long *)pC)[i]=MAX(((long *)pA)[i],((long *)pB)[i]);
    break;
    
  default: ga_error(" wrong data type ",type);
  }
}


static void do_minimum(void *pA, void *pB, void *pC, Integer nelems, Integer type){
  Integer i;

  switch(type){
    double aReal, aImag, bReal, bImag, cReal, cImag, temp1, temp2;
    
  case C_DBL:
    for(i = 0; i<nelems; i++)
      ((double*)pC)[i] = MIN(((double*)pA)[i],((double*)pB)[i]);
    break;
  case C_DCPL:
    for(i = 0; i<nelems; i++) {
      aReal = ((DoubleComplex*)pA)[i].real;
      bReal = ((DoubleComplex*)pB)[i].real;
      aImag = ((DoubleComplex*)pA)[i].imag;
      bImag = ((DoubleComplex*)pB)[i].imag;
      temp1 = aReal*aReal+aImag*aImag;
      temp2 = bReal*bReal+bImag*bImag;
      if(temp1<temp2){ 
	((DoubleComplex*)pC)[i].real=((DoubleComplex*)pA)[i].real; 
	((DoubleComplex*)pC)[i].imag=((DoubleComplex*)pA)[i].imag; 
      } 
      else{ 
	((DoubleComplex*)pC)[i].real=((DoubleComplex*)pB)[i].real; 
	((DoubleComplex*)pC)[i].imag=((DoubleComplex*)pB)[i].imag; 
      }
    }
    break;
  case C_INT:
    for(i = 0; i<nelems; i++)
      ((int*)pC)[i] =MIN(((int*)pA)[i],((int*)pB)[i]);
    break;
  case C_FLOAT:
    for(i = 0; i<nelems; i++)
      ((float*)pC)[i]=MIN(((float*)pA)[i],((float*)pB)[i]);
    break;
  case C_LONG:
    for(i = 0; i<nelems; i++)
      ((long *)pC)[i]=MIN(((long *)pA)[i],((long *)pB)[i]);
    break;
    
  default: ga_error(" wrong data type ",type);
  }
} 


/*\  generic operation of two patches
\*/
static void FATR ngai_elem2_patch_(g_a, alo, ahi, g_b, blo, bhi,
                         g_c, clo, chi, op)
Integer *g_a, *alo, *ahi;    /* patch of g_a */
Integer *g_b, *blo, *bhi;    /* patch of g_b */
Integer *g_c, *clo, *chi;    /* patch of g_c */
Integer op; //operation to be perform between g_a and g_b
{
    Integer i, j;
    Integer compatible;
    Integer atype, btype, ctype;
    Integer andim, adims[MAXDIM], bndim, bdims[MAXDIM], cndim, cdims[MAXDIM];
    Integer loA[MAXDIM], hiA[MAXDIM], ldA[MAXDIM];
    Integer loB[MAXDIM], hiB[MAXDIM], ldB[MAXDIM];
    Integer loC[MAXDIM], hiC[MAXDIM], ldC[MAXDIM];
    void *A_ptr, *B_ptr, *C_ptr, *tempA, *tempB, *tempC;
    Integer bvalue[MAXDIM], bunit[MAXDIM], baseldC[MAXDIM];
    Integer idx, n1dim;
    Integer atotal, btotal;
    Integer g_A = *g_a, g_B = *g_b;
    Integer me= ga_nodeid_(), A_created=0, B_created=0;
    char *tempname = "temp", notrans='n';

    ga_sync_();
    ga_check_handle(g_a, "gai_elem2_patch_");
    GA_PUSH_NAME("ngai_elem2_patch_");

    nga_inquire_internal_(g_a, &atype, &andim, adims);
    nga_inquire_internal_(g_b, &btype, &bndim, bdims);
    nga_inquire_internal_(g_c, &ctype, &cndim, cdims);

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
        if(*g_b != *g_c) {
            nga_copy_patch(&notrans, g_a, alo, ahi, g_c, clo, chi);
            andim = cndim;
            g_A = *g_c;
            nga_distribution_(&g_A, &me, loA, hiA);
        }
        else {
            if (!ga_duplicate(g_c, &g_A, tempname))
            ga_error("ga_dadd_patch: dup failed", 0L);
            nga_copy_patch(&notrans, g_a, alo, ahi, &g_A, clo, chi);
            andim = cndim;
            A_created = 1;
            nga_distribution_(&g_A, &me, loA, hiA);
        }
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
        
        /* compute "local" operation accoording to op */

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
        

         for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<cndim; j++) {
                        idx += bvalue[j] * baseldC[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiC[j]-loC[j])) bvalue[j] = 0;
                    }

                    switch(atype){
                      	case C_DBL:
                       	tempA=((double*)A_ptr)+idx;
                        tempB=((double*)B_ptr)+idx;
                    	tempC=((double*)C_ptr)+idx;
    			break;
            		case C_DCPL:
                    	tempA=((DoubleComplex*)A_ptr)+idx;
                    	tempB=((DoubleComplex*)B_ptr)+idx;
                    	tempC=((DoubleComplex*)C_ptr)+idx;
		        break;
                        case C_INT:
                    	tempA=((int*)A_ptr)+idx;
                    	tempB=((int*)B_ptr)+idx;
                    	tempC=((int*)C_ptr)+idx;
		        break;
			case C_FLOAT:
                    	tempA=((float*)A_ptr)+idx;
                    	tempB=((float*)B_ptr)+idx;
                    	tempC=((float*)C_ptr)+idx;
			break;
			case C_LONG:
                        tempA=((long *)A_ptr)+idx;
                        tempB=((long *)B_ptr)+idx;
                        tempC=((long *)C_ptr)+idx;
                        break;

                       default: ga_error(" wrong data type ",atype);
                   }   
                   switch(op){
                        case OP_ELEM_MULT:
                          do_multiply(tempA,tempB,tempC,hiC[0]-loC[0]+1,atype);
                           break;
                        case OP_ELEM_DIV:
                          do_divide(tempA,tempB,tempC,hiC[0]-loC[0]+1,atype);
                           break;
                        case  OP_ELEM_MAX:
                           do_maximum(tempA,tempB,tempC,hiC[0]-loC[0]+1,atype);
                           break;
                        case  OP_ELEM_MIN:
                           do_minimum(tempA,tempB,tempC,hiC[0]-loC[0]+1,atype);
                           break;
                        default: ga_error(" wrong operation ",op);
                   }
        }
        
        /* release access to the data */
        nga_release_       (&g_A, loC, hiC);
        nga_release_       (&g_B, loC, hiC); 
        nga_release_update_( g_c, loC, hiC); 

    }

    if(A_created) ga_destroy_(&g_A);
    if(B_created) ga_destroy_(&g_B);
    
    GA_POP_NAME;
    ga_sync_();
}

void ga_elem_multiply_(Integer *g_a, Integer *g_b, Integer *g_c){
 
   Integer atype, andim;
   Integer btype, bndim;
   Integer ctype, cndim;
   Integer alo[MAXDIM],ahi[MAXDIM];
   Integer blo[MAXDIM],bhi[MAXDIM];
   Integer clo[MAXDIM],chi[MAXDIM];
 
    nga_inquire_internal_(g_a,  &atype, &andim, ahi);
    nga_inquire_internal_(g_b,  &btype, &bndim, bhi);
    nga_inquire_internal_(g_c,  &ctype, &cndim, chi);
    if((andim!=bndim)||(andim!=cndim))
	ga_error("global arrays have different dimmensions.", andim);
    while(andim){
        alo[andim-1]=1;
        blo[bndim-1]=1;
        clo[cndim-1]=1;
        andim--;
        bndim--;
        cndim--;
    }
    ngai_elem2_patch_(g_a, alo, ahi, g_b, blo, bhi,g_c,clo,chi,OP_ELEM_MULT);

}


void ga_elem_divide_(Integer *g_a, Integer *g_b, Integer *g_c){
 
   Integer atype, andim;
   Integer btype, bndim;
   Integer ctype, cndim;
   Integer alo[MAXDIM],ahi[MAXDIM];
   Integer blo[MAXDIM],bhi[MAXDIM];
   Integer clo[MAXDIM],chi[MAXDIM];
 
    nga_inquire_internal_(g_a,  &atype, &andim, ahi);
    nga_inquire_internal_(g_b,  &btype, &bndim, bhi);
    nga_inquire_internal_(g_c,  &ctype, &cndim, chi);
    if((andim!=bndim)||(andim!=cndim))
        ga_error("global arrays have different dimmensions.", andim);
    while(andim){
        alo[andim-1]=1;
        blo[bndim-1]=1;
        clo[cndim-1]=1;
        andim--;
        bndim--;
        cndim--;
    }

  ngai_elem2_patch_(g_a, alo, ahi, g_b, blo, bhi,g_c,clo,chi,OP_ELEM_DIV);
 
}

 


void ga_elem_maximum_(Integer *g_a, Integer *g_b, Integer *g_c){

   Integer atype, andim;
   Integer btype, bndim;
   Integer ctype, cndim;
   Integer alo[MAXDIM],ahi[MAXDIM];
   Integer blo[MAXDIM],bhi[MAXDIM];
   Integer clo[MAXDIM],chi[MAXDIM];

    nga_inquire_internal_(g_a,  &atype, &andim, ahi);
    nga_inquire_internal_(g_b,  &btype, &bndim, bhi);
    nga_inquire_internal_(g_c,  &ctype, &cndim, chi);
    if((andim!=bndim)||(andim!=cndim))
        ga_error("global arrays have different dimmensions.", andim);
    while(andim){
        alo[andim-1]=1;
        blo[bndim-1]=1;
        clo[cndim-1]=1;
        andim--;
        bndim--;
        cndim--;
    }

    ngai_elem2_patch_(g_a, alo, ahi, g_b, blo, bhi,g_c,clo,chi,OP_ELEM_MAX);

}

 
void ga_elem_minimum_(Integer *g_a, Integer *g_b, Integer *g_c){
 
   Integer atype, andim;
   Integer btype, bndim;
   Integer ctype, cndim;
   Integer alo[MAXDIM],ahi[MAXDIM];
   Integer blo[MAXDIM],bhi[MAXDIM];
   Integer clo[MAXDIM],chi[MAXDIM];
 
    nga_inquire_internal_(g_a,  &atype, &andim, ahi);
    nga_inquire_internal_(g_b,  &btype, &bndim, bhi);
    nga_inquire_internal_(g_c,  &ctype, &cndim, chi);
    if((andim!=bndim)||(andim!=cndim))
        ga_error("global arrays have different dimmensions.", andim);
    while(andim){
        alo[andim-1]=1;
        blo[bndim-1]=1;
        clo[cndim-1]=1;
        andim--;
        bndim--;
        cndim--;
    }
 
    ngai_elem2_patch_(g_a, alo, ahi, g_b, blo, bhi,g_c,clo,chi,OP_ELEM_MIN);
 
}
 
void ga_elem_multiply_patch_(Integer *g_a,Integer *alo,Integer *ahi,Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c,Integer *clo,Integer *chi){

    ngai_elem2_patch_(g_a, alo, ahi, g_b, blo, bhi,g_c,clo,chi,OP_ELEM_MULT);

}

void ga_elem_divide_patch_(Integer *g_a,Integer *alo,Integer *ahi,
Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c, Integer *clo,Integer *chi){

    ngai_elem2_patch_(g_a, alo, ahi, g_b, blo, bhi,g_c,clo,chi,OP_ELEM_DIV);

}

void ga_elem_maximum_patch_(Integer *g_a,Integer *alo,Integer *ahi,
Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c,Integer *clo,Integer *chi){

    ngai_elem2_patch_(g_a, alo, ahi, g_b, blo, bhi,g_c,clo,chi,OP_ELEM_MAX);

}

void ga_elem_minimum_patch_(Integer *g_a,Integer *alo,Integer *ahi,
Integer *g_b,Integer *blo,Integer *bhi,Integer *g_c,Integer *clo,Integer *chi){

    ngai_elem2_patch_(g_a, alo, ahi, g_b, blo, bhi,g_c,clo,chi,OP_ELEM_MIN);

}

static void FATR ngai_elem3_patch_(g_a, alo, ahi, op)
Integer *g_a, *alo, *ahi;    /* patch of g_a */
Integer op; /*operation to be perform on g_a*/
/*do some preprocess jobs for stepMax and stepMax2*/
{
    Integer i, j;
    Integer atype;
    Integer andim, adims[MAXDIM];
    Integer loA[MAXDIM], hiA[MAXDIM], ldA[MAXDIM];
    void *A_ptr, *tempA;
    Integer bvalue[MAXDIM], bunit[MAXDIM], baseldA[MAXDIM];
    Integer idx, n1dim;
    Integer atotal;
    Integer me= ga_nodeid_();
 
    ga_sync_();
    ga_check_handle(g_a, "gai_elem3_patch_");
    GA_PUSH_NAME("ngai_elem3_patch_");
 
    nga_inquire_internal_(g_a, &atype, &andim, adims);

 /* check if patch indices and dims match */
    for(i=0; i<andim; i++)
        if(alo[i] <= 0 || ahi[i] > adims[i])
            ga_error("g_a indices out of range ", *g_a);
   
    /* find out coordinates of patches of g_a, g_b and g_c that I own */
    nga_distribution_(g_a, &me, loA, hiA);

   /*  determine subsets of my patches to access  */
    if (ngai_patch_intersect(alo, ahi, loA, hiA, andim)){
        nga_access_ptr(g_a, loA, hiA, &A_ptr, ldA);
   
        /* compute "local" operation accoording to op */
 
        /* number of n-element of the first dimension */
        n1dim = 1; for(i=1; i<andim; i++) n1dim *= (hiA[i] - loA[i] + 1);
 
        /* calculate the destination indices */
        bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
        /* baseld[0] = ld[0]
         * baseld[1] = ld[0] * ld[1]
         * baseld[2] = ld[0] * ld[1] * ld[2] .....
         */
        baseldA[0] = ldA[0]; baseldA[1] = baseldA[0] *ldA[1];
        for(i=2; i<andim; i++) {
            bvalue[i] = 0;
            bunit[i] = bunit[i-1] * (hiA[i-1] - loA[i-1] + 1);
            baseldA[i] = baseldA[i-1] * ldA[i];
        }
 
         for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<andim; j++) {
                        idx += bvalue[j] * baseldA[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
                    }
 
                    switch(atype){
                        case C_DBL:
                        /*double is the only type that is handled for Tao/GA project*/
                        tempA=((double*)A_ptr)+idx;
                        break;
                        case C_DCPL:
                        tempA=((DoubleComplex*)A_ptr)+idx;
                        case C_INT:
                        tempA=((int*)A_ptr)+idx;
                        case C_FLOAT:
                        tempA=((float*)A_ptr)+idx;
                        case C_LONG:
                        tempA=((long *)A_ptr)+idx;
 
                       default: ga_error(" ngai_elem3_patch_: wrong data type ",atype);
                   }

        	   switch(op){
                        case  OP_STEPMAX:
                           do_stepmax(tempA,hiA[0]-loA[0]+1, atype);
                           break;
                        case  OP_STEPMAX2:
                           do_stepmax2(tempA,hiA[0]-loA[0]+1, atype);
                           break;
                        default: ga_error(" wrong operation ",op);
                   }
        }
 
        /* release access to the data */
        nga_release_ (g_a, loA, hiA);
    }
 
    GA_POP_NAME;
    ga_sync_();
}

static Integer FATR has_negative_elem(g_a, alo, ahi)
Integer *g_a, *alo, *ahi;    /* patch of g_a */
/*returned value: 1=found; 0 = not found*/
{
    Integer i, j;
    Integer atype;
    Integer andim, adims[MAXDIM];
    Integer loA[MAXDIM], hiA[MAXDIM], ldA[MAXDIM];
    void *A_ptr; 
    double *tempA;
    Integer bvalue[MAXDIM], bunit[MAXDIM], baseldA[MAXDIM];
    Integer idx, n1dim;
    Integer atotal;
    Integer me= ga_nodeid_();
 
    ga_sync_();
    ga_check_handle(g_a, "has_negative_elem");
    GA_PUSH_NAME("has_negative_elem");
 
    nga_inquire_internal_(g_a, &atype, &andim, adims);

 /* check if patch indices and dims match */
    for(i=0; i<andim; i++)
        if(alo[i] <= 0 || ahi[i] > adims[i])
            ga_error("g_a indices out of range ", *g_a);
   
    /* find out coordinates of patches of g_a, g_b and g_c that I own */
    nga_distribution_(g_a, &me, loA, hiA);

   /*  determine subsets of my patches to access  */
    if (ngai_patch_intersect(alo, ahi, loA, hiA, andim)){
        nga_access_ptr(g_a, loA, hiA, &A_ptr, ldA);
   
        /* number of n-element of the first dimension */
        n1dim = 1; for(i=1; i<andim; i++) n1dim *= (hiA[i] - loA[i] + 1);
 
        /* calculate the destination indices */
        bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
        /* baseld[0] = ld[0]
         * baseld[1] = ld[0] * ld[1]
         * baseld[2] = ld[0] * ld[1] * ld[2] .....
         */
        baseldA[0] = ldA[0]; baseldA[1] = baseldA[0] *ldA[1];
        for(i=2; i<andim; i++) {
            bvalue[i] = 0;
            bunit[i] = bunit[i-1] * (hiA[i-1] - loA[i-1] + 1);
            baseldA[i] = baseldA[i-1] * ldA[i];
        }
 
         for(i=0; i<n1dim; i++) {
                    idx = 0;
                    for(j=1; j<andim; j++) {
                        idx += bvalue[j] * baseldA[j-1];
                        if(((i+1) % bunit[j]) == 0) bvalue[j]++;
                        if(bvalue[j] > (hiA[j]-loA[j])) bvalue[j] = 0;
                    }
 
                    switch(atype){
                        case C_DBL:
                        /*double is the only type that is handled for Tao/GA project*/
                        tempA=((double*)A_ptr)+idx;
		        for(i=0;i<hiA[0]-loA[0]+1;i++)
				if(tempA[i]<0) return 1;
                        break;
                        case C_DCPL:
                        case C_INT:
                        case C_FLOAT:
                        case C_LONG:
 
                       default: ga_error(" has_negative_elem: wrong data type ",atype);
                   }

        }
 
        /* release access to the data */
        nga_release_ (g_a, loA, hiA);
    }
 
    GA_POP_NAME;
    ga_sync_();
    return 0; /*negative element is not found in g_a*/
}

void ga_step_max2_patch_(g_xx,xxlo,xxhi, g_vv,vvlo,vvhi, g_xxll,xxlllo,xxllhi, g_xxuu,xxuulo,xxuuhi, result) 
     Integer *g_xx, *xxlo, *xxhi;    /* patch of g_xx */
     Integer *g_vv, *vvlo, *vvhi;    /* patch of g_vv */
     Integer *g_xxll, *xxlllo, *xxllhi;    /* patch of g_xxll */
     Integer *g_xxuu, *xxuulo, *xxuuhi;    /* patch of g_xxuu */
     double *result;
{
     Integer index;
     Integer g_C, *g_c=&g_C;
     double alpha = 1.0, beta = -1.0;
     double  result1, result2;

     
     	/*duplicatecate an array c to hold the temparary result */
     	ga_duplicate(g_xx, &g_C, "TempC");
     	if(g_C==0)
		ga_error("ga_step_max2_patch_:fail to duplicate array c", *g_c);

        /*First, compute xu - xx */
       nga_add_patch_(&alpha, g_xxuu, xxuulo, xxuuhi, &beta, g_xx, xxlo, xxhi, g_c, xxlo, xxhi); 
       /* Then, compute (xu-xx)/dx */
       ga_elem_divide_patch_(g_c, xxlo, xxhi, g_vv, vvlo, vvhi, g_c, xxlo, xxhi); 
        /*Now look at each element of the array g_c. 
	  If an element of g_c is positive infinity, then replace it with -GA_INFINITY */ 
        ngai_elem3_patch_(g_c, xxlo, xxhi, OP_STEPMAX2);  
        /*Then, we will select the maximum of the array g_c*/ 
        nga_select_elem_(g_c, "max", &result1, &index); 

        /*Now doing the same thing to get (xxll-xx)/dx */
        /*First, compute xl - xx */
       nga_add_patch_(&alpha, g_xxll, xxlllo, xxllhi, &beta, g_xx, xxlo, xxhi, g_c, xxlo, xxhi); 
       /* Then, compute (xl-xx)/dx */
       ga_elem_divide_patch_(g_c, xxlo, xxhi, g_vv, vvlo, vvhi, g_c, xxlo, xxhi); 
        /*Now look at each element of the array g_c. 
	  If an element of g_c is positive infinity, then replace it with -GA_INFINITY */ 
        ngai_elem3_patch_(g_c, xxlo, xxhi, OP_STEPMAX2);  
        /*Then, we will select the maximum of the array g_c*/ 
        nga_select_elem_(g_c, "max", &result2, &index); 
        *result = MAX(result1, result2);
     //if(*result==0.0) *result = -GA_INFINITY;
     *result = ABS(*result);
}

/*\ generic  routine for element wise operation between two array
\*/
#if 0 //I want to delete op parameter
void ga_step_max_patch_(g_a,  alo, ahi, g_b,  blo, bhi, result, op) 
#else
#endif

void ga_step_max_patch_(g_a,  alo, ahi, g_b,  blo, bhi, result) 
     Integer *g_a, *alo, *ahi;    /* patch of g_a */
     Integer *g_b, *blo, *bhi;    /* patch of g_b */
     double *result;
#if 0
     Integer op; /* operations */
#endif

{
     Integer index;
     //double result = -1;
     Integer *g_c;
     Integer g_C;

     if(*g_a == *g_b)
	*result = 1.0;
     else
     {
     	/*Now look at each element of the array g_a. If an element of g_a is negative, then simply return */ 
     	if(has_negative_elem(g_a, alo, ahi))
		ga_error("ga_step_max_patch_: g_a has negative element.", -1);
     
     	/*duplicatecate an array c to hold the temparate result = g_a/g_b; */
     	ga_duplicate(g_a, &g_C, "Temp");
     	if(g_C==0)
		ga_error("ga_step_max_patch_:fail to duplicate array c", *g_c);
        g_c = &g_C; 
     	ga_elem_divide_patch_(g_a, alo, ahi, g_b, blo, bhi, g_c, alo, ahi);

        /*Now look at each element of the array g_c. If an element of g_c is positive, then replace it with -GA_INFINITY */ 
        ngai_elem3_patch_(g_c, alo, ahi, OP_STEPMAX);  
        /*Then, we will select the maximum of the array g_c*/ 
        nga_select_elem_(g_c, "max", result, &index); 
     }
     if(*result==0.0) *result = -GA_INFINITY;
     *result = ABS(*result);
}


void ga_step_max_(Integer *g_a, Integer *g_b, double *retval)
{
   Integer atype, andim;
   Integer btype, bndim;
   Integer alo[MAXDIM],ahi[MAXDIM];
   Integer blo[MAXDIM],bhi[MAXDIM];
 
    nga_inquire_internal_(g_a,  &atype, &andim, ahi);
    nga_inquire_internal_(g_b,  &btype, &bndim, bhi);
    while(andim){
        alo[andim-1]=1;
        andim--;
        blo[bndim-1]=1;
        bndim--;
    }
    
#if 0
    ga_step_max_patch_(g_a, alo, ahi, g_b, blo, bhi, retval, OP_STEPMAX);
#else
    ga_step_max_patch_(g_a, alo, ahi, g_b, blo, bhi, retval);
#endif
}

void ga_step_max2_(Integer *g_xx, Integer *g_vv, Integer *g_xxll, Integer *g_xxuu,  double *retval)
{
   Integer xxtype, xxndim;
   Integer vvtype, vvndim;
   Integer xxlltype, xxllndim;
   Integer xxuutype, xxuundim;
   Integer xxlo[MAXDIM],xxhi[MAXDIM];
   Integer vvlo[MAXDIM],vvhi[MAXDIM];
   Integer xxlllo[MAXDIM],xxllhi[MAXDIM];
   Integer xxuulo[MAXDIM],xxuuhi[MAXDIM];
    
    nga_inquire_internal_(g_xx,  &xxtype, &xxndim, xxhi);
    nga_inquire_internal_(g_vv,  &vvtype, &vvndim, vvhi);
    nga_inquire_internal_(g_xxll,  &xxlltype, &xxllndim, xxllhi);
    nga_inquire_internal_(g_xxuu,  &xxuutype, &xxuundim, xxuuhi);
    while(xxndim){
        xxlo[xxndim-1]=1;
        xxndim--;
        vvlo[vvndim-1]=1;
        vvndim--;
        xxlllo[xxllndim-1]=1;
        xxllndim--;
        xxuulo[xxuundim-1]=1;
        xxuundim--;
    }
 
   ga_step_max2_patch_(g_xx,xxlo,xxhi, g_vv,vvlo,vvhi, g_xxll,xxlllo,xxllhi, g_xxuu,xxuulo,xxuuhi, retval);
}

