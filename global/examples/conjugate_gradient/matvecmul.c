#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include "ga.h"
#include "macdecls.h"

extern int na,nz;
extern int me, nproc;
extern int myfirstrow,mylastrow;
double *ga_vecptr;


void computeminverse(double *minvptr,double *aptr,int *myrowptr,int *mycolptr)
{
int i,j,k,l=0,lo,hi,one=1,zero=0;
double tmprowsum=0.0;
double *vecptr=ga_vecptr;
    for(k=0,i=myfirstrow;i<mylastrow;i++,k++){
       for(j=myrowptr[k];j<myrowptr[k+1];j++,l++){
         if(mycolptr[j]>=i){
           if(mycolptr[j]==i){
             minvptr[l]=aptr[j];
             if(minvptr[l]<0)minvptr[l]=1;
           }
           if(mycolptr[j]>i)
             minvptr[l]=1;
           break;
         }
       }
    }
}

void computeminverser(double *minvptr,double *rvecptr,double *minvrptr)
{
int i,j,k,l=0,lo,hi,one=1,zero=0;
double tmprowsum=0.0;
double *vecptr=ga_vecptr;
    for(k=0,i=myfirstrow;i<mylastrow;i++,k++){
       minvrptr[k]=minvptr[k]*rvecptr[k];
    }
}

void matvecmul(double *aptr,int ga_vec, double *myresultptr,int isvectormirrored, int *myrowptr, int *mycolptr)
{
int i,j,k,l=0,lo,hi,one=1,zero=0;
double d_zero=0.0,d_one=1.0;
double tmprowsum=0.0;
double *vecptr=ga_vecptr;

    if(isvectormirrored){
      NGA_Access(ga_vec,&zero,&zero,&vecptr,&zero);
    }
    else { /*preliminary, later this will be done inside loop*/
      na--;
      NGA_Get(ga_vec,&zero,&na,vecptr,&na);
      na++;
      //NGA_Access(ga_vec,&zero,&zero,&vecptr,&zero);
    }

    for(k=0,i=myfirstrow;i<mylastrow;i++,k++){
       for(j=myrowptr[k];j<myrowptr[k+1];j++,l++){
         tmprowsum+=aptr[l]*vecptr[mycolptr[l]];
       }
       myresultptr[k]=tmprowsum;
       tmprowsum=0.0;
    }
}
