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
extern double time_get;


void computeminverse(double *minvptr,double *aptr,int *myrowptr,int *mycolptr)
{
int i,j,k,l=0,lo,hi,one=1,zero=0;
double tmprowsum=0.0;
double *vecptr=ga_vecptr;
    for(k=0,i=myfirstrow;i<mylastrow;i++,k++){
       for(j=myrowptr[k];j<myrowptr[k+1];j++,l++){
         if(mycolptr[l]>=i){
           if(mycolptr[l]==i){
             /*printf("\n%d:i=%d j=%d aptr=%d",me,i,j,aptr[l]);*/
             minvptr[k]=10.0/aptr[l];
             if(minvptr[k]<0)minvptr[k]=1.0;
           }
           if(mycolptr[l]>i)
             minvptr[k]=1.0;
           /*printf("\n%d:l=%d i=%d mycolptr[l]=%d",me,l,i,mycolptr[l]);*/
           l+=(myrowptr[k+1]-j);
           break;
         }
       }
    }
}

void computeminverser(double *minvptr,double *rvecptr,double *minvrptr)
{
int i,j,k,l=0,lo,hi,one=1,zero=0;
double *vecptr=ga_vecptr;
    for(k=0,i=myfirstrow;i<mylastrow;i++,k++){
       minvrptr[k]=minvptr[k]*rvecptr[k];
    }
}

void matvecmul(double *aptr,int ga_vec, double *myresultptr,int isvectormirrored, int *myrowptr, int *mycolptr)
{
int i,j,k,l=0,lo,hi,one=1,zero=0;
double t0,d_zero=0.0,d_one=1.0;
double tmprowsum=0.0;
double *vecptr=ga_vecptr;

    if(isvectormirrored){
      NGA_Access(ga_vec,&zero,&zero,&vecptr,&zero);
    }
    else { /*preliminary, later this will be done inside loop*/
#if 1
      t0=MPI_Wtime();
      na--;
      NGA_Get(ga_vec,&zero,&na,vecptr,&na);
      na++;
      time_get+=(MPI_Wtime()-t0);
#else
      NGA_Access(ga_vec,&lo,&hi,&vecptr,&zero);
#endif
    }

    for(k=0,i=myfirstrow;i<mylastrow;i++,k++){
       for(j=myrowptr[k];j<myrowptr[k+1];j++,l++){
         tmprowsum=tmprowsum+aptr[l]*vecptr[mycolptr[l]];
       }
       myresultptr[k]=tmprowsum;
       tmprowsum=0.0;
    }
}
