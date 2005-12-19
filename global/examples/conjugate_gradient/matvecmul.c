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

static void matvecmul(double *aptr,int ga_vec, double *myresultptr,int isvectormirrored, int *myrowptr, int *mycolumnptr)
{
int i,j,k,l=0,lo,hi,one=1,zero=0;
double tmprowsum=0.0;
    if(isvectormirrored){
    }
    else {
      na--;
      NGA_Get(ga_vec,&zero,&na,ga_vecptr,&na);
      na++;
    }

    for(k=0,i=myfirstrow;i<mylastrow;i++,k++){
       for(j=myrowptr[k];j<myrowptr[k+1];j++,l++){
         tmprowsum+=aptr[l]*ga_vecptr[mycolumnptr[l]];
       }
       myresultptr[k]=tmprowsum;
       tmprowsum=0.0;
    }
}
