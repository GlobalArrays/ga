#include <stdio.h>
#include <math.h>
#include "ga.h"
#include "macdecls.h"
#include <mpi.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

int na,nz;
int bvec,dvec,amat,xvec,axvec,rvec,qvec,ridx,cidx;
int me, nproc;
int myfirstrow=0,mylastrow=0;
double sigma=-1.0;
void read_and_create(int,char **);
void finalize_arrays();

static void matvecmul(double *aptr,int ga_vec, double *myresultptr,int isvectormirrored, int *myrowptr, int *mycolumnptr)
{
int i,j,k,l=0,lo,hi,one=1,zero=0;
double vecptr[na];
double tmprowsum=0.0;

    na--;
    NGA_Get(ga_vec,&zero,&na,vecptr,&na);
    na++;

    for(k=0,i=myfirstrow;i<mylastrow;i++,k++){
       for(j=myrowptr[k];j<myrowptr[k+1];j++,l++){
         tmprowsum+=aptr[l]*vecptr[mycolumnptr[l]];
       }
       myresultptr[k]=tmprowsum;
       tmprowsum=0.0;
    }
}

void conjugate_gradient(int nit,int dopreconditioning)
{
int i,one=1,zero=0,negone=-1;
int lo,hi;
double d_one=1.0,d_zero=0.0,d_negone=-1.0;
double delta0,deltaold,deltanew,alpha,negalpha,beta,dtransposeq;
double *axvecptr,*qvecptr,*aptr,*xvecptr;
int *mycp,*myrp;

    NGA_Distribution(cidx,me,&lo,&hi);
    NGA_Access(cidx,&lo,&hi,&mycp,&zero);
    NGA_Access(amat,&lo,&hi,&aptr,&zero);

    NGA_Distribution(ridx,me,&lo,&hi);
    NGA_Access(ridx,&lo,&hi,&myrp,&zero);
    NGA_Access(axvec,&lo,&hi,&axvecptr,&zero);
    NGA_Access(qvec,&lo,&hi,&qvecptr,&zero);
    NGA_Access(qvec,&lo,&hi,&xvecptr,&zero);
   
    matvecmul(aptr,xvec,axvecptr,0,myrp,mycp);    /* compute Ax */

    GA_Add(&d_one,bvec,&d_negone,axvec,rvec);     /* r=q-Ax*/

    deltanew = GA_Ddot(rvec,rvec);                /* deltanew = r.r_tranpose */

    dvec = GA_Duplicate(rvec,"D");                /* d = r */
    GA_Copy(rvec,dvec);

    delta0 = deltanew;                            /* delta0 = deltanew */

    if(me==0)printf("\n\titer\tbeta\tdelta");

    for(i=0;i<nit /*&& deltanew>(sigma*sigma*delta0)*/;i++){

       matvecmul(aptr,dvec,qvecptr,0,myrp,mycp);  /* q = Ad */

       dtransposeq=GA_Ddot(dvec,qvec);            /* compute d_transpose.q */

       alpha = deltanew/dtransposeq;              /* deltanew/(d_transpose.q) */

       GA_Add(&d_one,xvec,&alpha,dvec,xvec);      /* x = x+ alpha.d*/

       if(i%50==0){
         matvecmul(aptr,xvec,axvecptr,0,myrp,mycp);/* compute Ax*/
	 GA_Add(&d_one,bvec,&d_negone,axvec,rvec);/* r=q-Ax*/   
       }
       else{
         negalpha = -alpha;                         
         GA_Add(&d_one,rvec,&negalpha,qvec,rvec);  /* r = r-alpha.q */ 
       }

       deltaold = deltanew;                        /* deltaold = deltanew*/

       deltanew = GA_Ddot(rvec,rvec);              /* deltanew = r_transpose.r*/

       beta = deltanew/deltaold;                   /* beta = deltanew/deltaold*/

       GA_Add(&d_one,rvec,&beta,dvec,dvec);        /* d = r + beta.d */

       if(me==0)printf("\n\t%d\t%0.4f\t%f",(i+1),beta,deltanew);
    }
    if(i < nit && me == 0)printf("\n Done with CG before reaching max iter");
}


int main(argc, argv)
int argc;
char **argv;
{
int heap=200000, stack=200000;
int dopreconditioning=0;
double time0,time1;
double d_one=1.0,d_zero=0.0,d_negone=-1.0;

    MPI_Init(&argc, &argv);                       /* initialize MPI */

    GA_Initialize();                            /* initialize GA */
    me=GA_Nodeid(); 
    nproc=GA_Nnodes();

    heap /= nproc;
    stack /= nproc;
    
    if(! MA_init(MT_F_DBL, stack, heap)) 
       GA_Error("MA_init failed",stack+heap);  /* initialize memory allocator*/ 
    
    read_and_create(argc,argv);
    if(me==0)printf("\nWarmup and initialization run");
    conjugate_gradient(1,dopreconditioning);
    if(me==0)printf("\n\nStarting Conjugate Gradient ....");
    time0=MPI_Wtime();
    conjugate_gradient(50,dopreconditioning);
    time1=MPI_Wtime();
    if(me==0)printf("\n%d:time per iteration=%f\n",me,(time1-time0)/50);
    /*
    GA_Print(dvec);
    matvecmul(amat,dvec,qvec,0);
    GA_Add(&d_one,qvec,&d_negone,bvec,rvec);
    time0=GA_Ddot(rvec,rvec);
    if(me==0)printf("\n%d:error is %f",me,time0);
    */

    finalize_arrays();
    MPI_Barrier(MPI_COMM_WORLD);

    if(me==0)printf("Terminating ..\n");
    GA_Terminate();
    MPI_Finalize();
    return 0;
}


void finalize_arrays()
{
     GA_Destroy(bvec);
     GA_Destroy(dvec);
     GA_Destroy(amat);
     GA_Destroy(xvec);
     GA_Destroy(axvec);
     GA_Destroy(rvec);
     GA_Destroy(qvec);
     GA_Destroy(ridx);
     GA_Destroy(cidx);
}     
