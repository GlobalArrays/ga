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
int bvec,dvec,svec,dmvec,m_dvec,amat,xvec,axvec,rvec,qvec,ridx,cidx;
int me, nproc;
int myfirstrow=0,mylastrow=0;
double epsilon=1e-4;
int isvectormirrored=0;
static int niter;
void read_and_create(int,char **);
void computeminverser(double *,double *, double *);
void computeminverse(double *,double *, int *,int *);
void finalize_arrays();

void conjugate_gradient(int nit,int dopreconditioning)
{
int i,one=1,zero=0,negone=-1;
int lo,hi;
double d_one=1.0,d_zero=0.0,d_negone=-1.0;
double delta0=0.0,deltaold=0.0,deltanew=0.0,alpha=0.0,negalpha,beta,dtransposeq;
double *axvecptr,*qvecptr,*aptr,*dmvecptr,*rvecptr,*dvecptr,*svecptr;
double time0;
int *mycp,*myrp;

    NGA_Distribution(cidx,me,&lo,&hi);
    NGA_Access(cidx,&lo,&hi,&mycp,&zero);
    NGA_Access(amat,&lo,&hi,&aptr,&zero);

    NGA_Distribution(ridx,me,&lo,&hi);
    NGA_Access(ridx,&lo,&hi,&myrp,&zero);
    NGA_Access(axvec,&lo,&hi,&axvecptr,&zero);
    NGA_Access(qvec,&lo,&hi,&qvecptr,&zero);

    if(dopreconditioning){
       NGA_Access(dmvec,&lo,&hi,&dmvecptr,&zero);
       NGA_Access(rvec,&lo,&hi,&rvecptr,&zero);
       NGA_Access(dvec,&lo,&hi,&dvecptr,&zero);
       NGA_Access(svec,&lo,&hi,&svecptr,&zero);
    }
   
    matvecmul(aptr,xvec,axvecptr,0,myrp,mycp);    /* compute Ax */

    GA_Add(&d_one,bvec,&d_negone,axvec,rvec);     /* r=b-Ax*/

    if(dopreconditioning){
      computeminverse(dmvecptr,aptr,myrp,mycp);
      computeminverser(dmvecptr,rvecptr,dvecptr);
    }
    else
      GA_Copy(rvec,dvec);

    deltanew = GA_Ddot(dvec,rvec);                /* deltanew = r.r_tranpose */

    delta0 = deltanew;                            /* delta0 = deltanew */

    if(me==0)printf("\n\tdelta0 is %f",delta0);
    if(me==0)printf("\n\titer\tbeta\tdelta");

    for(i=0;i<nit && deltanew>(epsilon*epsilon*delta0);i++){

       if(isvectormirrored)
         matvecmul(aptr,m_dvec,qvecptr,1,myrp,mycp);/* q = Ad */
       else
         matvecmul(aptr,dvec,qvecptr,0,myrp,mycp);

       dtransposeq=GA_Ddot(dvec,qvec);            /* compute d_transpose.q */

       alpha = deltanew/dtransposeq;              /* deltanew/(d_transpose.q) */

       GA_Add(&d_one,xvec,&alpha,dvec,xvec);      /* x = x+ alpha.d*/

       if(i>0 && i%50==0){
         matvecmul(aptr,xvec,axvecptr,0,myrp,mycp);/* compute Ax*/
	 GA_Add(&d_one,bvec,&d_negone,axvec,rvec);/* r=b-Ax*/   
       }
       else{
         negalpha = 0.0-alpha;                         
         GA_Add(&d_one,rvec,&negalpha,qvec,rvec);  /* r = r-alpha.q */ 
       }

       if(dopreconditioning)
         computeminverser(dmvecptr,rvecptr,svecptr);

       deltaold = deltanew;                        /* deltaold = deltanew*/

       if(dopreconditioning)
         deltanew = GA_Ddot(svec,rvec);            /* deltanew = r_transpose.r*/
       else
         deltanew = GA_Ddot(rvec,rvec);            /* deltanew = r_transpose.r*/

       beta = deltanew/deltaold;                   /* beta = deltanew/deltaold*/

       if(dopreconditioning)
         GA_Add(&d_one,svec,&beta,dvec,dvec);      /* d = r + beta.d */
       else
         GA_Add(&d_one,rvec,&beta,dvec,dvec);      /* d = r + beta.d */

       if(isvectormirrored)
         GA_Copy(dvec,m_dvec);                     /*copy from distributed */

       if(me==0)printf("\n\t%d\t%0.4f\t%f",(i+1),beta,deltanew);
    }
    if(i < nit && me == 0)printf("\n Done with CG before reaching max iter");
    niter = nit;
    /*
    GA_Zero(qvec);
    matvecmul(aptr,xvec,qvecptr,0,myrp,mycp);
    GA_Add(&d_one,qvec,&d_negone,bvec,rvec);
    time0=GA_Ddot(rvec,rvec);
    if(me==0)printf("\n%d:error is %f",me,time0);
    */
}

void initialize_arrays(int dpc)
{
double d_one=1.0;
    GA_Zero(dvec);
    GA_Fill(xvec,&d_one);
    GA_Zero(axvec);
    GA_Zero(rvec);
    GA_Zero(qvec);
    if(dpc){
       GA_Zero(dmvec);
       GA_Zero(svec);
    }
}

int main(argc, argv)
int argc;
char **argv;
{
int heap=200000, stack=200000;
int dopreconditioning=1;
double time0,time1;
double d_one=1.0,d_zero=0.0,d_negone=-1.0;

    MPI_Init(&argc, &argv);                    /* initialize MPI */
    GA_Initialize();                           /* initialize GA */

    me=GA_Nodeid(); 
    nproc=GA_Nnodes();
    if(me==0)printf("\n                          CONJUGATE GRADIENT EXAMPLE\n");
    if(argc<3){
       if(me==0){
         printf(" CORRECT USAGE IS:");
         printf("\n\n <launch commands> ga_cg.x na nz file");
         printf("\n\n where:");
         printf("\n\tna is array dimention (only square arrays supported)");
         printf("\n\tnz is number of non-zeros");
         printf("\n\tfile is either the input file or the word random");
         printf("\n\t  use the word random if you to use random input");
         printf("\n\t  input should be in row compressed format");
         printf("\n\t  file should have matrix a followed by row, col & b (Ax=b)");
         printf("\n\t  if file also has na and nz, pass them as 0's and the");
         printf("\n\t  program will read them from the file");
         printf("\n\nexample usages are:");
         printf("\n\tmpirun -np 4 ./ga_cg.x 5000 80000 /home/me/myinput.dat");
         printf("\n\tor");
         printf("\n\tmpirun -np 4 ./ga_cg.x 5000 80000 random\n\n");
         fflush(stdout);
       }
       GA_Terminate();
       MPI_Finalize();
       return 0;
    }

    heap /= nproc;
    stack /= nproc;
    if(! MA_init(MT_F_DBL, stack, heap)) 
       GA_Error("MA_init failed",stack+heap);  /* initialize memory allocator*/ 
    
    read_and_create(argc,argv);

    if(me==0)printf("\nWarmup and initialization run");
    initialize_arrays(dopreconditioning);
    conjugate_gradient(1,dopreconditioning);

    if(me==0)printf("\n\nStarting Conjugate Gradient ....");
    initialize_arrays(dopreconditioning);
    time0=MPI_Wtime();
    conjugate_gradient(na,dopreconditioning);
    time1=MPI_Wtime();

    if(me==0)printf("\n%d:time per iteration=%f\n",me,(time1-time0));

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
     if(isvectormirrored)
        GA_Destroy(m_dvec);
     GA_Destroy(amat);
     GA_Destroy(xvec);
     GA_Destroy(axvec);
     GA_Destroy(rvec);
     GA_Destroy(qvec);
     GA_Destroy(ridx);
     GA_Destroy(cidx);
}     
