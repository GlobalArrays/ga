#include <stdio.h>
#include <math.h>
#include "global.h"
#include "macommon.h"
#ifdef MPI
#include <mpi.h>
#else
#include "sndrcv.h"
#endif

#define N 100            /* dimension of matrices */


void do_work()
{
Integer ONE=1, ZERO=0;   /* useful constants */
Integer g_a, g_b;
Integer n=N, type=MT_F_DBL;
Integer me=GA_nodeid(), nproc=GA_nnodes();
Integer col, i, row;

/* Note: on all current platforms DoublePrecision == double */
DoublePrecision buf[N], err, alpha, beta;

     if(me==0)printf("Creating matrix A\n");
     /* create matrix A */
     if(! GA_create(&type, &n, &n, "A", &ZERO, &ZERO, &g_a))
          GA_error("create failed: A",n); 
     if(me==0)printf("OK\n");

     if(me==0)printf("Creating matrix B\n");
     /* create matrix B  so that it has dims and distribution of A*/
     if(! GA_duplicate(&g_a, &g_b, "B")) GA_error("duplicate failed",n); 
     if(me==0)printf("OK\n");

     GA_zero(&g_a);   /* zero the matrix */

     if(me==0)printf("Initializing matrix A\n");
     /* fill in matrix A with random values in range 0.. 1 */ 
     for(col=1+me; col<=n; col+= nproc){
         /* each process works on a different column in MIMD style */
         for(i=0; i<n; i++) buf[i]=sin((double)i + 0.1*col);
         GA_put(&g_a, &ONE, &n, &col, &col, buf, &n);
     }


     if(me==0)printf("Symmetrizing matrix A\n");
     GA_symmetrize(&g_a);   /* symmetrize the matrix A = 0.5*(A+A') */
   

     /* check if A is symmetric */ 
     if(me==0)printf("Checking if matrix A is symmetric\n");
     GA_transpose(&g_a, &g_b); /* B=A' */
     alpha=1.; beta=-1.;
     GA_add(&alpha, &g_a, &beta, &g_b, &g_b);  /* B= A - B */
     err= GA_ddot(&g_b, &g_b);
     
     if(me==0)printf("Error=%lf\n",(double)err);
     
     if(me==0)printf("\nChecking atomic accumulate \n");

     GA_zero(&g_a);   /* zero the matrix */
     for(i=0; i<n; i++) buf[i]=(DoublePrecision)i;

     /* everybody accumulates to the same location/row */
     alpha = 1.0;
     row = n/2;
     GA_acc(&g_a, &row, &row, &ONE, &n, buf, &ONE, &alpha);
     GA_sync();

     if(me==0){ /* node 0 is checking the result */

        GA_get(&g_a, &row, &row, &ONE, &n, buf, &ONE);
        for(i=0; i<n; i++) if(buf[i] != (DoublePrecision)nproc*i)
           GA_error("failed: column=",i);
        printf("OK\n\n");

     }
     
     GA_destroy(&g_a);
     GA_destroy(&g_b);
}
     


int main(argc, argv)
int argc;
char **argv;
{
Integer heap=20000, stack=20000;
Integer me, nproc;

#ifdef MPI
    MPI_Init(&argc, &argv);                       /* initialize MPI */
#else
    PBEGIN_(argc, argv);                        /* initialize TCGMSG */
#endif

    GA_initialize();                            /* initialize GA */
    me=GA_nodeid(); 
    nproc=GA_nnodes();
    if(me==0) printf("Using %ld processes\n",(long)nproc);

    heap /= nproc;
    stack /= nproc;
    if(! MA_init((Integer)MT_F_DBL, stack, heap)) 
       GA_error("MA_init failed",stack+heap);  /* initialize memory allocator*/ 
    
    do_work();

    if(me==0)printf("Terminating ..\n");
    GA_terminate();

#ifdef MPI
    MPI_Finalize();
#else
    PEND_();
#endif

    return 0;
}

