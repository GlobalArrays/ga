/****************************************************************************
* program: ga-mpi.c
* date:    Tue Oct  3 12:31:59 PDT 1995
* author:  Jarek Nieplocha
* purpose: This program demonstrates interface between GA and MPI. 
*          For a given square matrix, it creates a vector that contains maximum
*          elements for each matrix column. MPI group communication is used.
*
* notes:   The program can run in two modes:
*          1. Using TCGMSG calls available through the TCGMSG-MPI library
*             and MPI. In this mode initialization must be done with 
*             the TCGMSG PBEGIN call.
*          2. Using MPI calls only -- preprocessor symbol MPI must be defined.
*
****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "macommon.h"
#include "mpi.h"

#ifdef MPI
#   define ERROR_(str,code){\
           fprintf(stderr,"%s",(str)); MPI_Abort(MPI_COMM_WORLD,(int)code);\
    }
#else
/*  include files for the TCGMSG-MPI library */
#   include "sndrcv.h"
#   include "msgtypesc.h"
#endif

#define N 100           /* dimension of matrices */


void do_work()
{
Integer ONE=1, ZERO=0;   /* useful constants */
Integer g_a, g_b;
Integer n=N, type=MT_F_DBL;
Integer me=GA_nodeid(), nproc=GA_nnodes();
Integer col, i, j;

/* Note: on all current platforms DoublePrecision = double */
DoublePrecision buf[N], *max_col;

MPI_Comm GA_COMM, COL_COMM;
Integer *region_list, reg_proc;
Integer ilo,ihi, jlo,jhi, maidx, ld, prow, pcol;
int     *proc_list, root=0, grp_me=-1;

     if(me==0)printf("Creating matrix A\n");
     /* create matrix A */
     if(! GA_create(&type, &n, &n, "A", &ZERO, &ZERO, &g_a))
          GA_error("create failed: A",n); 
     if(me==0)printf("OK\n");

     if(me==0)printf("Creating matrix B\n");
     if(! GA_create(&type, &n, &ONE, "B", &ZERO, &ZERO, &g_b))
          GA_error("create failed: B",n); 
     if(me==0)printf("OK\n");

     GA_zero(&g_a);   /* zero the matrix */

     if(me==0)printf("Initializing matrix A\n");
     /* fill in matrix A with values: A(i,j) = (i+j) */ 
     for(col=1+me; col<=n; col+= nproc){
         /*
          * siple load balancing: 
          * each process works on a different column in MIMD style 
          */ 
          for(i=0; i<n; i++) buf[i]=(DoublePrecision)(i+col+1); 
          GA_put(&g_a, &ONE, &n, &col, &col, buf, &n); 
     }

     /* GA_print(&g_a);*/


     GA_distribution(&g_a, &me, &ilo, &ihi, &jlo, &jhi);

     region_list = (Integer*)malloc(5*nproc*sizeof(Integer));
     if (!region_list) GA_error("malloc 1 failed",nproc);

     proc_list = (int*)malloc(nproc*sizeof(int));
     if (!proc_list) GA_error("malloc 2 failed",nproc);

     GA_sync(); 
     if(jhi-jlo+1 >0){
        max_col = (DoublePrecision*)malloc(sizeof(DoublePrecision)*(jhi-jlo+1));
        if (!max_col) GA_error("malloc 3 failed",(jhi-jlo+1));
     

        /* get the list of processors that own this block column A[:,jlo:jhi] */
        GA_locate_region(&g_a, &ONE, &n, &jlo, &jhi, (Integer (*)[5])region_list, &reg_proc);
        for(i=0; i< reg_proc; i++) proc_list[i] = (int) region_list[5*i];
     }

     GA_mpi_communicator(&GA_COMM); /* get MPI communicator for GA processes */
     GA_proc_topology(&g_a, &me, &prow, &pcol);  /* block coordinates */


     if(me==0)printf("Splitting communicator according to distribution of A\n");

     /* GA on SP1 requires synchronization before and after message-passing !!*/
     GA_sync(); 
        
        if(me==0)printf("Computing max column elements\n");
        /* create communicator for processes that 'own' A[:,jlo:jhi] */
          MPI_Barrier(GA_COMM);
        if(pcol < 0 || prow <0)
           MPI_Comm_split(GA_COMM, MPI_UNDEFINED, MPI_UNDEFINED, &COL_COMM);
        else
           MPI_Comm_split(GA_COMM, (int)pcol, (int)prow, &COL_COMM);
        
        if(COL_COMM != MPI_COMM_NULL){
           MPI_Comm_rank(COL_COMM, &grp_me);

           /* each process computes max elements in the block it 'owns' */
           GA_access(&g_a, &ilo, &ihi, &jlo, &jhi, &maidx, &ld);
           maidx --; /* Fortran-C indexing conversion */
           for(j=0; j<jhi-jlo+1; j++){
               max_col[j] = 0.;  /* all matrix elements are positive */
               for(i=0; i<ihi-ilo+1; i++)
                   if(max_col[j] < DBL_MB[maidx +j*ld + i]){
                      max_col[j] = DBL_MB[maidx +j*ld + i];
                   }
                 
           }

           MPI_Reduce(max_col, buf, jhi-jlo+1, MPI_DOUBLE, MPI_MAX,
                      root, COL_COMM);
        
       
       }else fprintf(stderr,"process %d not participating\n",me);
     GA_sync(); 

     /* processes with rank=root in COL_COMM put results into g_b */
     ld = jhi-jlo+1;
     if(grp_me == root)
          GA_put(&g_b, &jlo, &jhi,  &ONE, &ONE, buf, &ld); 
        
     GA_sync(); 

     if(me==0)printf("Checking the result\n");
     if(me==0){
        GA_get(&g_b, &ONE, &n, &ONE, &ONE, buf, &n); 
        for(i=0; i< n; i++)if(buf[i] != (double)n+i+1){
            fprintf(stderr,"error:%d max=%lf should be:%ld\n",i,buf[i],n+i+1);
            GA_error("terminating...",0);
        }
     }
     
     if(me==0)printf("OK\n");

     GA_destroy(&g_a);
     GA_destroy(&g_b);
}
     


int main(argc, argv)
int argc;
char **argv;
{
Integer heap=20000, stack=20000;
Integer me, nproc;

#   ifdef MPI

      int myid, numprocs;
      MPI_Init(&argc, &argv);
      MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
      MPI_Comm_rank(MPI_COMM_WORLD,&myid);
      me = (Integer)myid;
      nproc = (Integer)numprocs;

#   else

      PBEGIN_(argc, argv);                        /* initialize TCGMSG */
      me=NODEID_(); 
      nproc=NNODES_();

#   endif

    if(me==0) printf("Using %ld processes\n",(long)nproc);

    heap /= nproc;
    stack /= nproc;
    if(! MA_init((Integer)MT_F_DBL, stack, heap)) 
       ERROR_("MA_init failed",stack+heap);     /* initialize memory allocator*/ 
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_ARE_FATAL);
    GA_initialize();                           /* initialize GA */
    
    do_work();

    if(me==0)printf("Terminating ..\n");
    GA_terminate();

#   ifdef MPI
      MPI_Finalize();
#   else
      PEND_();
#   endif

    return 0;
}

