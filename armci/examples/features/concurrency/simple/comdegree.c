/*$Id: comdegree.c,v 1.1.2.1 2007-06-20 17:41:49 vinod Exp $*/
/*
 *                                Copyright (c) 2006
 *                      Pacific Northwest National Laboratory,
 *                           Battelle Memorial Institute.
 *                              All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met: - Redistributions of source code must retain the above
 * copyright notice, this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * - Neither the name of the Battelle nor the names of its contributors
 *   may be used to endorse or promote products derived from this software
 *   without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
 * TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */


/*************************
 * This test checks the networks ability to overlap data transfers.
 * It does it both for an optimitic case (with no other communication) and
 * a more realistic case.
 * --Vinod Tipparaju
 * --Pacific Northwest National Laboratory
 * --vinod@pnl.gov
*************************/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#define DEBUG__ 

#if defined(PVM)
#   include <pvm3.h>
#   ifdef CRAY
#     define MPGROUP         (char *)NULL
#     define MP_INIT(arc,argv)
#   else
#     define MPGROUP           "mp_working_group"
#     define MP_INIT(arc,argv) pvm_init(arc, argv)
#   endif
#   define MP_FINALIZE()     pvm_exit()
#   define MP_BARRIER()      pvm_barrier(MPGROUP,-1)
#   define MP_MYID(pid)      *(pid)   = pvm_getinst(MPGROUP,pvm_mytid())
#   define MP_PROCS(pproc)   *(pproc) = (int)pvm_gsize(MPGROUP)
    void pvm_init(int argc, char *argv[]);
#elif defined(TCGMSG)
#   include <sndrcv.h>
    long tcg_tag =30000;
#   define MP_BARRIER()      SYNCH_(&tcg_tag)
#   define MP_INIT(arc,argv) PBEGIN_((argc),(argv))
#   define MP_FINALIZE()     PEND_()
#   define MP_MYID(pid)      *(pid)   = (int)NODEID_()
#   define MP_PROCS(pproc)   *(pproc) = (int)NNODES_()
#else
#   include <mpi.h>
#   define MP_BARRIER()      MPI_Barrier(MPI_COMM_WORLD)
#   define MP_FINALIZE()     MPI_Finalize()
#   define MP_INIT(arc,argv) MPI_Init(&(argc),&(argv))
#   define MP_MYID(pid)      MPI_Comm_rank(MPI_COMM_WORLD, (pid))
#   define MP_PROCS(pproc)   MPI_Comm_size(MPI_COMM_WORLD, (pproc));
#   define MP_TIMER()        MPI_Wtime()
#endif

#include "armci.h"


/***************************** macros ************************/
#define COPY(src, dst, bytes) memcpy((dst),(src),(bytes))
#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a) (((a) <0) ? -(a) : (a))

/***************************** global data *******************/
int me, nproc;


void create_array(void *a[], int size)
{
     armci_size_t bytes=size;
     int i, rc;

     rc = ARMCI_Malloc(a, bytes);
     assert(rc==0);
     
#ifdef DEBUG_
     printf("%d after malloc ndim=%d b=%d ptr=%p\n",me,ndim,(int) bytes,a[me]);
     fflush(stdout);
#endif

     assert(a[me]);
     bzero(a[me],bytes);
}

void destroy_array(void *ptr[])
{
    MP_BARRIER();

    assert(!ARMCI_Free(ptr[me]));
}



#define LCC 13
#define MAXPROC 128
    
void test_get_multidma()
{
int dim,elems;
int i,j, proc=1,Idx=1,idx=0;
void *b[MAXPROC], *a[MAXPROC];
int left = (me+nproc-1) % nproc;
int right = (me+1) % nproc;
int sendersright=0,sendersleft=0;
int loopcnt=10, itercount=5,less=2, strl; /*less>1 takes a partial plane */
double tt, t0[LCC],t1[LCC],t2[LCC],t3[LCC],t4=0,t5=0,t6=0;
armci_hdl_t hdl1,hdl2;

    for(i=0;i<LCC;i++){
       t0[i]=0;
       t1[i]=0;
       t2[i]=0;
       t3[i]=0;
    }

    /* create shared and local arrays */
    create_array(b, 1024*1024*10);
    create_array(a, 1024*1024*10);
    /* warmup */
    ARMCI_INIT_HANDLE(&hdl1);
    ARMCI_INIT_HANDLE(&hdl2);
    ARMCI_NbGet((double*)b[left],(double*)a[me],1024,left,&hdl1);
    ARMCI_NbGet((double*)b[right]+1024,(double*)a[me]+1024,1024,
                         right,&hdl2);
    ARMCI_Wait(&hdl1);
    ARMCI_Wait(&hdl2);

    ARMCI_Barrier();

    /*start test*/
    for(j=0;j<itercount;j++){
       for(i=0;i<loopcnt;i++){
         int lc, rc,wc,lc1,rc1,wc1,bytes;
        
         sendersright = (j+1)%nproc;
         sendersleft = (j+nproc-1)%nproc;

         bytes = 1024*pow(2,i);

         ARMCI_INIT_HANDLE(&hdl1);

         armci_msg_barrier();
         /*first time a regular call*/
         tt = MP_TIMER();
         ARMCI_NbGet((double*)b[left],(double*)a[me],bytes, left,&hdl1);
         ARMCI_Wait(&hdl1);
         t1[i] += (MP_TIMER()-tt);

         
         armci_msg_barrier();
         /*now time 1 left + 1 right but realize there is one xtra issue*/
         ARMCI_INIT_HANDLE(&hdl1);
         ARMCI_INIT_HANDLE(&hdl2);
         tt = MP_TIMER();
         ARMCI_NbGet((double*)b[left],(double*)a[me],bytes/2,left,&hdl1);
         ARMCI_NbGet((double*)b[right]+bytes/16,(double*)a[me]+bytes/16,bytes/2,
                         right,&hdl2);
         ARMCI_Wait(&hdl1);
         ARMCI_Wait(&hdl2);
         t2[i] += (MP_TIMER()-tt);

         ARMCI_Barrier();
         armci_msg_barrier();
         /*now time both to the left*/
         ARMCI_INIT_HANDLE(&hdl1);
         ARMCI_INIT_HANDLE(&hdl2);
         tt = MP_TIMER();
         ARMCI_NbGet((double*)b[left],(double*)a[me],bytes/2,left,&hdl1);
         ARMCI_NbGet((double*)b[left]+bytes/16,(double*)a[me]+bytes/16,bytes/2,
                         left,&hdl2);
         ARMCI_Wait(&hdl1);
         ARMCI_Wait(&hdl2);
         t3[i] += ( MP_TIMER()-tt);

         ARMCI_Barrier();
       }
    }
    MP_BARRIER();
    if(0==me){
       for(i=0;i<loopcnt;i++){
         fprintf(stderr,"\n%.0f\t%.2e\t%.2e\t%.2e",
                         1024.0*pow(2,i),t1[i]/loopcnt,t3[i]/loopcnt,
                         t2[i]/loopcnt);
       }
    }
    fflush(stdout);
    MP_BARRIER();
    MP_BARRIER();
    if((nproc-3)==me){
       for(i=0;i<loopcnt;i++){
         fprintf(stderr,"\n%.0f\t%.2e\t%.2e\t%.2e",
                         1024.0*pow(2,i),t1[i]/loopcnt,t3[i]/loopcnt,
                         t2[i]/loopcnt);
       }
    }
    fflush(stdout);
    MP_BARRIER();
#if 0
    for(j=0;j<nproc;j++) {
       if(j==me){
         for(i=0;i<loopcnt;i++){
           printf("\n%d:size=%f onesnd=%.2e twosnd=%.2e twosnddiffdir=%.2e\n",
                           me,1024.0*pow(2,i),t1[i]/loopcnt,t3[i]/loopcnt,
                           t2[i]/loopcnt);
         }
         MP_BARRIER();
       }
       else
         MP_BARRIER();
    }
#endif


    ARMCI_Barrier();

    destroy_array(b);
    destroy_array(a);
}


void test_put_multidma()
{
int dim,elems;
int i,j, proc=1,Idx=1,idx=0;
void *b[MAXPROC], *a[MAXPROC];
int left = (me+nproc-1) % nproc;
int right = (me+1) % nproc;
int sendersright=0,sendersleft=0;
int loopcnt=LCC, itercount=1000,less=2, strl; /*less>1 takes a partial plane */
double tt, t0[LCC],t1[LCC],t2[LCC],t3[LCC],t4=0,t5=0,t6=0;
armci_hdl_t hdl1,hdl2;


    /* create shared and local arrays */
    create_array(b, 1024*1024*10);
    create_array(a, 1024*1024*10);
    for(i=0;i<LCC;i++){
       t0[i]=0;
       t1[i]=0;
       t2[i]=0;
       t3[i]=0;
    }

    ARMCI_Barrier();
    for(j=0;j<itercount;j++){
       for(i=0;i<loopcnt;i++){
         int lc, rc,wc,lc1,rc1,wc1,bytes;
        
         sendersright = (j+1)%nproc;
         sendersleft = (j+nproc-1)%nproc;

         bytes = 1024*pow(2,i)/8;

         ARMCI_INIT_HANDLE(&hdl1);
         ARMCI_NbPut((double*)a[me]+1024,(double*)b[left]+1024,bytes,left,&hdl1);
         ARMCI_Wait(&hdl1);
         ARMCI_INIT_HANDLE(&hdl1);
         ARMCI_NbPut((double*)a[me]+1024,(double*)b[right]+1024,bytes,right,&hdl1);
         ARMCI_Wait(&hdl1);

         ARMCI_INIT_HANDLE(&hdl1);
         armci_msg_barrier();

         tt = MP_TIMER();
         ARMCI_NbPut((double*)a[me],(double*)b[left],bytes, left,&hdl1);
         ARMCI_Wait(&hdl1);
         t1[i] += (MP_TIMER()-tt);
         //lc=armci_notify(left);
         //tt = MP_TIMER();
         //rc = armci_notify_wait(right,&wc); 
         //t1[i] += (MP_TIMER()-tt);

         ARMCI_INIT_HANDLE(&hdl1);
         ARMCI_INIT_HANDLE(&hdl2);
         armci_msg_barrier();

         tt = MP_TIMER();
         ARMCI_NbPut((double*)a[me],(double*)b[left],bytes,left,&hdl1);
         ARMCI_NbPut((double*)a[me],(double*)b[right],bytes,
                         right,&hdl2);
         ARMCI_Wait(&hdl1);
         t2[i] += (MP_TIMER()-tt);
         //lc=armci_notify(left);
         //lc1=armci_notify(right);
         //tt = MP_TIMER();
         //rc1 = armci_notify_wait(left,&wc1); 
         //rc = armci_notify_wait(right,&wc); 
         //t2[i] += (MP_TIMER()-tt);
         //ARMCI_Wait(&hdl1);
         ARMCI_Wait(&hdl2);

         ARMCI_INIT_HANDLE(&hdl1);
         ARMCI_INIT_HANDLE(&hdl2);
         armci_msg_barrier();

         tt = MP_TIMER();
         ARMCI_NbPut((double*)a[me],(double*)b[left],bytes/2,left,&hdl1);
         ARMCI_NbPut((double*)a[me]+bytes/16,(double*)b[left]+bytes/16,bytes/2,
                         left,&hdl2);
         //ARMCI_Wait(&hdl1);
         //ARMCI_Wait(&hdl2);
         t3[i] += ( MP_TIMER()-tt);
         lc=armci_notify(left);
         tt = MP_TIMER();
         rc = armci_notify_wait(right,&wc); 
         t3[i] += ( MP_TIMER()-tt);
         ARMCI_Wait(&hdl1);
         ARMCI_Wait(&hdl2);

         ARMCI_Barrier();
       }
    }
    MP_BARRIER();
    if(0==me){
       for(i=0;i<loopcnt;i++){
         fprintf(stderr,"\n%.0f\t%.2e\t%.2e\t%.2e",
                         128.0*pow(2,i),t1[i]/itercount,t3[i]/itercount,
                         t2[i]/itercount);
       }
    }
    fflush(stdout);
    fflush(stdout);
    MP_BARRIER();
    MP_BARRIER();
    if((nproc-1)==me){
       for(i=0;i<loopcnt;i++){
         fprintf(stderr,"\n%.0f\t%.2e\t%.2e\t%.2e",
                         128.0*pow(2,i),t1[i]/itercount,t3[i]/itercount,
                         t2[i]/itercount);
       }
    }
    fflush(stdout);
    MP_BARRIER();
#if 0
    for(j=0;j<nproc;j++) {
       if(j==me){
         for(i=0;i<loopcnt;i++){
           printf("\n%d:size=%f onesnd=%.2e twosnd=%.2e twosnddiffdir=%.2e\n",
                           me,1024.0*pow(2,i),t1[i]/loopcnt,t3[i]/loopcnt,
                           t2[i]/loopcnt);
         }
         MP_BARRIER();
       }
       else
         MP_BARRIER();
    }
#endif


    ARMCI_Barrier();

    destroy_array(b);
    destroy_array(a);
}


int main(int argc, char* argv[])
{
    int ndim;

    MP_INIT(argc, argv);
    MP_PROCS(&nproc);
    MP_MYID(&me);

    
    ARMCI_Init();

    MP_BARRIER();
    if(me==0){
       printf("\nTesting transfer overlap with ARMCI put calls\n");
       printf("\nsize\tone-send\ttwo-sends\ttwo-sends-diff-dir\n");
       fflush(stdout);
       sleep(1);
    }
    MP_BARRIER();
    test_put_multidma();
    MP_BARRIER();
    if(me==0){
       printf("\nTesting transfer overlap with ARMCI get calls\n");
       printf("\nsize\tone-send\ttwo-sends\ttwo-sends-diff-dir\n");
       fflush(stdout);
       sleep(1);
    }
    MP_BARRIER();
    test_get_multidma();
    if(me==0)printf("\n");

    ARMCI_Finalize();
    MP_FINALIZE();
    return(0);
}
