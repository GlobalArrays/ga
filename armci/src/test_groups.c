/* $Id: test_groups.c,v 1.2 2004-06-28 17:45:19 manoj Exp $ */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef WIN32
#  include <windows.h>
#  define sleep(x) Sleep(1000*(x))
#else
#  include <unistd.h>
#endif

/* ARMCI is impartial to message-passing libs - we handle them with MP macros */
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
#   define MP_TIMER          armci_timer
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
#   define MP_TIMER         TCGTIME_
#else
#   include <mpi.h>
#   define MP_BARRIER()      MPI_Barrier(MPI_COMM_WORLD)
#   define MP_FINALIZE()     MPI_Finalize()
#   define MP_INIT(arc,argv) MPI_Init(&(argc),&(argv))
#   define MP_MYID(pid)      MPI_Comm_rank(MPI_COMM_WORLD, (pid))
#   define MP_PROCS(pproc)   MPI_Comm_size(MPI_COMM_WORLD, (pproc));
#   define MP_TIMER         MPI_Wtime
#endif

#include "armci.h"

#define MAXDIMS 7
#define MAXPROC 128
#define MINPROC 4

typedef int ARMCI_Datatype;

/***************************** macros ************************/
#define COPY(src, dst, bytes) memcpy((dst),(src),(bytes))
#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a) (((a) <0) ? -(a) : (a))

/***************************** global data *******************/
int me, nproc;
void* work[MAXPROC]; /* work array for propagating addresses */

#ifdef PVM
void pvm_init(int argc, char *argv[])
{
    int mytid, mygid, ctid[MAXPROC];
    int np, i;

    mytid = pvm_mytid();
    if((argc != 2) && (argc != 1)) goto usage;
    if(argc == 1) np = 1;
    if(argc == 2)
        if((np = atoi(argv[1])) < 1) goto usage;
    if(np > MAXPROC) goto usage;

    mygid = pvm_joingroup(MPGROUP);

    if(np > 1)
        if (mygid == 0) 
            i = pvm_spawn(argv[0], argv+1, 0, "", np-1, ctid);

    while(pvm_gsize(MPGROUP) < np) sleep(1);

    /* sync */
    pvm_barrier(MPGROUP, np);
    
    printf("PVM initialization done!\n");
    
    return;

usage:
    fprintf(stderr, "usage: %s <nproc>\n", argv[0]);
    pvm_exit();
    exit(-1);
}
#endif
          
void create_array(void *a[], int elem_size, int ndim, int dims[])
{
     int bytes=elem_size, i, rc;

     assert(ndim<=MAXDIMS);
     for(i=0;i<ndim;i++)bytes*=dims[i];

     rc = ARMCI_Malloc(a, bytes);
     assert(rc==0);
     
     assert(a[me]);
     
}

void destroy_array(void *ptr[])
{
    MP_BARRIER();

    assert(!ARMCI_Free(ptr[me]));
}

#define GNUM 3
#define ELEMS 10

void test_groups(int dryrun) {
  
    int pid_list[MAXPROC], pid_list1[MAXPROC];
    int value = -1, bytes;
    ARMCI_Group groupA, groupB;

    pid_list[0]=0;
    pid_list[1]=1;
    pid_list[2]=2;

    pid_list1[0]=0;
    pid_list1[1]=2;
    pid_list1[2]=3;
    
    MP_BARRIER();

#if 1
    /* create group 1 */
    ARMCI_Group_create(GNUM, pid_list, &groupA);
#endif

#if 1
    /* create group 2 */
    ARMCI_Group_create(GNUM, pid_list1, &groupB);
#endif

    /* ------------------------ GROUP A ------------------------- */ 
#if 1
    if(me<GNUM) { /* group A */
       int grp_me, grp_size;
       int i,j,src_proc=2,dst_proc=0;
       double *ddst_put[MAXPROC];
       double dsrc[ELEMS];
       int elems[2] = {MAXPROC,ELEMS};

       ARMCI_Group_rank(&groupA, &grp_me);
       ARMCI_Group_size(&groupA, &grp_size);
       if(grp_me==0) printf("GROUP SIZE = %d\n", grp_size);
       printf("%d:group rank = %d\n", me, grp_me);
       
       bytes = ELEMS*sizeof(double);       
       ARMCI_Malloc_group((void **)ddst_put, bytes, &groupA);
       // ARMCI_Malloc((void **)ddst_put, bytes);
       // create_array((void**)ddst_put, sizeof(double), 2, elems);
       
       for(i=0; i<ELEMS; i++) dsrc[i]=i*1.001*(me+1); 
       for(i=0; i<ELEMS; i++) ddst_put[me][i]=0.0;
       
       armci_msg_group_barrier(&groupA);
       ARMCI_Fence(dst_proc);

       if(me==src_proc) {
	  ARMCI_Put(dsrc, &ddst_put[dst_proc][0], bytes, dst_proc);
       }
       
       armci_msg_group_barrier(&groupA);
       ARMCI_Fence(dst_proc);
       
       /* Verify*/
       if(me==dst_proc) {
	  for(j=0; j<ELEMS; j++) {
	     if(ABS(ddst_put[me][j]-j*1.001*(src_proc+1)) > 0.1) {
		printf("The value is: %lf\n", ddst_put[me][j]);
		ARMCI_Error("groups: armci put failed...1", 0);
	     }
	  }
	  printf("\n%d: Test O.K. Verified\n", dst_proc);
       }
       armci_msg_group_barrier(&groupA);
       ARMCI_Free_group(ddst_put[grp_me], &groupA);
    }
#endif

    /* ------------------------ GROUP B ------------------------- */ 
#if 0
    if(me>0) { /* group B */
       int grp_me, grp_size;
       int i,j,dst_proc=2;
       double *ddst_put[MAXPROC];
       double dsrc[ELEMS];
       int elems[2] = {MAXPROC,ELEMS};

       ARMCI_Group_rank_(groupB, &grp_me);
       ARMCI_Group_size_(groupB, &grp_size);
       if(grp_me==0) printf("GROUP SIZE = %d\n", grp_size);
       printf("%d:group rank = %d\n", me, grp_me);
       MPI_Comm_rank(commB, &grp_me);
       printf("%d:group rank = %d\n", me, grp_me);
       
       bytes = ELEMS*sizeof(double);
       ARMCI_Malloc_group((void **)ddst_put, bytes, commB);
       // ARMCI_Malloc((void **)ddst_put, bytes);
       // create_array((void**)ddst_put, sizeof(double), 2, elems);
       
       for(i=0; i<ELEMS; i++) dsrc[i]=i*1.001*(me+1); 
       for(i=0; i<ELEMS; i++) ddst_put[me][i]=0.0;
       
       if(me==1) {
	  ARMCI_Put(dsrc, &ddst_put[dst_proc][0], bytes, dst_proc);
       }
       
       armci_comm_barrier(commB);
       ARMCI_Fence(dst_proc);
       
       /* Verify*/
       if(me==dst_proc) {
	  for(j=0; j<ELEMS; j++) {
	     if(ABS(ddst_put[me][j]-j*1.001) > 0.1) {
		printf("The value is: %lf\n", ddst_put[me][j]);
		ARMCI_Error("groups: armci put failed...1", 0);
	     }
	  }
	  printf("\n%d: Test O.K. Verified\n", dst_proc);
       }

       armci_comm_barrier(commB);
       ARMCI_Free_group(ddst_put[grp_me], commB);
    }
#endif

    ARMCI_AllFence();
    MP_BARRIER();
    
    if(!dryrun)if(me==0){printf("O.K.\n"); fflush(stdout);}
}


/* we need to rename main if linking with frt compiler */
#ifdef FUJITSU_FRT
#define main MAIN__
#endif

int main(int argc, char* argv[])
{

    MP_INIT(argc, argv);
    MP_PROCS(&nproc);
    MP_MYID(&me);

/*    printf("nproc = %d, me = %d\n", nproc, me);*/
    
    if( (nproc<MINPROC || nproc>MAXPROC) && me==0)
       ARMCI_Error("Test works for up to %d processors\n",MAXPROC);

    if(me==0){
       printf("ARMCI test program (%d processes)\n",nproc); 
       fflush(stdout);
       sleep(1);
    }
    
    ARMCI_Init();

    if(me==0){
      printf("\n Testing ARMCI Groups!\n\n");
      fflush(stdout);
    }

    test_groups(0);
    
    ARMCI_AllFence();
    MP_BARRIER();
    if(me==0){printf("\nSuccess!!\n"); fflush(stdout);}
    sleep(2);
	
    MP_BARRIER();
    ARMCI_Finalize();
    MP_FINALIZE();
    return(0);
}
