#include <stdio.h>
#ifdef LAPI
#  include "lapidefs.h"
#endif
#include <errno.h>
#include "armcip.h"
#include "copy.h"
#include "memlock.h"
#include "shmem.h"
#include "signaltrap.h"


/* global variables */
int armci_me, armci_nproc;
int armci_clus_me, armci_nclus, armci_master;
int armci_clus_first, armci_clus_last;
static int armci_initialized=0;
static int armci_terminating =0;

double armci_internal_buffer[BUFSIZE_DBL];

#if defined(SYSV) || defined(WIN32)
#   include "locks.h"
    lockset_t lockid;
#endif


void ARMCI_Cleanup()
{
#if defined(SYSV) || defined(WIN32)
    Delete_All_Regions();
#ifndef LAPI
    DeleteLocks(lockid);
#endif

    /* in case of an error notify server that it is time to quit */
#if defined(DATA_SERVER)
    if(armci_nclus >1){
        /* send quit request to server unless it is already dead */
        armci_wait_for_server();
        armci_CleanupSockets();
    }
#endif

#endif
}


void armci_die(char *msg, int code)
{
   
    if(armci_terminating)return;
    else armci_terminating=1;

    fprintf(stdout,"%d:%s: %d\n",armci_me, msg, code); fflush(stdout);
    fprintf(stderr,"%d:%s: %d\n",armci_me, msg, code);
    if(errno)perror("System Error Message:");
    ARMCI_Cleanup();
    armci_msg_abort(code);
}


void armci_die2(char *msg, int code1, int code2)
{
    if(armci_terminating)return;
    else armci_terminating=1;

    fprintf(stdout,"%d:%s: (%d,%d)\n",armci_me,msg,code1,code2); fflush(stdout);
    fprintf(stderr,"%d:%s: (%d,%d)\n",armci_me,msg,code1,code2);
    if(errno)perror("System Error Message:");
    ARMCI_Cleanup();
    armci_msg_abort(code1);
}


void ARMCI_Error(char *msg, int code)
{
    armci_die(msg,code);
}


#if defined(SYSV) || defined(WIN32)
void armci_allocate_locks()
{
#if defined(LAPI)

    int rc, bytes;
    _armci_lapi_mutexes = (int**) malloc(armci_nproc*sizeof(int*));
    if(!_armci_lapi_mutexes) armci_die("failed to allocate  ARMCI mutexes",0);

    if(armci_master == armci_me) {
      bytes = NUM_LOCKS*sizeof(int);
    }
    else bytes =0;

    rc = ARMCI_Malloc((void**)_armci_lapi_mutexes, bytes);
    if(rc) armci_die("failed to allocate ARMCI mutex array",rc);
    if(armci_master == armci_me)
     bzero(_armci_lapi_mutexes[armci_me],sizeof(int)*NUM_LOCKS);

#else
    if(armci_master==armci_me)CreateInitLocks(NUM_LOCKS, &lockid);
    armci_msg_clus_brdcst(&lockid, sizeof(lockid));
    if(armci_master != armci_me)InitLocks(NUM_LOCKS, lockid);
#endif
}

#endif


void ARMCI_Set_shm_limit(unsigned long shmemlimit)
{
#if defined(SYSV) || defined(WIN32)
#define EXTRASHM  1024   /* extra shmem used internally in ARMCI */
unsigned long limit;
    limit = armci_clus_info[armci_clus_me].nslave * shmemlimit + EXTRASHM;
    armci_set_shmem_limit(limit);
#endif
}


int ARMCI_Uses_shm()
{
    int uses=0;
    if(!armci_initialized)armci_die("ARMCI not yet initialized",0);

#if defined(SYSV) || defined(WIN32)
#   ifdef LAPI
      if(armci_nproc != armci_nclus)uses= 1;
#   else
      if(armci_nproc >1) uses= 1;
#   endif
#endif
/*    fprintf(stderr,"uses shmem %d\n",uses);*/
    return uses;
}



int ARMCI_Init()
{
    int rc;

    if(armci_initialized)return 0;
    else armci_initialized=1;

    armci_nproc = armci_msg_nproc();
    armci_me = armci_msg_me();


#ifdef CRAY
    cmpl_proc=-1;
#endif

#ifdef LAPI
    armci_init_lapi();
#endif

    armci_init_clusinfo();

    /* trap signals to cleanup ARMCI system resources in case of crash */
    if(armci_me==armci_master) ARMCI_ParentTrapSignals();
    ARMCI_ChildrenTrapSignals();

    armci_init_fence();

#if defined(SYSV) || defined(WIN32)

    /* init shared memory */
    if(ARMCI_Uses_shm())
      if(armci_master == armci_me) armci_shmem_init();

    /* allocate locks */
    if (armci_nproc > 1) armci_allocate_locks();

#endif

#if defined(DATA_SERVER)
    if(armci_nclus >1) armci_start_server();
#endif


    armci_msg_barrier();

    memlock_table_array = malloc(armci_nproc*sizeof(void*));
    if(!memlock_table_array) armci_die("malloc failed for ARMCI lock array",0); 

    rc = ARMCI_Malloc(memlock_table_array, MAX_SLOTS*sizeof(memlock_t));
    if(rc) armci_die("failed to allocate ARMCI memlock array",rc); 

    bzero(memlock_table_array[armci_me],MAX_SLOTS*sizeof(memlock_t));
    armci_msg_barrier();
/*    fprintf(stderr,"%d ready \n",armci_me);*/

    return 0;
}


void ARMCI_Finalize()
{
    if(!armci_initialized)return;
/*    armci_initialized=0;*/

    armci_msg_barrier();
    if(armci_me==armci_master) ARMCI_ParentRestoreSignals();

#if defined(DATA_SERVER)
    if(armci_nclus >1){
       armci_wait_for_server();
       armci_msg_barrier(); /* need to sync before closing sockets */
    }
#endif

    ARMCI_Cleanup();
    armci_msg_barrier();
}



int ARMCI_PutS( void *src_ptr,  /* pointer to 1st segment at source*/ 
		int src_stride_arr[], /* array of strides at source */
		void* dst_ptr,        /* pointer to 1st segment at destination*/
		int dst_stride_arr[], /* array of strides at destination */
		int count[],          /* number of segments at each stride levels: count[0]=bytes*/
		int stride_levels,    /* number of stride levels */
                int proc              /* remote process(or) ID */
                )
{
    int rc, direct=1;

    if(src_ptr == NULL || dst_ptr == NULL) return FAIL;
    if(count[0]<0)return FAIL3;
    if(stride_levels <0 || stride_levels > MAX_STRIDE_LEVEL) return FAIL4;
    if(proc<0)return FAIL5;

    ORDER(PUT,proc); /* ensure ordering */
    direct=SAMECLUSNODE(proc);


    /* use direct protocol for remote access when performance is better */
#   if defined(LAPI) && !defined(LAPI2)
      if(!direct)
         if(stride_levels==0 || count[0]> LONG_PUT_THRESHOLD )direct=1;
#   endif

#ifndef LAPI
    if(!direct)
       rc = armci_pack_strided(PUT, NULL, proc, src_ptr, src_stride_arr,
                       dst_ptr, dst_stride_arr, count, stride_levels, -1, -1);
    else
#endif
       rc = armci_op_strided( PUT, NULL, proc, src_ptr, src_stride_arr, 
                              dst_ptr, dst_stride_arr, count, stride_levels, 0);

    if(rc) return FAIL6;
    else return 0;

}


int ARMCI_GetS( void *src_ptr,  /* pointer to 1st segment at source*/ 
		int src_stride_arr[],   /* array of strides at source */
		void* dst_ptr,          /* pointer to 1st segment at destination*/
		int dst_stride_arr[],   /* array of strides at destination */
		int count[],            /* number of segments at each stride levels: count[0]=bytes*/
		int stride_levels,      /* number of stride levels */
                int proc                /* remote process(or) ID */
                )
{
    int rc,direct=1;

    if(src_ptr == NULL || dst_ptr == NULL) return FAIL;
    if(count[0]<0)return FAIL3;
    if(stride_levels <0 || stride_levels > MAX_STRIDE_LEVEL) return FAIL4;
    if(proc<0)return FAIL5;

    ORDER(GET,proc); /* ensure ordering */
    direct=SAMECLUSNODE(proc);

    /* use direct protocol for remote access when performance is better */
#   if defined(LAPI) && !defined(LAPI2)
      if(!direct)
        if( stride_levels==0 || count[0]> LONG_GET_THRESHOLD)direct=1;
        else{
          int i;
          int chunks=1;
          direct = 1;
          for(i=1; i<= stride_levels;i++){
              chunks *= count[i];
              if(chunks>MAX_CHUNKS_SHORT_GET){
                 direct = 0;
                 break;
                 }
              }
          }
#   endif

#ifndef LAPI2
    if(!direct)
       rc = armci_pack_strided(GET, NULL, proc, src_ptr, src_stride_arr,
                       dst_ptr, dst_stride_arr, count, stride_levels,-1,-1);
    else
#endif
       rc = armci_op_strided(GET, NULL, proc, src_ptr, src_stride_arr, 
                               dst_ptr, dst_stride_arr, count, stride_levels,0);

    if(rc) return FAIL6;
    else return 0;

}



int ARMCI_PutV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              )
{
    int rc, i,direct=1;


    if(len<1) return FAIL;
    for(i=0;i<len;i++){
        if(darr[i].src_ptr_array == NULL || darr[i].dst_ptr_array ==NULL) return FAIL2;
        if(darr[i].bytes<1)return FAIL3;
        if(darr[i].ptr_array_len <1) return FAIL4;
    }

    if(proc<0 || proc >= armci_nproc)return FAIL5;

    ORDER(PUT,proc); /* ensure ordering */
    direct=SAMECLUSNODE(proc);

    /* use direct protocol for remote access when performance is better */
#   ifdef LAPI
      if(!direct)
          if(len <5 || darr[0].ptr_array_len <5) direct=1;
#   endif


    if(direct)
         rc = armci_copy_vector(PUT, darr, len, proc);
    else
         rc = armci_pack_vector(PUT, NULL, darr, len, proc);

    if(rc) return FAIL6;
    else return 0;

}


int ARMCI_GetV( armci_giov_t darr[], /* descriptor array */
                int len,  /* length of descriptor array */
                int proc  /* remote process(or) ID */
              )
{
    int rc, i,direct=1;

    if(len<1) return FAIL;
    for(i=0;i<len;i++){
      if(darr[i].src_ptr_array==NULL ||darr[i].dst_ptr_array==NULL)return FAIL2;
      if(darr[i].bytes<1)return FAIL3;
      if(darr[i].ptr_array_len <1) return FAIL4;
    }

    if(proc<0 || proc >= armci_nproc)return FAIL5;

    ORDER(GET,proc); /* ensure ordering */
    direct=SAMECLUSNODE(proc);

    /* use direct protocol for remote access when performance is better */
#   ifdef LAPI
      if(!direct)
          if(len <5 || darr[0].ptr_array_len <8) direct=1;
#   endif


    if(direct)
       rc = armci_copy_vector(GET, darr, len, proc);
    else
       rc = armci_pack_vector(GET, NULL, darr, len, proc);

    if(rc) return FAIL6;
    else return 0;
}


int ARMCI_AccS( int  optype,            /* operation */
                void *scale,            /* scale factor x += scale*y */
                void *src_ptr,          /* pointer to 1st segment at source*/ 
		int src_stride_arr[],   /* array of strides at source */
		void* dst_ptr,          /* pointer to 1st segment at destination*/
		int dst_stride_arr[],   /* array of strides at destination */
		int count[],            /* number of segments at each stride levels: count[0]=bytes*/
		int stride_levels,      /* number of stride levels */
                int proc                /* remote process(or) ID */
                )
{
    int rc, direct=1;

    if(src_ptr == NULL || dst_ptr == NULL) return FAIL;
    if(src_stride_arr == NULL || dst_stride_arr ==NULL) return FAIL2;
    if(count[0]<0)return FAIL3;
    if(stride_levels <0 || stride_levels > MAX_STRIDE_LEVEL) return FAIL4;
    if(proc<0)return FAIL5;

    ORDER(optype,proc); /* ensure ordering */
    direct=SAMECLUSNODE(proc);

#   if defined(ACC_COPY)
       if(armci_me != proc) direct=0;
#   endif
 
    if(direct)
      rc = armci_op_strided( optype, scale, proc, src_ptr, src_stride_arr, 
                           dst_ptr, dst_stride_arr, count, stride_levels,1);
    else
      rc = armci_pack_strided(optype, scale, proc, src_ptr, src_stride_arr, 
                              dst_ptr,dst_stride_arr,count,stride_levels,-1,-1);

    if(rc) return FAIL6;
    else return 0;
}



int ARMCI_AccV( int op,              /* oeration code */
                void *scale,         /*scaling factor for accumulate */
                armci_giov_t darr[], /* descriptor array */
                int len,             /* length of descriptor array */
                int proc             /* remote process(or) ID */
              )
{
    int rc, i,direct=1;

    if(len<1) return FAIL;
    for(i=0;i<len;i++){
      if(darr[i].src_ptr_array==NULL ||darr[i].dst_ptr_array==NULL)return FAIL2;
      if(darr[i].bytes<1)return FAIL3;
      if(darr[i].ptr_array_len <1) return FAIL4;
    }

    if(proc<0 || proc >= armci_nproc)return FAIL5;

    ORDER(op,proc); /* ensure ordering */
    direct=SAMECLUSNODE(proc);

#   if defined(ACC_COPY)
       if(armci_me != proc) direct=0;
#   endif

    if(direct)
         rc = armci_acc_vector( op, scale, darr, len, proc);
    else
         rc = armci_pack_vector(op, scale, darr, len, proc);

    if(rc) return FAIL6;
    else return 0;
}


#if !(defined(SYSV) || defined(WIN32))
void ARMCI_Set_shmem_limit(unsigned long shmemlimit)
{
   /* not applicable here
    * aborting would  make user's life harder
    */
}
#endif
