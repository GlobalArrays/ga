/* $Id: armci.c,v 1.58 2002-12-14 00:29:22 d3h325 Exp $ */

/* DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */

#define  EXTERN

#include <stdio.h>
#ifdef CRAY
#  include <sys/category.h>
#  include <sys/resource.h>
#  include <unistd.h>
#endif
#ifdef LAPI
#  include "lapidefs.h"
#endif
#include <errno.h>
#include "armcip.h"
#include "copy.h"
#include "memlock.h"
#include "shmem.h"
#include "signaltrap.h"

#ifdef GA_USE_VAMPIR
#include "armci_vampir.h"
#endif

/* global variables */
int armci_me, armci_nproc;
int armci_clus_me, armci_nclus, armci_master;
int armci_clus_first, armci_clus_last;
int _armci_initialized=0;
int _armci_terminating =0;
thread_id_t armci_usr_tid;
#ifndef HITACHI
double armci_internal_buffer[BUFSIZE_DBL];
#endif
#if defined(SYSV) || defined(WIN32) || defined(MMAP) || defined(HITACHI)
#   include "locks.h"
    lockset_t lockid;
#endif
#ifdef QUADRICS
#include <elan/elan.h>
#endif


void ARMCI_Cleanup()
{
#ifdef GA_USE_VAMPIR
  vampir_begin(ARMCI_CLEANUP,__FILE__,__LINE__);
#endif
#if (defined(SYSV) || defined(WIN32) || defined(MMAP))&& !defined(HITACHI) 
    Delete_All_Regions();
#if !defined(LAPI) 
    DeleteLocks(lockid);
#endif

    /* in case of an error notify server that it is time to quit */
#if defined(DATA_SERVER)
    if(armci_nclus >1){
        /* send quit request to server unless it is already dead */
        armci_wait_for_server();
        armci_transport_cleanup();
    }
#endif
#ifndef WIN32
    ARMCI_RestoreSignals();
#endif
#endif
#ifdef GA_USE_VAMPIR
  vampir_end(ARMCI_CLEANUP,__FILE__,__LINE__);
#endif
}


static void armci_perror_msg()
{
    char perr_str[80];
    if(!errno)return;
    sprintf(perr_str,"Last System Error Message from Task %d:",armci_me);
    perror(perr_str);
}


static void armci_abort(int code)
{
    armci_perror_msg();
    ARMCI_Cleanup();
#ifdef CRAY
    limit(C_PROC,0,L_CORE,1L); /* MPI_Abort on Cray dumps core!!! - sqeeze it */
    chdir("/"); /* we should not be able to write core file here */
#endif

    /* data server process cannot use message-passing library to abort
     * it simply exits, parent will get SIGCHLD and abort the program
     */
#if defined(DATA_SERVER)
    if(armci_me<0)_exit(1);
    else
#endif
    armci_msg_abort(code);
}


void armci_die(char *msg, int code)
{
    if(_armci_terminating)return;
    else _armci_terminating=1;

    if(SERVER_CONTEXT){
       fprintf(stdout,"%d(s):%s: %d\n",armci_me, msg, code); fflush(stdout);
       fprintf(stderr,"%d(s):%s: %d\n",armci_me, msg, code);
    }else{
      fprintf(stdout,"%d:%s: %d\n",armci_me, msg, code); fflush(stdout);
      fprintf(stderr,"%d:%s: %d\n",armci_me, msg, code);
    }
    armci_abort(code);
}


void armci_die2(char *msg, int code1, int code2)
{
    if(_armci_terminating)return;
    else _armci_terminating=1;

    if(SERVER_CONTEXT){
      fprintf(stdout,"%d(s):%s: (%d,%d)\n",armci_me,msg,code1,code2); 
	  fflush(stdout);
      fprintf(stderr,"%d(s):%s: (%d,%d)\n",armci_me,msg,code1,code2);
    }else{
      fprintf(stdout,"%d:%s: (%d,%d)\n",armci_me,msg,code1,code2); 
	  fflush(stdout);
      fprintf(stderr,"%d:%s: (%d,%d)\n",armci_me,msg,code1,code2);
    }

    armci_abort(code1);
}


void ARMCI_Error(char *msg, int code)
{
    armci_die(msg,code);
}


void armci_allocate_locks()
{
    /* note that if ELAN_ACC is defined the scope of locks is limited to SMP */ 
#if defined(HITACHI) || \
     (defined(QUADRICS) && defined(_ELAN_LOCK_H) && !defined(ELAN_ACC))
       armcill_allocate_locks(NUM_LOCKS);
#elif (defined(SYSV) || defined(WIN32) || defined(MMAP)) && !defined(HITACHI)
       if(armci_nproc == 1)return;    
       if(armci_master==armci_me)CreateInitLocks(NUM_LOCKS, &lockid);
       armci_msg_clus_brdcst(&lockid, sizeof(lockid));
       if(armci_master != armci_me)InitLocks(NUM_LOCKS, lockid);
#endif
    
}


void ARMCI_Set_shm_limit(unsigned long shmemlimit)
{
#if (defined(SYSV) || defined(WIN32)  || defined(MMAP)) && !defined(HITACHI)
#define EXTRASHM  1024   /* extra shmem used internally in ARMCI */
unsigned long limit;
    limit = armci_clus_info[armci_clus_me].nslave * shmemlimit + EXTRASHM;
    armci_set_shmem_limit(limit);
#endif
}


 
/*\ allocate and initialize memory locking data structure
\*/
void armci_init_memlock()
{
    int bytes = MAX_SLOTS*sizeof(memlock_t);
    int rc, msize_per_proc=bytes;
    
#ifdef MEMLOCK_SHMEM_FLAG    
    /* last proc on node allocates memlock flag in shmem */
    if(armci_clus_last == armci_me) bytes += sizeof(int);
#endif

    memlock_table_array = malloc(armci_nproc*sizeof(void*));
    if(!memlock_table_array) armci_die("malloc failed for ARMCI lock array",0);

    rc = ARMCI_Malloc(memlock_table_array, bytes);
    if(rc) armci_die("failed to allocate ARMCI memlock array",rc);

    armci_msg_barrier();

    bzero(memlock_table_array[armci_me],bytes);

#ifdef MEMLOCK_SHMEM_FLAG    
    /* armci_use_memlock_table is a pointer to local memory variable=1
     * we overwrite the pointer with address of shared memory variable 
     * armci_use_memlock_table and initialize it >0
     */
    armci_use_memlock_table = (int*) (msize_per_proc + 
                              (char*) memlock_table_array[armci_clus_last]);  
                              
    /* printf("%d: last=%d bytes=%d ptr =(%d, %d)\n",
           armci_me,armci_clus_last,bytes,armci_use_memlock_table,  
           memlock_table_array[armci_clus_last]); fflush(stdout); */

    if(armci_clus_last == armci_me) *armci_use_memlock_table =1+armci_me;

#endif

    armci_msg_barrier();
}


#if defined(SYSV) || defined(WIN32)
static void armci_check_shmmax()
{
  long mylimit, limit;
  mylimit = limit = (long) armci_max_region();
  armci_msg_bcast_scope(SCOPE_MASTERS, &limit, sizeof(long), 0);
  if(mylimit != limit){
     printf("%d:Shared memory limit detected by ARMCI is %ld bytes on node %s vs %ld on %s\n",
            armci_me,mylimit<<10,armci_clus_info[armci_clus_me].hostname,
            limit<<10, armci_clus_info[0].hostname);
     fflush(stdout); sleep(1);
     armci_die("All nodes must have the same SHMMAX limit if NO_SHM is not defined",0);
  }
}
#endif


int ARMCI_Init()
{
    _armci_initialized++;
    if(_armci_initialized>1)return 0;
#ifdef GA_USE_VAMPIR
    vampir_init(NULL,NULL,__FILE__,__LINE__);
    armci_vampir_init(__FILE__,__LINE__);
    vampir_begin(ARMCI_INIT,__FILE__,__LINE__);
#endif

    armci_nproc = armci_msg_nproc();
    armci_me = armci_msg_me();
	armci_usr_tid = THREAD_ID_SELF(); /*remember the main user thread id */

#ifdef _CRAYMPP
    cmpl_proc=-1;
#endif
#ifdef LAPI
    armci_init_lapi();
#endif
#ifdef QUADRICS
    shmem_init();
    /*printf("after shmem_init\n"); */
#endif

    armci_init_clusinfo();

    /* trap signals to cleanup ARMCI system resources in case of crash */
    if(armci_me==armci_master) ARMCI_ParentTrapSignals();
    ARMCI_ChildrenTrapSignals();

    armci_init_fence();

#if defined(SYSV) || defined(WIN32)
    /* init shared memory */
    if(ARMCI_Uses_shm() ) armci_shmem_init();
#   if defined(QUADRICS) && !defined(NO_SHM)
       if(armci_me == armci_master)armci_check_shmmax();
#   endif
#endif

    /* allocate locks: we need to do it before server is started */
    armci_allocate_locks();
#   if defined(DATA_SERVER) || defined(ELAN_ACC)
       if(armci_nclus >1) armci_start_server();
#   endif
    armci_msg_barrier();

    armci_init_memlock(); /* allocate data struct for locking memory areas */

/*    fprintf(stderr,"%d ready \n",armci_me);*/
    armci_msg_barrier();
    armci_msg_gop_init();
#ifdef GA_USE_VAMPIR
    vampir_end(ARMCI_INIT,__FILE__,__LINE__);
#endif    
    return 0;
}


void ARMCI_Finalize()
{
    _armci_initialized--;
    if(_armci_initialized)return;
#ifdef GA_USE_VAMPIR
    vampir_begin(ARMCI_FINALIZE,__FILE__,__LINE__);
#endif

    _armci_terminating =1;;
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
#ifdef GA_USE_VAMPIR
    vampir_end(ARMCI_FINALIZE,__FILE__,__LINE__);
    vampir_finalize(__FILE__,__LINE__);
#endif
}


#if !(defined(SYSV) || defined(WIN32))
void ARMCI_Set_shmem_limit(unsigned long shmemlimit)
{
   /* not applicable here
    * aborting would  make user's life harder
    */
}
#endif



void ARMCI_Copy(void *src, void *dst, int n)
{
#ifdef GA_USE_VAMPIR
 vampir_begin(ARMCI_COPY,__FILE__,__LINE__);
#endif
 armci_copy(src,dst,n);
#ifdef GA_USE_VAMPIR
 vampir_end(ARMCI_COPY,__FILE__,__LINE__);
#endif
}

extern void cpu_yield();
void armci_util_wait_int(volatile int *p, int val, int maxspin)
{
int count=0;
extern void cpu_yield();
       while(*p != val)
            if((++count)<maxspin) armci_util_spin(count,(int *)p);
            else{
               cpu_yield();
               count =0;
            }
}
  

/*\ returns 1 if specified process resides on the same smp node as calling task 
\*/
int ARMCI_Same_node(int proc)
{
   int direct=SAMECLUSNODE(proc);
   return direct;
}


/*\ blocks the calling process until a nonblocking operation represented
 *  by the user handle completes
\*/
int ARMCI_Wait(armci_hdl_t usr_hdl){
armci_ihdl_t nb_handle = (armci_ihdl_t)usr_hdl;
int success=0;
int direct=SAMECLUSNODE(nb_handle->proc);

    if(direct)return(success);
    if(nb_handle){
#     ifdef ARMCI_NB_WAIT
        if(nb_handle->tag==0){
              ARMCI_NB_WAIT(nb_handle->cmpl_info);
              return(success);
        }
#     endif
#     ifdef COMPLETE_HANDLE
       COMPLETE_HANDLE(nb_handle->bufid,nb_handle->tag,(&success));
#     endif
    }
    return(success);
}


static unsigned int _armci_nb_tag=0;
unsigned int _armci_get_next_tag(){
    return((++_armci_nb_tag));
}
