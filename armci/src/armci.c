/* $Id: armci.c,v 1.27 1999-11-24 01:41:08 d3h325 Exp $ */

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

#define MEMLOCK_SHMEM_FLAG  

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
#ifndef WIN32
    ARMCI_RestoreSignals();
#endif
#endif
}


void armci_perror_msg()
{
    char perr_str[80];
    if(!errno)return;
    sprintf(perr_str,"Last System Error Message from Task %d:",armci_me);
    perror(perr_str);
}

void armci_die(char *msg, int code)
{
   
    if(armci_terminating)return;
    else armci_terminating=1;

    fprintf(stdout,"%d:%s: %d\n",armci_me, msg, code); fflush(stdout);
    fprintf(stderr,"%d:%s: %d\n",armci_me, msg, code);
    armci_perror_msg();
    ARMCI_Cleanup();

    /* data server process cannot use message-passing library to abort
     * it simply exits, parent will get SIGCHLD and abort the program
     */
#if defined(DATA_SERVER)
    if(armci_me<0)_exit(1);
    else
#endif
    armci_msg_abort(code);
}


void armci_die2(char *msg, int code1, int code2)
{
    if(armci_terminating)return;
    else armci_terminating=1;

    fprintf(stdout,"%d:%s: (%d,%d)\n",armci_me,msg,code1,code2); fflush(stdout);
    fprintf(stderr,"%d:%s: (%d,%d)\n",armci_me,msg,code1,code2);
    armci_perror_msg();
    ARMCI_Cleanup();

    /* data server process cannot use message-passing library to abort
     * it simply exits, parent will get SIGCHLD and abort the program
     */
#if defined(DATA_SERVER)
    if(armci_me<0)_exit(1);
    else
#endif
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




int ARMCI_Init()
{

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

    armci_init_memlock();

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
 armci_copy(src,dst,n);
}
