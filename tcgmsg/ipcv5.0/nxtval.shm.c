/* $Id: nxtval.shm.c,v 1.6 2002-09-21 17:49:20 vinod Exp $ */

#include "tcgmsgP.h"
long nxtval_counter=0;
long *nxtval_shmem = &nxtval_counter;

#ifdef GA_USE_VAMPIR
#include "tcgmsg_vampir.h"
#endif

#define LEN 2
#define INCR 1                 /* increment for NXTVAL */
#define BUSY -1L               /* indicates somebody else updating counter*/


#if defined(__i386__) && defined(__GNUC__)
#   define TESTANDSET testandset
#   define LOCK if(nproc>1)acquire_spinlock((int*)(nxtval_shmem+1))
#   define UNLOCK if(nproc>1)release_spinlock((int*)(nxtval_shmem+1))

static inline int testandset(int *spinlock)
{
  int ret;
  __asm__ __volatile__("xchgl %0, %1"
        : "=r"(ret), "=m"(*spinlock)
        : "0"(1), "m"(*spinlock));

  return ret;
}

static void acquire_spinlock(int *mutex)
{
int loop=0, maxloop =100;
   while (TESTANDSET(mutex)){
      loop++;
      if(loop==maxloop){ usleep(1); loop=0; }
  }
}

static release_spinlock(int *mutex)
{
   *mutex =0;
}

#endif

#ifndef LOCK
#   define LOCK  if(nproc>1)Error("nxtval: sequential version with silly mproc ", (Integer) *mproc);
#   define UNLOCK
#endif


long NXTVAL_(long *mproc)
/*
  Get next value of shared counter.

  mproc > 0 ... returns requested value
  mproc < 0 ... server blocks until abs(mproc) processes are queued
                and returns junk
  mproc = 0 ... indicates to server that I am about to terminate

*/
{
  long shmem_swap();
  long local;
  long sync_type= INTERNAL_SYNC_TYPE;
  long nproc=  NNODES_(); 
  long server=nproc-1; 

#ifdef GA_USE_VAMPIR
  long me = NODEID_();
  vampir_begin(TCGMSG_NXTVAL,__FILE__,__LINE__);
#endif

     if (DEBUG_) {
       (void) printf("%2ld: nxtval: mproc=%ld\n",NODEID_(), *mproc);
       (void) fflush(stdout);
     }

     if (*mproc < 0) {
           SYNCH_(&sync_type);
           /* reset the counter value to zero */
           if( NODEID_() == server) *nxtval_shmem = 0;
           SYNCH_(&sync_type);
     }
     if (*mproc > 0) {
#ifdef GA_USE_VAMPIR
           vampir_start_comm(server,me,sizeof(long),TCGMSG_NXTVAL);
#endif

           LOCK;
             local = *nxtval_shmem;
             *nxtval_shmem += INCR;
           UNLOCK;

#ifdef GA_USE_VAMPIR
           vampir_end_comm(server,me,sizeof(long),TCGMSG_NXTVAL);
#endif
     }

#ifdef GA_USE_VAMPIR
  vampir_end(TCGMSG_NXTVAL,__FILE__,__LINE__);
#endif
  return local;
}

