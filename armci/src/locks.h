#ifndef _ARMCI_LOCKS_H_
#define _ARMCI_LOCKS_H_
#include <sys/types.h>
#define NUM_LOCKS 128

#ifdef SGI

#  define SGI_SPINS 100
#  include <ulocks.h>

   typedef struct {
        int id;
        ulock_t * lock_array[NUM_LOCKS];
   }lockset_t;

   extern lockset_t lockset;
#  define NATIVE_LOCK(x)    (void) uswsetlock(lockset.lock_array[(x)],SGI_SPINS)
#  define NATIVE_UNLOCK(x)  (void) usunsetlock(lockset.lock_array[(x)])

#elif defined(CONVEX)

#  include <sys/cnx_ail.h>
   typedef struct{
        unsigned state;
        unsigned pad[15];
   } lock_t;

   typedef int lockset_t;
   extern lock_t *lock_array;
   extern void setlock(unsigned * volatile lp);
   extern void unsetlock(unsigned  * volatile lp);
#  define NATIVE_LOCK(x)    (void) setlock(&lock_array[x].state)
#  define NATIVE_UNLOCK(x)  (void) unsetlock(&lock_array[(x)].state)

#elif defined(WIN32)

   typedef int lockset_t;
   extern void setlock(int);
   extern void unsetlock(int);
#  define NATIVE_LOCK(x)   setlock(x)
#  define NATIVE_UNLOCK(x) unsetlock(x)

#elif defined(CRAY_YMP)
#  include <tfork.h>

    typedef int lockset_t;
    extern  lock_t cri_l[NUM_LOCKS];
#  pragma  _CRI common cri_l

#  define NATIVE_LOCK(x)   t_lock(cri_l+(x))
#  define NATIVE_UNLOCK(x) t_unlock(cri_l+(x))

#elif defined(CRAY_T3E)
#  include <limits.h>
   static long armci_lock_var=0;
   typedef int lockset_t;
#  define INVALID (long)(_INT_MIN_64 +1)
#  define NATIVE_LOCK(x)  while( shmem_swap(&armci_lock_var,INVALID,(x)) )
#  define NATIVE_UNLOCK(x) shmem_swap(&armci_lock_var, 0, (x))

#elif defined(SYSV)

#  include "semaphores.h"
#  undef NUM_LOCKS
#  define NUM_LOCKS SEMMSL

#  define NATIVE_LOCK(x)   P_semaphore(x)
#  define NATIVE_UNLOCK(x) V_semaphore(x)

#elif defined(CYGNUS)

typedef int lockset_t;
#  define NATIVE_LOCK(x) if(armci_nproc>1)armci_die("does not run in parallel",0) 
#  define NATIVE_UNLOCK(x) if(armci_nproc>1)armci_die("does not run in parallel",0)  

#else

typedef int lockset_t;

#endif

extern void CreateInitLocks(int num, lockset_t *id);
extern void InitLocks(int num , lockset_t id);
extern void DeleteLocks(lockset_t id);

#endif
