#ifndef _ARMCI_LOCKS_H_
#define _ARMCI_LOCKS_H_
#include <sys/types.h>
#define MAX_LOCKS 128 
#define NUM_LOCKS MAX_LOCKS 

#ifndef EXTERN
#   define EXTERN extern
#endif

#ifndef CYGNUS
#include "spinlock.h"
#endif

#if 0
/* code disabled until more tests of pthread locking is done */
#if !defined(SPINLOCK) &&  defined(SGI) 
/* check if Pthreads locks in Posix 1003.1c support processes*/
#include <pthread.h>
#include <unistd.h>
#ifdef _POSIX_THREAD_PROCESS_SHARED
#  define PMUTEXES
#endif
#endif
#endif


#if defined(PTHREADS) && !(defined(PMUTEXES) || defined(SPINLOCK))
# if defined(LINUX) && defined(__sparc__) && defined(GM)
#    define PMUTEXES
#    include <pthread.h>
# else
     cannot run
# endif
#endif

#if defined(SPINLOCK) || defined(PMUTEXES)
#  include "shmem.h"
   typedef struct {
     long off;
     long idlist[SHMIDLEN];
   }lockset_t;
   extern lockset_t lockid;
#endif


#if defined(SPINLOCK)

#  define NAT_LOCK(x) armci_acquire_spinlock((LOCK_T*)(_armci_int_mutexes+(x)))
#  define NAT_UNLOCK(x) armci_release_spinlock((LOCK_T*)(_armci_int_mutexes+(x)))
   EXTERN PAD_LOCK_T *_armci_int_mutexes;

#elif defined(PMUTEXES)

#  define NAT_LOCK(x) pthread_mutex_lock(_armci_int_mutexes +x)
#  define NAT_UNLOCK(x) pthread_mutex_unlock(_armci_int_mutexes +x)
#  define LOCK_T pthread_mutex_t
#  define PAD_LOCK_T LOCK_T
   EXTERN PAD_LOCK_T *_armci_int_mutexes;

#elif defined(SGI)

#  define SGI_SPINS 100
#  include <ulocks.h>

   typedef struct {
        int id;
        ulock_t * lock_array[NUM_LOCKS];
   }lockset_t;

   extern lockset_t lockset;
#  define NAT_LOCK(x)    (void) uswsetlock(lockset.lock_array[(x)],SGI_SPINS)
#  define NAT_UNLOCK(x)  (void) usunsetlock(lockset.lock_array[(x)])

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
#  define NAT_LOCK(x)    (void) setlock(&lock_array[x].state)
#  define NAT_UNLOCK(x)  (void) unsetlock(&lock_array[(x)].state)

#elif defined(WIN32)

   typedef int lockset_t;
   extern void setlock(int);
   extern void unsetlock(int);
#  define NAT_LOCK(x)   setlock(x)
#  define NAT_UNLOCK(x)  unsetlock(x)

#elif defined(CRAY_YMP)
#  include <tfork.h>

    typedef int lockset_t;
    extern  lock_t cri_l[NUM_LOCKS];
#  pragma  _CRI common cri_l

#  define NAT_LOCK(x)   t_lock(cri_l+(x))
#  define NAT_UNLOCK(x) t_unlock(cri_l+(x))

#elif defined(CRAY_T3E) || defined(QUADRICS)
#  include <limits.h>
#ifdef DECOSF
#  define  _INT_MIN_64 (LONG_MAX-1)
#endif
   static long armci_lock_var=0;
   typedef int lockset_t;
#  define INVALID (long)(_INT_MIN_64 +1)
#  define NAT_LOCK(x) while( shmem_swap(&armci_lock_var,INVALID,(x)) )
#  define NAT_UNLOCK(x) shmem_swap(&armci_lock_var, 0, (x))


#elif  defined(SYSV) && defined(LAPI)

int **_armci_int_mutexes;
#  define NAT_LOCK(x)  armci_lapi_lock(_armci_int_mutexes[armci_master]+x)
#  define NAT_UNLOCK(x)  armci_lapi_unlock(_armci_int_mutexes[armci_master]+x)
   typedef int lockset_t;

#elif defined(CYGNUS)

   typedef int lockset_t;
#  define NAT_LOCK(x) armci_die("does not run in parallel",0) 
#  define NAT_UNLOCK(x) armci_die("does not run in parallel",0)  

#elif defined(LAPI)

#  include <pthread.h>
   typedef int lockset_t;
   extern pthread_mutex_t _armci_mutex_thread;
#  define NAT_LOCK(x)   pthread_mutex_lock(&_armci_mutex_thread)
#  define NAT_UNLOCK(x) pthread_mutex_unlock(&_armci_mutex_thread)


#elif defined(FUJITSU)
   typedef int lockset_t;
#  include "fujitsu-vpp.h"

#elif defined(SYSV)

#  include "semaphores.h"
#  undef NUM_LOCKS
#  define NUM_LOCKS ((MAX_LOCKS< SEMMSL) ? MAX_LOCKS:SEMMSL)

#  define NAT_LOCK(x)   P_semaphore(x)
#  define NAT_UNLOCK(x)  V_semaphore(x)

#ifndef _LOCKS_C_
#define CreateInitLocks Sem_CreateInitLocks
#define InitLocks Sem_InitLocks
#define DeleteLocks Sem_DeleteLocks
#endif

#else

   typedef int lockset_t;
#  define NAT_LOCK(x) 
#  define NAT_UNLOCK(x) 

#endif

extern void CreateInitLocks(int num, lockset_t *id);
extern void InitLocks(int num , lockset_t id);
extern void DeleteLocks(lockset_t id);

#define NATIVE_LOCK(x) if(armci_nproc>1) { NAT_LOCK(x); }
#define NATIVE_UNLOCK(x) if(armci_nproc>1) { NAT_UNLOCK(x); }

#endif
