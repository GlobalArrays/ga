#ifndef _ARMCI_LOCKS_H_
#define _ARMCI_LOCKS_H_

#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif

#define MAX_LOCKS 1024
#define NUM_LOCKS MAX_LOCKS 

#if !(defined(PMUTEX) || defined(PSPIN) || defined(CYGNUS) )
#   include "spinlock.h"
#endif

#if !(defined(PMUTEX) || defined(PSPIN) || defined(SPINLOCK))
#   error cannot run
#endif

#if (defined(SPINLOCK) || defined(PMUTEX) || defined(PSPIN))
#   include "armci_shmem.h"
typedef struct {
    long off;
    long idlist[SHMIDLEN];
} lockset_t;
extern lockset_t lockid;
#endif

#if defined(PMUTEX)
#   warning SPINLOCK: pthread_mutex_lock
#   include <pthread.h>
#   include <unistd.h>
#   define NAT_LOCK(x,p) pthread_mutex_lock(_armci_int_mutexes +x)
#   define NAT_UNLOCK(x,p) pthread_mutex_unlock(_armci_int_mutexes +x)
#   define LOCK_T pthread_mutex_t
#   define PAD_LOCK_T LOCK_T
extern PAD_LOCK_T *_armci_int_mutexes;

#elif defined(PSPIN)
#   warning SPINLOCK: pthread_spin_lock
#   include <pthread.h>
#   include <unistd.h>
#   define NAT_LOCK(x,p) pthread_spin_lock(_armci_int_mutexes +x)
#   define NAT_UNLOCK(x,p) pthread_spin_unlock(_armci_int_mutexes +x)
#   define LOCK_T pthread_spinlock_t
#   define PAD_LOCK_T LOCK_T
extern PAD_LOCK_T *_armci_int_mutexes;

#elif defined(SPINLOCK)
#   define NAT_LOCK(x,p) armci_acquire_spinlock((LOCK_T*)(_armci_int_mutexes+(x)))
#   define NAT_UNLOCK(x,p) armci_release_spinlock((LOCK_T*)(_armci_int_mutexes+(x)))
extern PAD_LOCK_T *_armci_int_mutexes;

#elif defined(WIN32)
typedef int lockset_t;
extern void setlock(int);
extern void unsetlock(int);
#   define NAT_LOCK(x,p)   setlock(x)
#   define NAT_UNLOCK(x,p)  unsetlock(x)

#elif defined(CYGNUS)
typedef int lockset_t;
#   define NAT_LOCK(x,p) armci_die("does not run in parallel",0) 
#   define NAT_UNLOCK(x,p) armci_die("does not run in parallel",0)  

#elif defined(SYSV) || defined(MACX)
#   include "semaphores.h"
#   undef NUM_LOCKS
#   define NUM_LOCKS ((MAX_LOCKS< SEMMSL) ? MAX_LOCKS:SEMMSL)
#   define NAT_LOCK(x,p)   P_semaphore(x)
#   define NAT_UNLOCK(x,p)  V_semaphore(x)
#   ifndef _LOCKS_C_
#       define CreateInitLocks Sem_CreateInitLocks
#       define InitLocks Sem_InitLocks
#       define DeleteLocks Sem_DeleteLocks
#   endif

#else
#   error
#endif

extern void CreateInitLocks(int num, lockset_t *id);
extern void InitLocks(int num , lockset_t id);
extern void DeleteLocks(lockset_t id);

#define NATIVE_LOCK(x,p) if(armci_nproc>1) { NAT_LOCK(x,p); }
#define NATIVE_UNLOCK(x,p) if(armci_nproc>1) { NAT_UNLOCK(x,p); }

#endif /* _ARMCI_LOCKS_H_ */
