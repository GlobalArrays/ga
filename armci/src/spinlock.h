#if defined(LINUX) || defined(CYGWIN)

#if defined(__i386__) || defined(__alpha) || defined(__ia64)
#  define SPINLOCK 
#  if defined(__GNUC__)
#     if defined(__i386__)
#          include "tas-i386.h"
#     elif  defined(__ia64)
#          include "tas-ia64.h"
#     else
#          include "tas-alpha.h"
#     endif
#     define TESTANDSET testandset
#  else
#     define TESTANDSET gcc_testandset
#     define RELEASE_SPINLOCK gcc_clear_spinlock
#  endif
   extern int gcc_testandset();
   extern void gcc_clear_spinlock();
#endif

#elif defined(DECOSF)
#include "tas-alpha.h"
#define SPINLOCK 
#define TESTANDSET testandset

#elif defined(SGI)
#include <mutex.h>
#define SPINLOCK 
#define TESTANDSET(x) __lock_test_and_set((x), 1)
#define RELEASE_SPINLOCK __lock_release

#elif defined(AIX)
#include <sys/atomic_op.h>
#define SPINLOCK 
#define TESTANDSET(x) (_check_lock((x), 0, 1)==TRUE) 
#define RELEASE_SPINLOCK(x) _clear_lock((x),0) 

#elif defined(SOLARIS)
#define SPINLOCK  
#define TESTANDSET(x) (!_lock_try((x))) 
#define RELEASE_SPINLOCK _lock_clear 

#elif defined(MACX)
#include "tas-ppc.h"
#define SPINLOCK  
#define TESTANDSET(x) (! __compare_and_swap((long int *)(x),0,1)) 

#elif defined(HPUX__)
extern int _acquire_lock();
extern void _release_lock();
#define SPINLOCK  
#define TESTANDSET(x) (!_acquire_lock((x))) 
#define RELEASE_SPINLOCK _release_lock 
#endif



#ifdef SPINLOCK

#include <stdio.h>
#include <unistd.h>

#ifndef DBL_PAD
#   define DBL_PAD 8
#endif

/* make sure that locks are not sharing the same cache line */
typedef struct{
double  lock[DBL_PAD];
}pad_lock_t;

#define LOCK_T int
#define PAD_LOCK_T pad_lock_t

#if defined(__GNUC__)
#   define INLINE inline 
#else
#   define INLINE 
#endif


static INLINE void armci_init_spinlock(LOCK_T *mutex)
{
  *mutex =0;
}

static INLINE void armci_acquire_spinlock(LOCK_T *mutex)
{
int loop=0, maxloop =100;

   while (TESTANDSET(mutex)){
      loop++;
      if(loop==maxloop){ 
#if 0
         extern int armci_me;
         printf("%d:spinlock sleeping\n",armci_me); fflush(stdout);
#endif
         usleep(1);
         loop=0;
      }
  }
}



#ifdef  RELEASE_SPINLOCK
#  define  armci_release_spinlock RELEASE_SPINLOCK
#else
static INLINE void armci_release_spinlock(LOCK_T *mutex)
{
#ifdef MEMORY_BARRIER
  MEMORY_BARRIER ();
#endif
  *mutex =0;
}

#endif

#endif
