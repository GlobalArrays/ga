#if defined(__i386__)

#if defined(__GNUC__)
#include "tas-i386.h"
#define TESTANDSET testandset
#else
#define TESTANDSET gcc_testandset
extern int gcc_testandset(int *s);
#endif

#include <unistd.h>
#define INLINE inline
#define SPINLOCK 

#elif defined(SGI)
#include <mutex.h>
#define INLINE 
#define SPINLOCK 
#define TESTANDSET(x) __lock_test_and_set((x), 1)
#define RELEASE_SPINLOCK __lock_release

#elif defined(AIX)
#include <sys/atomic_op.h>
#define SPINLOCK 
#define INLINE 
#define TESTANDSET(x) (_check_lock((x), 0, 1)==TRUE) 
#define RELEASE_SPINLOCK(x) _clear_lock((x),0) 

#elif defined(SOLARIS)
#define SPINLOCK  
#define INLINE 
#define TESTANDSET(x) (!_lock_try((x))) 
#define RELEASE_SPINLOCK _lock_clear 

#elif defined(HPUX__)
extern int _acquire_lock();
extern void _release_lock();
#define SPINLOCK  
#define INLINE 
#define TESTANDSET(x) (!_acquire_lock((x))) 
#define RELEASE_SPINLOCK _release_lock 

#endif



#ifdef SPINLOCK

#include <stdio.h>

#ifndef DBL_PAD
#   define DBL_PAD 8
#endif

/* make sure that locks are not sharing the same cache line */
typedef struct{
double  lock[DBL_PAD];
}pad_lock_t;

#define LOCK_T int
#define PAD_LOCK_T pad_lock_t


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
  *mutex =0;
}

#endif

#endif
