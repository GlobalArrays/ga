#include <sys/types.h>

#define NUM_LOCKS 128
#define FILE_LEN 200
#define ERROR ga_error

#ifdef SGI
#  include <ulocks.h>
   extern ulock_t *lock_array[NUM_LOCKS];
#  define NATIVE_LOCK(x)    (void) ussetlock(lock_array[(x)])
#  define NATIVE_UNLOCK(x)  (void) usunsetlock(lock_array[(x)])
#elif defined(CONVEX)
#  include <sys/cnx_ail.h>
   typedef struct{
        unsigned state;
        unsigned pad[15];
   } lock_t;
   extern lock_t *lock_array;
   extern void setlock(unsigned * volatile lp);
   extern void unsetlock(unsigned  * volatile lp);
#  define NATIVE_LOCK(x)    (void) setlock(&lock_array[x].state)
#  define NATIVE_UNLOCK(x)  (void) unsetlock(&lock_array[(x)].state)
#endif


extern void CreateInitLocks(long, long *);
extern void InitLocks(long, long);
extern void DeleteLocks(long);
