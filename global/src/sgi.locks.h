#include <ulocks.h>
#include <sys/types.h>

#define NUM_LOCKS 100
#define ARENA_LEN 200

#define ERROR ga_error
#define SGI_LOCK(x)   (void) ussetlock(lock_array[(x)])
#define SGI_UNLOCK(x)  (void) usunsetlock(lock_array[(x)])


extern ulock_t *lock_array[NUM_LOCKS];
extern usptr_t *arena_ptr;

extern void CreateInitLocks(ulock_t **lock_array, long num_locks, long *lockid);
extern void InitLocks(ulock_t **lock_array, long num_locks, long lockid);
extern void DeleteLocks(long lockid);
