#include "sgi.locks.h"



ulock_t *lock_array[NUM_LOCKS];
static char arena_name[ARENA_LEN];
usptr_t *arena_ptr;

void CreateInitLocks(ulock_t **lock_array, long num_locks, long *lockid)
{
#include <unistd.h>
#include "types.f2c.h"
extern Integer cluster_nodes;
long i;

   if(num_locks > NUM_LOCKS) ERROR("To many locks requested", num_locks);
   *lockid = (long)getpid();
   sprintf(arena_name,"ga.arena.%ld", *lockid);

  (void) usconfig(CONF_ARENATYPE, US_GENERAL);
  (void) usconfig(CONF_INITUSERS, (unsigned int)cluster_nodes); 
   arena_ptr = usinit(arena_name);    
   if(!arena_ptr) ERROR("Failed to Create Arena", *lockid);
 
   for(i=0; i<num_locks; i++){
       lock_array[i] = usnewlock(arena_ptr); 
       if(lock_array[i] == NULL) ERROR("Failed to Create Lock", i);
   }
}   
   

void InitLocks(ulock_t **lock_array, long num_locks, long lockid)
{
long i;
   sprintf(arena_name,"ga.arena.%ld", lockid);
   (void) usconfig(CONF_ARENATYPE, US_GENERAL);
   arena_ptr = usinit(arena_name);
   if(!arena_ptr) ERROR("Failed to Attach to Arena", lockid);
   for(i=0; i<num_locks; i++){
       if(lock_array[i] == NULL) ERROR("Failed to Attach Lock", i);
   }
}   


void DeleteLocks(long lockid)
{
  usdetach (arena_ptr);
  arena_ptr = 0;
/*  sprintf(arena_name,"ga.arena.%ld", lockid);*/
  (void)unlink(arena_name); /* ignore error code -- file might be already gone*/
}

