#include "locks.h"
#include <unistd.h>
#include <stdio.h>


#ifdef SGI
ulock_t *lock_array[NUM_LOCKS];
static char arena_name[FILE_LEN];
usptr_t *arena_ptr;

extern char *getenv(const char *);

void CreateInitLocks(long num_locks, long *lockid)
{
#include "types.f2c.h"
extern Integer cluster_nodes;
long i;
char *tmp;

   if(num_locks > NUM_LOCKS) ERROR("To many locks requested", num_locks);
   *lockid = (long)getpid();
   if (!(tmp = getenv("ARENA_DIR"))) tmp = "/tmp";
   sprintf(arena_name,"%s/ga.arena.%ld", tmp,*lockid);

  (void) usconfig(CONF_ARENATYPE, US_GENERAL);
  (void) usconfig(CONF_INITUSERS, (unsigned int)cluster_nodes); 
   arena_ptr = usinit(arena_name);    
   if(!arena_ptr) ERROR("Failed to Create Arena", *lockid);
 
   for(i=0; i<num_locks; i++){
       lock_array[i] = usnewlock(arena_ptr); 
       if(lock_array[i] == NULL) ERROR("Failed to Create Lock", i);
   }
}   
   

void InitLocks(long num_locks, long lockid)
{
long i;
char *tmp;
   if (!(tmp = getenv("ARENA_DIR"))) tmp = "/tmp";
   sprintf(arena_name,"%s/ga.arena.%ld", tmp, lockid);
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
  (void)unlink(arena_name); /* ignore error code -- file might be already gone*/
}

#endif

#ifdef CONVEX
#include <sys/param.h>
#include <sys/file.h>
#include <sys/cnx_mman.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/cnx_ail.h>

lock_t *lock_array;
static char file_name[FILE_LEN];
static int fd=-1;
static unsigned shmem_size=-1;


void CreateInitLocks(long num_locks, long *lockid)
{
long i;

   if(num_locks > NUM_LOCKS) ERROR("To many locks requested", num_locks);
   *lockid = (long)getpid();
   sprintf(file_name,"/tmp/ga.locks.%ld", *lockid);
   if ( (fd = open(file_name, O_RDWR|O_CREAT, 0666)) < 0 )
      ERROR("CreateInitLocks: failed to open temporary file",0);

   shmem_size = (NUM_LOCKS)*sizeof(lock_t);
   lock_array = (lock_t*) mmap((caddr_t) 0, shmem_size,
                     PROT_READ|PROT_WRITE,
                     MAP_ANONYMOUS|CNX_MAP_SEMAPHORE|MAP_SHARED, fd, 0);

   if(((unsigned)lock_array)%16)ERROR("CreateInitLocks: not aligned",0);
   for (i=0; i<NUM_LOCKS; i++)
       lock_array[i].state = 0;
}


void InitLocks(long num_locks, long lockid)
{
long i;

   if(num_locks > NUM_LOCKS) ERROR("To many locks requested", num_locks);
   sprintf(file_name,"/tmp/ga.locks.%ld", lockid);
   if ( (fd = open(file_name, O_RDWR|O_CREAT, 0666)) < 0 )
      ERROR("InitLocks: failed to open temporary file",0);

   shmem_size = (NUM_LOCKS)*sizeof(lock_t);
   lock_array = (lock_t*)  mmap((caddr_t) 0, shmem_size,
                     PROT_READ|PROT_WRITE,
                     MAP_ANONYMOUS|CNX_MAP_SEMAPHORE|MAP_SHARED, fd, 0);
   if(((unsigned)lock_array)%16)ERROR("InitLocks: not aligned",0);
}


void DeleteLocks(long lockid)
{
  lock_array = 0;
  (void)unlink(file_name); /* ignore error code -- file might be already gone*/
  (void)munmap((char *) shmem_size, 0);
}


void setlock(unsigned * volatile lp)
{
volatile unsigned flag;

       flag = fetch_and_inc32(lp);
       while(flag){
          flag = fetch32(lp);
       }
}
       

void unsetlock(unsigned * volatile lp)
{
       (void)fetch_and_clear32(lp);
}

#endif

