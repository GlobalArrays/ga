/* $Id: locks.c,v 1.5 1999-10-18 18:52:42 d3h325 Exp $ */
#include "locks.h"
#include "armcip.h"
#ifndef WIN32
#   include <unistd.h>
#endif
#include <stdio.h>

extern void armci_die(char*,int);

#define FILE_LEN 200

/********************* all SGI systems ****************/
#ifdef SGI
lockset_t lockset;
static char arena_name[FILE_LEN];
usptr_t *arena_ptr;
static int avail =0;

extern char *getenv(const char *);

void CreateInitLocks(int num_locks, lockset_t *lockid)
{
int i;
char *tmp;

   if(num_locks > NUM_LOCKS) armci_die("To many locks requested", num_locks);
   lockset.id = (int)getpid();
   if (!(tmp = getenv("ARENA_DIR"))) tmp = "/tmp";
   sprintf(arena_name,"%s/armci_arena%d.%ld", tmp,armci_clus_me,lockset.id);

  (void) usconfig(CONF_ARENATYPE, US_GENERAL);
  (void) usconfig(CONF_INITUSERS, (unsigned int)
                  armci_clus_info[armci_clus_me].nslave+1); /* +1 for server */
   arena_ptr = usinit(arena_name);    
   if(!arena_ptr) armci_die("Failed to Create Arena", 0);
/*   else fprintf(stderr,	"created arena %x\n",arena_ptr); */
 
   for(i=0; i<num_locks; i++){
       lockset.lock_array[i] = usnewlock(arena_ptr); 
       if(lockset.lock_array[i] == NULL) armci_die("Failed to Create Lock", i);
   }

   *lockid = lockset;
   avail = 1;
}   
   

void InitLocks(int num_locks, lockset_t lockid)
{
int i;
char *tmp;

/*   if(avail) armci_die("Arena already attached", avail); */
   lockset = lockid;
   if (!(tmp = getenv("ARENA_DIR"))) tmp = "/tmp";
   sprintf(arena_name,"%s/armci_arena%d.%ld", tmp,armci_clus_me,lockset.id);

   (void) usconfig(CONF_ARENATYPE, US_GENERAL);
   arena_ptr = usinit(arena_name);
   if(!arena_ptr) armci_die("Failed to Attach to Arena", lockid.id);
/*   else fprintf(stderr,	"attached arena %x\n",arena_ptr); */

   for(i=0; i<num_locks; i++){
       if(lockset.lock_array[i] == NULL) armci_die("Failed to Attach Lock", i);
   }
   avail = 1;
}   


void DeleteLocks(lockset_t lockid)
{
 /*  fprintf(stderr,	"deleting arena %x\n",arena_ptr);*/
  if(!avail)return;
  else avail = 0;
  usdetach (arena_ptr);
  arena_ptr = 0;
  (void)unlink(arena_name); /* ignore armci_die code -- file might be already gone*/
}

#endif

/***************** Convex/HP Exemplar ****************/
#if defined(CONVEX)
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


void CreateInitLocks(int num_locks, lockset_t *lockid)
{
int i;

   if(num_locks > NUM_LOCKS) armci_die("To many locks requested", num_locks);
   *lockid = (int)getpid();
   sprintf(file_name,"/tmp/ga.locks.%ld", *lockid);
   if ( (fd = open(file_name, O_RDWR|O_CREAT, 0666)) < 0 )
      armci_die("CreateInitLocks: failed to open temporary file",0);

   shmem_size = (NUM_LOCKS)*sizeof(lock_t);
   lock_array = (lock_t*) mmap((caddr_t) 0, shmem_size,
                     PROT_READ|PROT_WRITE,
                     MAP_ANONYMOUS|CNX_MAP_SEMAPHORE|MAP_SHARED, fd, 0);

   if(((unsigned)lock_array)%16)armci_die("CreateInitLocks: not aligned",0);
   for (i=0; i<NUM_LOCKS; i++)
       lock_array[i].state = 0;
}


void InitLocks(int num_locks, lockset_t  lockid)
{
int i;

   if(num_locks > NUM_LOCKS) armci_die("To many locks requested", num_locks);
   sprintf(file_name,"/tmp/ga.locks.%ld", lockid);
   if ( (fd = open(file_name, O_RDWR|O_CREAT, 0666)) < 0 )
      armci_die("InitLocks: failed to open temporary file",0);

   shmem_size = (NUM_LOCKS)*sizeof(lock_t);
   lock_array = (lock_t*)  mmap((caddr_t) 0, shmem_size,
                     PROT_READ|PROT_WRITE,
                     MAP_ANONYMOUS|CNX_MAP_SEMAPHORE|MAP_SHARED, fd, 0);
   if(((unsigned)lock_array)%16)armci_die("InitLocks: not aligned",0);
}


void DeleteLocks(lockset_t  lockid)
{
  lock_array = 0;
  (void)unlink(file_name); /* ignore armci_die code -- file might be already gone*/
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

#if defined(WIN32)
/****************************** Windows NT ********************************/
#include <process.h>
#include <windows.h>

HANDLE mutex_arr[NUM_LOCKS];
static int parent_pid;
static int num_alloc_locks=0;

void CreateInitLocks(int num_locks, lockset_t  *lockid)
{

   if(num_locks > NUM_LOCKS) armci_die("To many locks requested", num_locks);
   *lockid = parent_pid = _getpid();

   InitLocks(num_locks, *lockid);
}

    
void InitLocks(int num_locks, lockset_t lockid)
{
   int i;
   char lock_name[64];

   for(i=0;i<num_locks;i++){
        sprintf(lock_name,"ARMCImutex.%d.%d",(int)lockid,i);   
        mutex_arr[i] = CreateMutex(NULL, FALSE, lock_name);
        if( mutex_arr[i] == NULL) armci_die("armci_die creating mutexes",i);
   }
   num_alloc_locks = num_locks;
}

void DeleteLocks(lockset_t lockid)
{
    int i;
    for(i=0;i<num_alloc_locks;i++)
        (void)CloseHandle(mutex_arr[i]);
}

void setlock(int mutex)
{
    int rc;
    if(mutex >num_alloc_locks || mutex <0)armci_die("setlock: invalid",mutex);
    rc =WaitForSingleObject(mutex_arr[mutex],INFINITE);

    switch(rc) {
    case WAIT_OBJECT_0:  /* OK */ break;
    case WAIT_ABANDONED: /*abandoned: some process crashed holding mutex? */
                        armci_die("setlock: mutex abandoned",mutex);
    default:            /* some other problem */
                        fprintf(stderr,"WaitForSingleObject code=%d\n",rc);
                        armci_die("setlock: failed",mutex);
    }
}

void unsetlock(int mutex)
{
    if(mutex >num_alloc_locks || mutex <0)armci_die("unsetlock: invalid",mutex);
    if(ReleaseMutex(mutex_arr[mutex])==FALSE)armci_die("unsetlock: failed",mutex);
}


#endif

#ifdef CRAY_YMP

lock_t  cri_l[NUM_LOCKS];
#pragma  _CRI common cri_l

void CreateInitLocks(int num_locks, lockset_t  *lockid)
{
   int i;
   if(num_locks > NUM_LOCKS) armci_die("To many locks requested", num_locks);

   for(i=0;i<num_locks;i++)cri_l[i]=0;
}


void InitLocks(int num_locks, lockset_t lockid)
{
}


void DeleteLocks(lockset_t lockid)
{
}
#endif
