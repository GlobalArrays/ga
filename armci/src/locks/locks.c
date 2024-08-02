#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* $Id: locks.c,v 1.15.6.1 2006-12-14 13:24:36 manoj Exp $ */
#define _LOCKS_C_
#include "armcip.h"
#include "locks.h"
#if HAVE_UNISTD_H
#   include <unistd.h>
#endif
#if HAVE_STDIO_H
#   include <stdio.h>
#endif

PAD_LOCK_T *_armci_int_mutexes;

#if !defined(armci_die)
extern void armci_die(char*,int);
#endif

#if defined(SPINLOCK) || defined(PMUTEX) || defined(PSPIN)

void **ptr_arr;

void CreateInitLocks(int num_locks, lockset_t *plockid)
{
int locks_per_proc, size;
  ptr_arr = (void**)malloc(armci_nproc*sizeof(void*));
  locks_per_proc = (num_locks*armci_nclus)/armci_nproc + 1;
  size=locks_per_proc*sizeof(PAD_LOCK_T);
  PARMCI_Malloc(ptr_arr, size);
  _armci_int_mutexes = (PAD_LOCK_T*) ptr_arr[armci_master];
  
  if(!_armci_int_mutexes) armci_die("Failed to create spinlocks",size);

#ifdef PMUTEX
  if(armci_me == armci_master) {
       int i;
       pthread_mutexattr_t pshared;
       if(pthread_mutexattr_init(&pshared))
            armci_die("armci_allocate_locks: could not init mutex attr",0);
#      ifndef LINUX
         if(pthread_mutexattr_setpshared(&pshared,PTHREAD_PROCESS_SHARED))
            armci_die("armci_allocate_locks: could not set PROCESS_SHARED",0);
#      endif

       for(i=0; i< locks_per_proc*armci_clus_info[armci_clus_me].nslave; i++){
             if(pthread_mutex_init(_armci_int_mutexes+i,&pshared))
                armci_die("armci_allocate_locks: could not init mutex",i);
       }
  }
#elif defined(PSPIN)
  if(armci_me == armci_master) {
       for(i=0; i< locks_per_proc*armci_clus_info[armci_clus_me].nslave; i++){
             if(pthread_spin_init(_armci_int_mutexes+i,PTHREAD_PROCESS_SHARED))
                armci_die("armci_allocate_locks: could not init mutex",i);
       }
  }
#else
  bzero((char*)ptr_arr[armci_me],size);
#endif
} 

void InitLocks(int num_locks, lockset_t lockid)
{
    /* what are you doing here ? 
       All processes should've called CreateInitLocks().
       Check preprocessor directtives and see lock allocation in armci_init */
    armci_die("InitLocks(): what are you doing here ?",armci_me);
}


void DeleteLocks(lockset_t lockid)
{
  _armci_int_mutexes = (PAD_LOCK_T*)0;
}

#elif defined(WIN32)
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

#else
/*********************** every thing else *************************/

void CreateInitLocks(int num_locks, lockset_t  *lockid)
{}

void InitLocks(int num_locks, lockset_t lockid)
{
}


void DeleteLocks(lockset_t lockid)
{
}

#endif

