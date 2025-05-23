#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* $Id: semaphores.c,v 1.12 2005-03-10 19:11:23 vinodtipparaju Exp $ */
#include "semaphores.h"
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_UNISTD_H
#   include <unistd.h>
#endif

int num_sem_alloc=0;
void perror();

#include "armcip.h"

struct sembuf sops;
int semaphoreID;

int SemGet(num_sem)
    int num_sem;
{
  semaphoreID = semget(IPC_PRIVATE,num_sem, IPC_CREAT | 0600);
  if(semaphoreID<0){
    fprintf(stderr," Semaphore Allocation Failed \nsuggestions to fix the problem: \n");
    fprintf(stderr," 1. run ipcs and ipcrm -s commands to clean any semaphore ids\n");
    fprintf(stderr," 2. verify if constant SEMMSL defined in file semaphore.h is set correctly for your system\n");
    fprintf(stderr," 3. recompile semaphore.c\n");
       sleep(1);
       perror("Error message from failed semget:");
       armci_die(" exiting ...", num_sem);
    }
       
    num_sem_alloc = num_sem;
    return(semaphoreID);
}

void SemInit(id,value)
    int id,value;
{
  int i, semid, num_sem;
  union semun semctl_arg;

    semctl_arg.val = value;

    if(id == ALL_SEMS){ semid = 0; num_sem = num_sem_alloc;}
      else { semid = id; num_sem = 1;}

    for(i=0; i< num_sem; i++){ 
       if( semctl(semaphoreID, semid, SETVAL,semctl_arg )<0){ 
         perror((char*)0);
         armci_die("SemInit error",id);
       }
       semid++;
    }
}


/*  release semaphore(s) */
void SemDel()
{
    union semun dummy;

    /* this is only to avoid compiler whinning about the unitialized variable*/
    dummy.val=0; 

    (void) semctl(semaphoreID,0,IPC_RMID,dummy);
}


void Sem_CreateInitLocks(int num, lockset_t *id)
{
     *id = SemGet(num);
     SemInit(ALL_SEMS,1);
}


void Sem_InitLocks(int num, lockset_t id)
{
    semaphoreID = id;
    num_sem_alloc = num;
}


void Sem_DeleteLocks(lockset_t id)
{
    union semun dummy;

    /* this is only to avoid compiler whinning about the unitialized variable*/
    dummy.val=0; 

    (void) semctl(id,0,IPC_RMID,dummy);
}


