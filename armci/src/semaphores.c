#include "semaphores.h"
#include <stdio.h>

int num_sem_alloc=0;
void perror();
#ifdef SUN
int  fprintf();
void fflush();
int semget(),semctl();
#endif


struct sembuf sops;
int semaphoreID;

#if defined(ARDENT) || defined(ENCORE) || defined(SEQUENT) || \
    defined(ULTRIX) || defined(AIX)    || defined(HPUX) || defined(KSR)
union semun {
   long val;
   struct semid_ds *buf;
   ushort *array;
};
#elif !defined(LINUX) && !defined(SUN) && !defined(SGI)
union semun {
   int val;
   struct semid_ds *buf;
   ushort *array;
};
#endif


int SemGet(num_sem)
    int num_sem;
{
  semaphoreID = semget(IPC_PRIVATE,num_sem,0600);
  if(semaphoreID<0){
    fprintf(stderr," Semaphore Allocation Failed \nsuggestions to fix the problem: \n");
    fprintf(stderr," 1. run ipcs and ipcrm -s commands to clean any semaphore ids\n");
    fprintf(stderr," 2. verify if constant SEMMSL defined in file semaphore.h is set correctly for your system\n");
    fprintf(stderr," 3. recompile semaphore.c\n");
       sleep(1);
       perror((char*)0);
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
    (void) semctl(semaphoreID,0,IPC_RMID,NULL);
}


void CreateInitLocks(int num, lockset_t *id)
{
     *id = SemGet(num);
     SemInit(ALL_SEMS,1);
}


void InitLocks(int num, lockset_t id)
{
    semaphoreID = id;
    num_sem_alloc = num;
}


void DeleteLocks(lockset_t id)
{
    union semun dummy;
    (void) semctl(id,0,IPC_RMID,dummy);
}


