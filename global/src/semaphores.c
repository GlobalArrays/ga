#include "semaphores.h"
#include <stdio.h>

int sem_init=0;
void perror();
#ifdef SUN
int  fprintf();
void fflush();
int semget(),semctl();
#endif


#define MAX_SEM  10  

int SemGet(num_sem)
    int num_sem;
{
    if(num_sem<1)return(0);
    if(num_sem>MAX_SEM)return(0);
 
    semaphoreID = semget(IPC_PRIVATE,num_sem,0600);
    if(semaphoreID<0){
       fprintf(stderr,"SemGet failed \n");
       perror((char*)0);
    }
       
    sem_init = num_sem;
    return(semaphoreID);
}

void SemInit(id,value)
    int id,value;
{
#if defined(ARDENT) || defined(ENCORE) || defined(SEQUENT) || \
    defined(ULTRIX) || defined(AIX)    || defined(HPUX) || defined(KSR)
union semun {
   long val;
   struct semid_ds *buf;
   ushort *array;
} semctl_arg;
#else
union semun {
   int val;
   struct semid_ds *buf;
   ushort *array;
} semctl_arg;
#endif

    /*fprintf(stderr,"SemInit %d %d\n",id,value);*/
   
    semctl_arg.val = value;
    if(id >= sem_init || id<0 ) 
      fprintf(stderr,"attempt to intialize invalid semaphore %d %d\n",
                                                         id,sem_init);
    else if( semctl(semaphoreID, id,SETVAL,semctl_arg )<0){ 
         fprintf(stderr,"SemInit error\n");
         perror((char*)0);
    }
}


/*  release semaphore(s) */
void SemDel()
{
    (void) semctl(semaphoreID,NULL,IPC_RMID,NULL);
}

