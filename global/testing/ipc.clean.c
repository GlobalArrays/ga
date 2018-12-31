#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif
#if HAVE_SYS_IPC_H
#   include <sys/ipc.h>
#endif
#if HAVE_SYS_SEM_H
#   include <sys/sem.h>
#endif
#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_ERRNO_H
#   include <errno.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif

#if !HAVE_UNION_SEMUN
union semun {
        int val;                    /* value for SETVAL */
        struct semid_ds *buf;       /* buffer for IPC_STAT, IPC_SET */
        unsigned short int *array;  /* array for GETALL, SETALL */
        struct seminfo *__buf;      /* buffer for IPC_INFO */
};
#endif

#define MAX_SEM  10 
 
struct sembuf sops;
int semaphoreID;
int sem_init=0;

#define P_      -1
#define V_       1
#define P(s)  \
{\
  sops.sem_num = (s);\
  sops.sem_op  =  P_;\
  sops.sem_flg =  0; \
  semop(semaphoreID,&sops,1);\
}
#define V(s) \
{\
  sops.sem_num = (s);\
  sops.sem_op  =  V_;\
  sops.sem_flg =  0; \
  semop(semaphoreID,&sops,1);\
}


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
    union semun semctl_arg;
    fprintf(stderr,"SemInit %d %d\n",id,value);
   
    semctl_arg.val = value;
    if(id >= sem_init || id<0 ) 
      fprintf(stderr,"attempt to intialize invalid semaphore %d %d\n",
                                                         id,sem_init);
    else if( semctl(semaphoreID, id,SETVAL,semctl_arg )<0){ 
         fprintf(stderr,"SemInit error\n");
         perror((char*)0);
    }
    fprintf(stderr,"exiting SemInit \n");
}


/*  release semaphore(s) */
void SemDel()
{
     semctl(semaphoreID,NULL,IPC_RMID,NULL);
}



/*\
 * (char *) CreateSharedRegion((long *) id, (long *) size)
 * long DetachSharedRegion((long) id, (long) size, (char *) addr)
 * long DeleteSharedRegion((long) id)
 * long DeleteSharedAll()
 * (char *) AttachSharedRegion((long) id, (long) size))
\*/

void Error( str, code)
     char *str;
     int code;
{
fprintf(stderr,"%s %d\n",str, code);
exit(0);
}

   /* Bizarre sequent has sysv semaphores but proprietary shmem */
   /* Encore has sysv shmem but is limited to total of 16384bytes! */
#if defined(SYSV)

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <errno.h>
#include <stdio.h>


#ifdef SUN
extern char *shmat();
#endif

char *CreateSharedRegion(id, size)
     long *size, *id;
{
  char *temp;

  /* Create the region */
  if ( (*id = shmget(IPC_PRIVATE, (int) *size, 
                     (int) (IPC_CREAT | 00600))) < 0 ){
    fprintf(stderr,"id=%d size=%d\n",*id, (int) *size);
    perror((char*)0);
    Error("CreateSharedRegion: failed to create shared region", (long) *id);
  }

  /* Attach to the region */
  if ( (temp = shmat((int) *id, (char *) NULL, 0)) == (char *) NULL){
    perror((char*)0);
    Error("CreateSharedRegion: failed to attach to shared region", (long) 0);
  }

  return temp;
}

long DetachSharedRegion( id, size, addr)
     long id, size;
     char *addr;
{
  return shmdt(addr);
}

long DeleteSharedRegion(id)
     long id;
{
  return shmctl((int) id, IPC_RMID, (struct shmid_ds *) NULL);
}

char *AttachSharedRegion(id, size)
     long id, size;
{
  char *temp;

  if ( (temp = shmat((int) id, (char *) NULL, 0)) == (char *) NULL)
    Error("AttachSharedRegion: failed to attach to shared region", (long) 0);

  return temp;
}

#endif

int main(int argc, char **argv)
{
int from=0, to, i;
    if(argc<2){
      printf("Usage:\n ipc.clean [<from>] <to> \n single argument is interpreted as <to> with <from> = 0 assumed\n");
      return 1;
    }
    if(argc=2) sscanf(argv[1],"%d",&to);
    else {
         sscanf(argv[1],"%d",&from);
         sscanf(argv[2],"%d",&to);
    }
    if(from>to && to <0){
       printf("wrong arguments\n");
       return 1;
    }
    for(i=from;i<=to;i++){ 
      semaphoreID =i;
      SemDel();
      DeleteSharedRegion((long)i);
    }

    return 0;
}
