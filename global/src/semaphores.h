#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

/* how many semaphores are available ? */
#ifndef SEMMSL
#   ifdef AIX
#         define SEMMSL 8094
#   else
#         define SEMMSL 12
#   endif
#endif

extern struct sembuf sops;
extern int semaphoreID;
int semop();
#define ALL_SEMS -1

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

void SemInit(), SemDel();
int  SemGet();

