#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

static struct sembuf sops;
static int semaphoreID;
int semop();

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

