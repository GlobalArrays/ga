#ifndef _SEMAPHORES_H_
#define _SEMAPHORES_H_

#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif
#if HAVE_SYS_IPC_H
#   include <sys/ipc.h>
#endif
#if HAVE_SYS_SEM_H
#   include <sys/sem.h>
#endif

#if !HAVE_UNION_SEMUN
union semun {
        int val;                    /* value for SETVAL */
        struct semid_ds *buf;       /* buffer for IPC_STAT, IPC_SET */
        unsigned short int *array;  /* array for GETALL, SETALL */
        struct seminfo *__buf;      /* buffer for IPC_INFO */
};
#endif

/* how many semaphores are available ? */
#ifndef SEMMSL
#   ifdef AIX
#         define SEMMSL 8094
#   else
#         define SEMMSL 16
#   endif
#endif

extern struct sembuf sops;
extern int semaphoreID;
int semop();
#define ALL_SEMS -1

#define _P_code      -1
#define _V_code       1
#define P_semaphore(s)  \
{\
  sops.sem_num = (s);\
  sops.sem_op  =  _P_code;\
  sops.sem_flg =  0; \
  semop(semaphoreID,&sops,1);\
}
#define V_semaphore(s) \
{\
  sops.sem_num = (s);\
  sops.sem_op  =  _V_code;\
  sops.sem_flg =  0; \
  semop(semaphoreID,&sops,1);\
}

typedef int lockset_t;

#endif
