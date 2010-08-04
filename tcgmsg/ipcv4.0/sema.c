#if HAVE_CONFIG_H
#   include "config.h"
#endif

/** @file
 * These routines simplify the interface to semaphores for use in mutual
 * exclusion and queuing. Hopefully I can also make this portable.
 *
 * An external routine Error is assumed which is called upon an error
 * and tidies up by calling SemSetDestroyAll.
 *
 * In most cases errors cause an internal hard failure (by calling Error).
 *
 * 1) make an array of n_sem semaphores, returning the id associated
 *    with the entire set. All the semaphore values are initialized to value
 *    which should be a positve integer (queuing) or 0 (synchronization).
 *    The semaphores in the set are indexed from 0 to n_sem-1.
 *
 *    Integer SemSetCreate(Integer n_sem, Integer value)
 *
 * 2) Decrement and test the value associated with the semaphore specified by 
 *    (sem_set_id, sem_num). In effect this:
 *
 *    if (value >= 0) {
 *       continue execution
 *    }
 *    else {
 *       wait in queue for the semaphore
 *    }
 *    decrement value
 *
 *    void SemWait(Integer sem_set_id, Integer sem_num)
 *
 * 3) Increment the value associated with the semaphore specified by
 *    (sem_set_id, sem_num). If value <= 0 (i.e. there are processes
 *    in the queue) this releases the next process.
 *
 *    void SemPost(Integer sem_set_id, Integer sem_num)
 *    
 * 4) Return the current value associated with the semaphore sepcified by
 *    (sem_set_id, sem_num).
 *
 *    Integer SemValue(Integer sem_set_id, Integer sem_num)
 *
 * 5) Destroy the set of semaphores. Any other processes that are accessing
 *    or try to access the semaphore set should get an error.
 *    On the SUN (all system V machines?) the semaphore sets should
 *    be destroyed explicitly before the final process exits.
 *    0 is returned if OK. -1 implies an error.
 *
 *    Integer SemSetDestroy(Integer sem_set_id)
 *
 * 6) Destroy all the semaphore sets that are known about. This is really
 *    meant for an error routine to call to try and tidy up. Though all
 *    applications could call it before the last process exits.
 *    0 is returned if OK. -1 implies an error.
 *
 *    Integer SemSetDestroyAll()
*/

extern void Error();

/********************************************************************
  Most system V compatible machines
 ********************************************************************/

/* 
   The value used for our semaphore is equal to the value of the
   System V semaphore (which is always positive) minus the no. of
   processes in the queue. That is because our interface was modelled
   after that of Alliant whose semaphore can take on negative values.
*/

#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif
#if HAVE_SYS_IPC_H
#   include <sys/ipc.h>
#endif
#if HAVE_SYS_SEM_H
#   include <sys/sem.h>
#endif

#include "typesf2c.h"

#ifndef HAVE_UNION_SEMUN
/* according to X/OPEN we have to define it ourselves */
union semun {
    int val;                    /* value for SETVAL */
    struct semid_ds *buf;       /* buffer for IPC_STAT, IPC_SET */
    unsigned short int *array;  /* array for GETALL, SETALL */
    struct seminfo *__buf;      /* buffer for IPC_INFO */
};
#endif

/* this global structure maintains a list of allocated semaphore sets
   which is used for SemSetDestroyAll */

#define MAX_SEM_SETS 20
static int sem_set_id_list[MAX_SEM_SETS];
static int num_sem_set = 0;

/* TODO Is there an autoconf test to determine this number? */
#define MAX_N_SEM 512 
/* #define MAX_N_SEM 40 */

/** Initialise sem_set_id_list */
void InitSemSetList()
{
    int i;

    for (i=0; i<MAX_SEM_SETS; i++) {
        sem_set_id_list[i] = -1;
    }
}


Integer SemSetCreate(Integer n_sem, Integer value)
{
    int semid, i;
    union semun arg;

    /* Check for errors and initialise data if first entry */

    if ( (n_sem <= 0) || (n_sem >= MAX_N_SEM) ) {
        Error("SemSetCreate: n_sem has invalid value", (Integer) n_sem);
    }

    if (num_sem_set == 0) {
        InitSemSetList();
    }
    else if (num_sem_set >= MAX_SEM_SETS) {
        Error("SemSetCreate: Exceeded man no. of semaphore sets",
                (Integer) num_sem_set);
    }

    /* Actually make the semaphore set */

    if ( (semid = semget(IPC_PRIVATE, (int) n_sem, IPC_CREAT | 00600)) < 0) {
        Error("SemSetCreate: failed to create semaphore set", (Integer) semid);
    }

    /* Put the semid in the first empty slot in sem_set_id_list */

    for (i=0; i < MAX_SEM_SETS; i++) {
        if (sem_set_id_list[i] == -1) {
            sem_set_id_list[i] = semid;
            break;
        }
    }
    if (i == MAX_SEM_SETS) {
        Error("SemSetCreate: internal error puting semid in list", (Integer) i);
    }

    num_sem_set++;

    /* Now set the value of all the semaphores */

    arg.val = (int) value;
    for (i=0; i<n_sem; i++){ 
        if (semctl(semid, i, SETVAL, arg) == -1){ 
            Error("SemSetCreate: error setting value for semaphore", (Integer) i);
        }
    }

    return semid;
}


void SemWait(Integer sem_set_id, Integer sem_num)
{
    struct sembuf sops;

    sops.sem_num = sem_num;   /* semaphore no. */
    sops.sem_op = -1;         /* decrement by 1 */
    sops.sem_flg = 0;         /* block */

    if (semop((int) sem_set_id, &sops, 1) == -1) {
        Error("SemWait: error from semop", (Integer) -1);
    }
}


void SemPost(Integer sem_set_id, Integer sem_num)
{
    struct sembuf sops;

    sops.sem_num = sem_num;   /* semaphore no. */
    sops.sem_op =  1;         /* increment by 1 */
    sops.sem_flg = 0;         /* not used? */

    if (semop((int) sem_set_id, &sops, 1) == -1) {
        Error("SemPost: error from semop", (Integer) -1);
    }
}


Integer SemValue(Integer sem_set_id, Integer sem_num)
{
    /* See note at top of SUN code section about semaphore value */

    union semun arg;
    int semval, semncnt;

    if ((semval = semctl((int) sem_set_id, (int) sem_num, GETVAL, arg)) == -1) {
        Error("SemValue: error getting value for semaphore", (Integer) sem_num);
    }

    if ((semncnt = semctl((int) sem_set_id, (int) sem_num, GETNCNT, arg)) == -1) {
        Error("SemValue: error getting ncnt for semaphore", (Integer) sem_num);
    }

    return semval-semncnt;
}


Integer SemSetDestroy(Integer sem_set_id)
{
    union semun arg;
    int i;

    /* Remove the sem_set_id from the internal list of ids */

    for (i=0; i<MAX_SEM_SETS; i++) {
        if (sem_set_id_list[i] == sem_set_id) {
            sem_set_id_list[i] = -1;
            break;
        }
    }

    num_sem_set--;

    /* System call to delete the id */

    return (Integer) semctl((int) sem_set_id, 0, IPC_RMID, arg);
}

  
Integer SemSetDestroyAll()
{
    Integer i, status=0;

    for (i=0; i<MAX_SEM_SETS; i++) {
        if (sem_set_id_list[i] != -1) {
            status += SemSetDestroy((Integer) sem_set_id_list[i]);
        }
    }

    if (status) {
        status = -1;
    }

    return status;
}
