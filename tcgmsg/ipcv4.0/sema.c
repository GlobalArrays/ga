/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/sema.c,v 1.3 1995-02-24 02:14:15 d3h325 Exp $ */

/*
  These routines simplify the interface to semaphores for use in mutual
  exclusion and queuing. Hopefully I can also make this portable.

  An external routine Error is assumed which is called upon an error
  and tidies up by calling SemSetDestroyAll.

  In most cases errors cause an internal hard failure (by calling Error).

  1) make an array of n_sem semaphores, returning the id associated
     with the entire set. All the semaphore values are initialized to value
     which should be a positve integer (queuing) or 0 (synchronization).
     The semaphores in the set are indexed from 0 to n_sem-1.

     long SemSetCreate(long n_sem, long value)

  2) Decrement and test the value associated with the semaphore specified by 
     (sem_set_id, sem_num). In effect this:

     if (value >= 0) {
        continue execution
     }
     else {
        wait in queue for the semaphore
     }
     decrement value

     void SemWait(long sem_set_id, long sem_num)

  3) Increment the value associated with the semaphore specified by
     (sem_set_id, sem_num). If value <= 0 (i.e. there are processes
     in the queue) this releases the next process.

     void SemPost(long sem_set_id, long sem_num)
     
  4) Return the current value associated with the semaphore sepcified by
     (sem_set_id, sem_num).

     long SemValue(long sem_set_id, long sem_num)

  5) Destroy the set of semaphores. Any other processes that are accessing
     or try to access the semaphore set should get an error.
     On the SUN (all system V machines?) the semaphore sets should
     be destroyed explicitly before the final process exits.
     0 is returned if OK. -1 implies an error.

     long SemSetDestroy(long sem_set_id)

  6) Destroy all the semaphore sets that are known about. This is really
     meant for an error routine to call to try and tidy up. Though all
     applications could call it before the last process exits.
     0 is returned if OK. -1 implies an error.

     long SemSetDestroyAll()
*/

extern void Error();

#ifdef SYSV

/********************************************************************
  Most system V compatible machines
 ********************************************************************/

/* 

   The value used for our semaphore is equal to the value of the
   System V semaphore (which is always positive) minus the no. of
   processes in the queue. That is because our interface was modelled
   after that of Alliant whose semaphore can take on negative values.
*/

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/sem.h>

#if defined(ARDENT) || defined(ENCORE) || defined(SEQUENT) || \
    defined(ULTRIX) || defined(AIX)    || defined(HPUX) || defined(KSR) || \
    defined(DECOSF)
union semun {
   long val;
   struct semid_ds *buf;
   ushort *array;
};
#endif


/* this global structure maintains a list of allocated semaphore sets
   which is used for SemSetDestroyAll */

#define MAX_SEM_SETS 20
static int sem_set_id_list[MAX_SEM_SETS];
static int num_sem_set = 0;

#define MAX_N_SEM 40

void InitSemSetList()
/* Initialise sem_set_id_list */
{
  int i;
  
  for (i=0; i<MAX_SEM_SETS; i++)
    sem_set_id_list[i] = -1;
}

long SemSetCreate(n_sem, value)
     long n_sem;
     long value;
{
  int semid, i;
  union semun arg;

  /* Check for errors and initialise data if first entry */

  if ( (n_sem <= 0) || (n_sem >= MAX_N_SEM) )
    Error("SemSetCreate: n_sem has invalid value", (long) n_sem);

  if (num_sem_set == 0)
    InitSemSetList();
  else if (num_sem_set >= MAX_SEM_SETS)
    Error("SemSetCreate: Exceeded man no. of semaphore sets",
          (long) num_sem_set);

  /* Actually make the semaphore set */

  if ( (semid = semget(IPC_PRIVATE, (int) n_sem, IPC_CREAT | 00600)) < 0)
    Error("SemSetCreate: failed to create semaphore set", (long) semid);

  /* Put the semid in the first empty slot in sem_set_id_list */

  for (i=0; i < MAX_SEM_SETS; i++) {
    if (sem_set_id_list[i] == -1) {
      sem_set_id_list[i] = semid;
      break;
    }
  }
  if (i == MAX_SEM_SETS)
    Error("SemSetCreate: internal error puting semid in list", (long) i);

  num_sem_set++;

  /* Now set the value of all the semaphores */

  arg.val = (int) value;
  for (i=0; i<n_sem; i++)
    if (semctl(semid, i, SETVAL, arg) == -1)
      Error("SemSetCreate: error setting value for semaphore", (long) i);

  return semid;
}

void SemWait(sem_set_id, sem_num)
     long sem_set_id;
     long sem_num;
{
  struct sembuf sops;

  sops.sem_num = sem_num;   /* semaphore no. */
  sops.sem_op = -1;         /* decrement by 1 */
  sops.sem_flg = 0;         /* block */

  if (semop((int) sem_set_id, &sops, 1) == -1)
    Error("SemWait: error from semop", (long) -1);
}

void SemPost(sem_set_id, sem_num)
     long sem_set_id;
     long sem_num;
{
  struct sembuf sops;

  sops.sem_num = sem_num;   /* semaphore no. */
  sops.sem_op =  1;         /* increment by 1 */
  sops.sem_flg = 0;         /* not used? */

  if (semop((int) sem_set_id, &sops, 1) == -1)
    Error("SemPost: error from semop", (long) -1);
}

long SemValue(sem_set_id, sem_num)
     long sem_set_id;
     long sem_num;
{
  /* See note at top of SUN code section about semaphore value */

  union semun arg;
  int semval, semncnt;
  
  if ( (semval = semctl((int) sem_set_id, (int) sem_num, GETVAL, arg)) == -1)
    Error("SemValue: error getting value for semaphore", (long) sem_num);
  
  if ( (semncnt = semctl((int) sem_set_id, (int) sem_num, GETNCNT, arg)) == -1)
    Error("SemValue: error getting ncnt for semaphore", (long) sem_num);
  
  return semval-semncnt;
}

long SemSetDestroy(sem_set_id)
     long sem_set_id;
{
  union semun arg;
  int i;

 /* Remove the sem_set_id from the internal list of ids */

  for (i=0; i<MAX_SEM_SETS; i++)
    if (sem_set_id_list[i] == sem_set_id) {
      sem_set_id_list[i] = -1;
      break;
    }

  num_sem_set--;

  /* System call to delete the id */
  
  return (long) semctl((int) sem_set_id, 0, IPC_RMID, arg);
}
  
long SemSetDestroyAll()
{
  long i, status=0;

  for (i=0; i<MAX_SEM_SETS; i++)
    if (sem_set_id_list[i] != -1)
      status += SemSetDestroy((long) sem_set_id_list[i]);

  if (status)
    status = -1;

  return status;
}

#endif

#ifdef ALLIANT
/*************************************************************
    Alliant Concentrix 5.0 and Concentrix FX/2800
 *************************************************************/

/* This is very specific to the Alliant. */

#include <sys/rtsem.h>
#include <sys/errno.h>

extern int errno;

/* On the alliant semaphores are handed out one at a time rather than
   in sets, so have to maintain sets manually */

#define MAX_SEM_SETS 20
#define MAX_N_SEM 128

static struct sem_set_list_struct {
  int id[MAX_N_SEM];                       /* alliant semaphore id */
  int n_sem;                               /* no. of semaphores in set */
} sem_set_list[MAX_SEM_SETS];

static int num_sem_set = 0;


void InitSemSetList()
/* Initialise sem_set_list */
{
  int i, j;
  
  for (i=0; i<MAX_SEM_SETS; i++) {
    sem_set_list[i].n_sem = 0;
    for (j=0; j<MAX_N_SEM; j++)
      sem_set_list[i].id[j] = -1;
  }
}

long SemSetCreate(n_sem, value)
     long n_sem;
     long value;
{
  int semid, i, j;

  /* Check for errors and initialise data if first entry */

  if ( (n_sem <= 0) || (n_sem >= MAX_N_SEM) )
    Error("SemSetCreate: n_sem has invalid value", (long) n_sem);

  if (num_sem_set == 0)
    InitSemSetList();
  else if (num_sem_set >= MAX_SEM_SETS)
    Error("SemSetCreate: Exceeded man no. of semaphore sets",
          (long) num_sem_set);

  /* Find first empty slot in sem_set_list */

  for (i=0; i < MAX_SEM_SETS; i++) 
    if (sem_set_list[i].n_sem == 0)
      break;

  if (i == MAX_SEM_SETS)
    Error("SemSetCreate: internal error puting semid in list", (long) i);

  /* Actually make the semaphore set */

  for (j=0; j<n_sem; j++) {
    if ( (semid = sem_create(value, value, SEMQUE_FIFO, 0)) < 0)
      Error("SemSetCreate: failed to create semaphore", (long) j);
    sem_set_list[i].id[j] = semid;
  }

  num_sem_set++;
  
  return i;
}

void SemWait(sem_set_id, sem_num)
     long sem_set_id;
     long sem_num;
{
interrupted:
  if (sem_wait(sem_set_list[sem_set_id].id[sem_num]) < 0) {
    if (errno == EINTR)
      goto interrupted;   /* got zapped by a signal ... try again */
    else
      Error("SemWait: error from sem_wait", (long) -1);
  }
}

void SemPost(sem_set_id, sem_num)
     long sem_set_id;
     long sem_num;
{
  if (sem_post(sem_set_list[sem_set_id].id[sem_num]) < 0)
    Error("SemPost: error from sem_post", (long) -1);
}

long SemValue(sem_set_id, sem_num)
     long sem_set_id;
     long sem_num;
{
  SEM_INFO info;

  if (sem_info(sem_set_list[sem_set_id].id[sem_num], &info, sizeof info) < 0)
    Error("SemValue: error from sem_info", (long) -1);

  return info.curval;
}

long SemSetDestroy(sem_set_id)
     long sem_set_id;
{
  int status=0, i;

  /* Close each semaphore in the set */

  for (i=0; i<sem_set_list[sem_set_id].n_sem; i++) {
    status += sem_destroy(sem_set_list[sem_set_id].id[i]);
    sem_set_list[sem_set_id].id[i] = -1;
  }

  sem_set_list[sem_set_id].n_sem = 0;

  num_sem_set--;

  if (status)
    status = -1;

  return (long) status;
}
  
long SemSetDestroyAll()
{
  int i, status=0;

  for (i=0; i<MAX_SEM_SETS; i++)
    if (sem_set_list[i].n_sem)
      status += SemSetDestroy(i);

  if (status)
    status = -1;

  return (long) status;
}

#endif
#if defined(CONVEX) || defined(APOLLO)

#include <stdio.h>
#include <sys/param.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/types.h>

#define MAX_SEM_SETS 20
#define MAX_N_SEM 100

/* On the convex a semaphore is a structure but on the apollo
   it is an array which does not need dereferencing.  Use ADDR
   to generate the address of a semaphore */
#ifdef APOLLO
#define ADDR(x) x
#else
#define ADDR(x) &x
#endif

extern char *mktemp();

struct sem_set_struct {
  int n_sem;                    /* no. of semaphores in set */
  semaphore lock[MAX_N_SEM];    /* locks for changing value */
  semaphore wait[MAX_N_SEM];    /* locks for queing */
  int value[MAX_N_SEM];         /* values */
};

static int num_sem_set = 0;
static struct sem_set_struct *sem_sets;
static int fd = -1;
static char template[] = "/tmp/SEMA.XXXXXX";
static char *filename = (char *) NULL;

void InitSemSets()
/* Initialise sem_sets and allocate associated shmem region */
{
  int i, j;
  unsigned size = sizeof(struct sem_set_struct) * MAX_SEM_SETS;

#ifndef APOLLO
  /* Generate scratch file to identify region ... mustn't do this
     on the APOLLO */

  filename = mktemp(template);
  if ( (fd = open(filename, O_RDWR|O_CREAT, 0666)) < 0 )
    Error("InitSemSets: failed to open temporary file",0);
#endif

  sem_sets = (struct sem_set_struct *) mmap((caddr_t) 0, &size,
                     PROT_READ|PROT_WRITE,
                     MAP_ANON|MAP_HASSEMAPHORE|MAP_SHARED, fd, 0);

#ifdef APOLLO
  if (sem_sets == (struct sem_set_struct *) 0)
    Error("InitSemSets: mmap failed", (long) -1);
#else
  if (sem_sets == (struct sem_set_struct *) -1)
    Error("InitSemSets: mmap failed", (long) -1);
#endif

  for (i=0; i<MAX_SEM_SETS; i++) {
    sem_sets[i].n_sem = 0;
    for (j=0; j<MAX_N_SEM; j++) {
      mclear(ADDR(sem_sets[i].lock[j]));
      mclear(ADDR(sem_sets[i].wait[j]));
      sem_sets[i].value[j] = 0;
    }
  }
}

long SemSetCreate(n_sem, value)
     long n_sem;
     long value;
{
  int i;

  /* Check for errors and initialise data if first entry */

  if ( (n_sem <= 0) || (n_sem >= MAX_N_SEM) )
    Error("SemSetCreate: n_sem has invalid value",n_sem);

  if (num_sem_set == 0)
    InitSemSets();
  else if (num_sem_set >= MAX_SEM_SETS)
    Error("SemSetCreate: Exceeded man no. of semaphore sets",
          num_sem_set);

  /* Initialize the values */

  for (i=0; i<n_sem; i++)
    sem_sets[num_sem_set].value[i] = value;

  sem_sets[num_sem_set].n_sem = n_sem;

  num_sem_set++;

  return (long) (num_sem_set - 1);
}

void SemWait(sem_set_id, sem_num)
     long sem_set_id;
     long sem_num;
{
  if ( (sem_set_id < 0) || (sem_set_id >= num_sem_set) )
    Error("SemWait: invalid sem_set_id",sem_set_id);
  if ( (sem_num < 0) || (sem_num >= sem_sets[sem_set_id].n_sem) )
    Error("SemWait: invalid semaphore number in set",sem_num);

  while (1) {

    /* Get the lock around the whole semaphore */

    (void) mset(ADDR(sem_sets[sem_set_id].lock[sem_num]), 1);

    /* If the value is positive fall thru, else wait */

    if (sem_sets[sem_set_id].value[sem_num] > 0)
      break;
    else {
      (void) mclear(ADDR(sem_sets[sem_set_id].lock[sem_num]));
      (void) mset(ADDR(sem_sets[sem_set_id].wait[sem_num]), 1);
    }
  }

  /* Are ready to go ... decrement the value and release lock */

  sem_sets[sem_set_id].value[sem_num]--;
  (void) mclear(ADDR(sem_sets[sem_set_id].lock[sem_num]));

}

void SemPost(sem_set_id, sem_num)
     long sem_set_id;
     long sem_num;
{
  int i;

  if ( (sem_set_id < 0) || (sem_set_id >= num_sem_set) )
    Error("SemPost: invalid sem_set_id",sem_set_id);
  if ( (sem_num < 0) || (sem_num >= sem_sets[sem_set_id].n_sem) )
    Error("SemPost: invalid semaphore number in set",sem_num);

  /* Get the lock around the whole semaphore */

  (void) mset(ADDR(sem_sets[sem_set_id].lock[sem_num]), 1);
  
  /* Read and increment the value. If is now zero wake up
     up the queue */

  sem_sets[sem_set_id].value[sem_num]++;
  i = sem_sets[sem_set_id].value[sem_num];

  (void) mclear(ADDR(sem_sets[sem_set_id].lock[sem_num]));
  if (i >= 0)
    (void) mclear(ADDR(sem_sets[sem_set_id].wait[sem_num]));
}

long SemValue(sem_set_id, sem_num)
     long sem_set_id;
     long sem_num;
{
  int i;

  if ( (sem_set_id < 0) || (sem_set_id >= num_sem_set) )
    Error("SemValue: invalid sem_set_id",sem_set_id);
  if ( (sem_num < 0) || (sem_num >= sem_sets[sem_set_id].n_sem) )
    Error("SemValue: invalid semaphore number in set",sem_num);

  /* There seems no point in getting the lock just to read
     the value and it seems more useful not to (e.g. debugging) */

  i = sem_sets[sem_set_id].value[sem_num];

  return (long) (i-1);
}

long SemSetDestroy(sem_set_id)
     long sem_set_id;
{

  if ( (sem_set_id < 0) || (sem_set_id >= num_sem_set) )
    return -1;

  sem_sets[sem_set_id].n_sem = 0;

  return (long) 0;
}
  
long SemSetDestroyAll()
{
  long i, status=0;

  for (i=0; i<num_sem_set; i++)
    if (sem_sets[i].n_sem)
      status += SemSetDestroy(i);

  if (fd >= 0) {
    (void) close(fd);
    fd = -1;
    (void) unlink(filename);
  }

  status += munmap((char *) sem_sets, 0);

  if (status)
    status = -1;

  return status;
}

#endif
