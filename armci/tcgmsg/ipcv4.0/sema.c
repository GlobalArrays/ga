#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/sema.c,v 1.17 2003-05-08 15:44:43 edo Exp $ */

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

#if defined(SYSV) && !defined(SPPLOCKS) && !defined(MACX)

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

#if !HAVE_UNION_SEMUN
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

#if defined(SOLARIS) || defined (AIX) || defined(LINUX64)
#   define MAX_N_SEM 512 
#else
#   define MAX_N_SEM 40
#endif

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

#if defined(SPPLOCKS)

#include <stdio.h>
#include <unistd.h>  
#define MAX_SEMA 512
static volatile int *val;
#define NAME_LEN 200

#ifdef SGI
# include <ulocks.h>
  static usptr_t *arena_ptr;
  static ulock_t *locks[MAX_SEMA];
  static char arena_name[NAME_LEN];
# define EIGHT 8
# define LOCK ussetlock
# define UNLOCK usunsetlock
#define  JUMP EIGHT

#include "sndrcvP.h"

extern char *getenv(const char *);

long SemSetCreate(long n_sem, long value)
{
  int i;
  char *tmp;
  if (!(tmp = getenv("ARENA_DIR"))) tmp = "/tmp";

   sprintf(arena_name,"%s/tcgmsg.arena.%ld",tmp, (long)getpid()); 
#ifdef PRIVATE_ARENA
   (void) usconfig(CONF_ARENATYPE, US_SHAREDONLY);
#endif
   (void) usconfig(CONF_INITUSERS, (unsigned int)SR_clus_info[SR_clus_id].nslave );
#ifdef SGI
    (void) usconfig(CONF_INITSIZE, 1024*1024);
#endif

  if (!(arena_ptr = usinit(arena_name)))
    Error("SemSetCreate: failed to create arena", 0L);

  /* Magic factors of EIGHT here to ensure that values are
     in different cache lines to avoid aliasing */

  if (!(val = (int *) usmalloc(EIGHT*MAX_SEMA*sizeof(int), arena_ptr)))
    Error("SemSetCreate: failed to get shmem", (long) (MAX_SEMA*sizeof(int)));

  for (i=0; i<n_sem; i++) {
    if (!(locks[i] = usnewlock(arena_ptr)))
      Error("SemSetCreate: failed to create lock", (long) i);
    val[i*EIGHT] = (int) value;
  }
  return 1L;
}

long SemSetDestroyAll()
{
/*  usdetach (arena_ptr);*/
  arena_ptr = 0;
  unlink(arena_name);
  return 0;
}

#endif


#ifdef SPPLOCKS
#include <sys/param.h>
#include <sys/file.h>
#include <sys/cnx_mman.h>
#include <sys/mman.h>
#include <sys/types.h>

#define SIXTEEN 16
#define JUMP SIXTEEN
typedef struct{
        int state;
        int pad[15]; 
} lock_t;

static lock_t *locks;

#  define    LOCK(x) set_lock(&x.state)
#  define  UNLOCK(x) unset_lock(&x.state)
#  define INILOCK(x) init_lock(&x.state)


void init_lock(int * volatile ip)
{
    *ip = 1;
}

void set_lock(int * volatile ip)
{
    while (1) {
        while (!(*ip));
        if (__ldcws32(ip))
                        break;
    }
}

void unset_lock(int *ip)
{
  *ip = 1;
  asm("sync");
}

static int fd = -1;
static char template[] = "/tmp/SEMA.XXXXXX";
static char *filename = (char *) NULL;
static unsigned shmem_size;

long SemSetCreate(long n_sem, long value)
{
  int i;
  shmem_size = SIXTEEN*MAX_SEMA*sizeof(int)+MAX_SEMA*sizeof(lock_t);

  if ( (n_sem <= 0) || (n_sem >= MAX_SEMA) )
    Error("SemSetCreate: n_sem has invalid value",n_sem);

  /* allocate shared memory for locks and semaphore val */
  filename = mktemp(template);
  if ( (fd = open(filename, O_RDWR|O_CREAT, 0666)) < 0 )
    Error("SemSetCreate: failed to open temporary file",0);
  val = (int *) mmap((caddr_t) 0, shmem_size,
                     PROT_READ|PROT_WRITE,
                     MAP_ANONYMOUS|CNX_MAP_SEMAPHORE|MAP_SHARED, fd, 0);
  locks = (lock_t*)( val + SIXTEEN*MAX_SEMA);

  /* initialize locks and semaphore values */
  for (i=0; i<n_sem; i++) {
    INILOCK(locks[i]);
    val[i*SIXTEEN] = (int) value;
  }
  return 1L;
}

long SemSetDestroyAll()
{
  long status=0;
  if((int)unlink(filename)==-1)Error("SemSetDestroyAll: unlink failed",0);
  status = munmap((char *) shmem_size, 0);
  if(status)status = -1;
  return status;
}

#endif


double __tcgmsg_fred__=0.0;

void Dummy()
{
  int n = 200;			/* This seems optimal */
  while(n--)
    __tcgmsg_fred__++;
}

void SemWait(long sem_set_id, long sem_num)
{
  int value = 0;
  int off = sem_num*JUMP;

  if ( (sem_num < 0) || (sem_num >= MAX_SEMA) )
    Error("SemWait: invalid sem_num",sem_num);

  while (value<=0) {
    LOCK(locks[sem_num]);
    value = val[off];
    if (value>0)
      val[off]--;
    UNLOCK(locks[sem_num]);
    if (value<=0) 
      Dummy();
  }
}

void SemPost(long sem_set_id, long sem_num)
{
  int off = sem_num*JUMP;
  if ( (sem_num < 0) || (sem_num >= MAX_SEMA) )
    Error("SemPost: invalid sem_num",sem_num);

  LOCK(locks[sem_num]);
  val[off]++;
  UNLOCK(locks[sem_num]);
}

long SemValue(long sem_set_id, long sem_num)
{
  Error("SemValue: not implemented", sem_num);
  return 1;
}

long SemSetDestroy(long sem_set_id)
{
  return(SemSetDestroyAll());
}

#endif


#if defined(MACX)


#include <unistd.h>
#include <sys/fcntl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/semaphore.h>

#define MAX_SEMA 32
static int fd = -1;
static char template[] = "/tmp/SEMA.XXXXXX";
static char *filename = (char *) NULL;
static unsigned shmem_size;

#if defined(NAMED_SEMAPHORES_SUPPORTED)

static sem_t *sem_arr;

long SemSetCreate(long n_sem, long value)
{
  int i;
  shmem_size = MAX_SEMA*sizeof(sem_t);

  if ( (n_sem <= 0) || (n_sem >= MAX_SEMA) )
    Error("SemSetCreate: n_sem has invalid value",n_sem);

  /* allocate shared memory for locks and semaphore val */
  filename = mktemp(template);
  if ( (fd = shm_open(filename, O_CREAT|O_RDWR, 0666)) < 0 )
    Error("SemSetCreate: failed to open temporary shm file",0);
  sem_arr = (sem_t*) mmap((caddr_t)0, shmem_size, PROT_READ|PROT_WRITE,
                     MAP_ANON|MAP_HASSEMAPHORE|MAP_SHARED, fd, (off_t)0);
  if(!sem_arr)Error("SemSetCreate: failed to mmap",0);

  /* initialize locks and semaphore values */
  for (i=0; i<n_sem; i++) {
      if(sem_init(sem_arr+i,1,1)<0)
             Error("SemSetCreate: sem_init failed",(long)i);
  }
  return 1L;
}

long SemSetDestroyAll()
{
  long status=0;
  if((int)unlink(filename)==-1)Error("SemSetDestroyAll: unlink failed",0);
  status = munmap((char *) shmem_size, 0);
  if(status)status = -1;
  return status;
}


void SemWait(long sem_set_id, long sem_num)
{
  if ( (sem_num < 0) || (sem_num >= MAX_SEMA) )
    Error("SemWait: invalid sem_num",sem_num);
  if(sem_wait(sem_arr+sem_num)<0)
    Error("SemWait: failed",sem_num);
}

void SemPost(long sem_set_id, long sem_num)
{ 
  if ( (sem_num < 0) || (sem_num >= MAX_SEMA) )
    Error("SemPost: invalid sem_num",sem_num);
  if(sem_post(sem_arr+sem_num)<0)
    Error("SemPost: failed",sem_num);
 
}
  
long SemValue(long sem_set_id, long sem_num)  
{ 
  Error("SemValue: not implemented", sem_num);
  return 1L;
}

#else


typedef struct{
        int state;
        int pad[15];
} lock_t;
static lock_t *locks;


static char template1[] = "/tmp/SEMA1.XXXXXX";
static char *filename1 = (char *) NULL;
static sem_t *sem;
static lock_t *locks;

#include <stdio.h>

long SemSetCreate(long n_sem, long value)
{
  int i;
  shmem_size = MAX_SEMA*sizeof(lock_t);

  if ( (n_sem <= 0) || (n_sem >= MAX_SEMA) )
    Error("SemSetCreate: n_sem has invalid value",n_sem);

  /* allocate shared memory for locks and semaphore val */
  locks = (lock_t*) mmap((caddr_t)0, shmem_size, PROT_READ|PROT_WRITE,
                     MAP_ANON|MAP_SHARED, -1, (off_t)0);
  if(locks == (lock_t*)-1)Error("SemSetCreate: failed to mmap",shmem_size);

  filename1 = mktemp(template1);
  sem = sem_open(filename1, O_CREAT|O_EXCL, 0666, 1); 
  if(!sem)Error("SemSetCreate: failed to sem_open",0);
  
  /* initialize locks and semaphore values */
  bzero(locks,shmem_size);
  return 1L;
}

long SemSetDestroyAll()
{
  long status=0;
  status = munmap((char *) locks, shmem_size);
  if(status)status = -1;
  sem_unlink(filename1);
  return status;
}

double __tcgmsg_fred__=0.0;
void Dummy()
{
  int n = 200;                  /* This seems optimal */  
  while(n--) 
    __tcgmsg_fred__++;
}   
  
void SemWait(long sem_set_id, long sem_num)
{ 
  int value = 0, count=0;
  
  if ( (sem_num < 0) || (sem_num >= MAX_SEMA) )
    Error("SemWait: invalid sem_num",sem_num);
   
  while (value<=0) {
    if(sem_wait(sem)<0)Error("SemWait: sem_op error",sem_num);;
    value = locks[sem_num].state;
    if (value>0)
      locks[sem_num].state--;
    if(sem_post(sem)<0)Error("SemWait: sem_op error",sem_num);;
    if (value<=0) Dummy();
    count++;
    if(count%1000 == 999)usleep(1);
  }
} 
  
void SemPost(long sem_set_id, long sem_num)
{ 
  if ( (sem_num < 0) || (sem_num >= MAX_SEMA) )
    Error("SemPost: invalid sem_num",sem_num); 
    
  if(sem_wait(sem)<0)Error("SemPost: sem_op error",sem_num);;
      locks[sem_num].state++;
  if(sem_post(sem)<0)Error("SemWait: sem_op error",sem_num);;
} 

long SemValue(long sem_set_id, long sem_num)
{ 
  if ( (sem_num < 0) || (sem_num >= MAX_SEMA) )
    Error("SemVal: invalid sem_num",sem_num); 
  return (long)locks[sem_num].state;
}   
#endif  
#endif  
