#include "mpi.h"
#include "tcgmsgP.h"

#ifdef SGIUS
#include        <ulocks.h>
#include        <malloc.h>
ulock_t *nxtval_lock;
static usptr_t *arena_ptr;
static char arena_name[256];
typedef struct { 
        long pid;
        void *lptr;
        void *cptr;
}arena_t;
arena_t arenaid;

#define LOCK   ussetlock(nxtval_lock)
#define UNLOCK usunsetlock(nxtval_lock)
        
#endif


#define LEN 2
static long *pnxtval_counter;
#define INCR 1                 /* increment for NXTVAL */
#define BUSY -1L               /* indicates somebody else updating counter*/


arena_t ArenaCreate()
{
  arena_t id;
  unsigned int nprocs;

  id.pid = (long)getpid();
  nprocs = (unsigned int)NNODES_();
  sprintf(arena_name,"/tmp/nxtval.arena.%ld",id.pid);

  (void)usconfig(CONF_ARENATYPE,US_GENERAL);
  (void)usconfig(CONF_AUTOGROW,(unsigned int)1);
  (void)usconfig(CONF_INITUSERS,nprocs);
  arena_ptr = usinit(arena_name);
  if(!arena_ptr) Error("Failed to create nxtval arena lock",0);

  nxtval_lock = (ulock_t *)usnewlock(arena_ptr);
  if(!nxtval_lock) Error("Failed to create lock",0);
  id.lptr = nxtval_lock;

  pnxtval_counter = (long*) usmalloc (sizeof(long),arena_ptr); 
  if(!pnxtval_counter) Error("Failed to create shmem",0);
  id.cptr = pnxtval_counter;

  return id;

}


void ArenaAttach(arena_t id)
{

  sprintf(arena_name,"/tmp/nxtval.arena.%ld",id.pid);
  (void)usconfig(CONF_ARENATYPE,US_GENERAL);
  arena_ptr = usinit(arena_name);
  if(!arena_ptr) Error("Failed to create nxtval arena lock",0);
 
  nxtval_lock = (ulock_t*)id.lptr;
  if(!nxtval_lock) Error("Failed to create lock",0);

  pnxtval_counter = (long*)id.cptr;
  if(!pnxtval_counter) Error("Failed to create shmem",0);

}



Int NXTVAL_(mproc)
     Int  *mproc;
/*
  Get next value of shared counter.

  mproc > 0 ... returns requested value
  mproc < 0 ... server blocks until abs(mproc) processes are queued
                and returns junk
  mproc = 0 ... indicates to server that I am about to terminate

*/
{
  long local, old;
  int rc;

  int  server = (int)NNODES_() -1;         /* id of server process */

  if (SR_parallel) {
     if (DEBUG_) {
       (void) printf("%2ld: nxtval: mproc=%ld\n",NODEID_(), *mproc);
       (void) fflush(stdout);
     }

     if (*mproc < 0) {
           rc=MPI_Barrier(MPI_COMM_WORLD); 
           if(rc!=MPI_SUCCESS)Error("nxtval: barrier failed",0);

           /* reset the counter value to zero */
           if( NODEID_() == server) *pnxtval_counter = 0;

           rc=MPI_Barrier(MPI_COMM_WORLD); 
           if(rc!=MPI_SUCCESS)Error("nxtval: barrier failed",0);
     }
     if (*mproc > 0) {

           LOCK;

             local = *pnxtval_counter;
             *pnxtval_counter +=INCR; 

           UNLOCK;

     }
   } else {
     /* Not running in parallel ... just do a simulation */
     static int count = 0;
     if (*mproc == 1)
       return count++;
     else if (*mproc == -1) {
       count = 0;
      return 0;
    }
    else
      Error("nxtval: sequential version with silly mproc ", (Int) *mproc);
  }

  return (Int)local;
}

/*\ initialization for nxtval -- called in PBEGIN
\*/
void install_nxtval()
{
   int rc;
   int me = (int)NODEID_(), root=0;

   if(me==root){
        arenaid = ArenaCreate();
        *pnxtval_counter = (long)0;
   } 

   rc  = MPI_Bcast(&arenaid,sizeof(arena_t),MPI_BYTE,root,MPI_COMM_WORLD);
   if(rc!=MPI_SUCCESS)Error("install_nxtval:broadcast failed",0);

   if(me!=root)ArenaAttach(arenaid);
 
   if(DEBUG_)
   fprintf(stderr,"%d arena struct %d lptr=%x cptr=%x\n", me, arenaid.pid,
           arenaid.lptr, arenaid.cptr);

   rc=MPI_Barrier(MPI_COMM_WORLD); 
   if(rc!=MPI_SUCCESS)Error("init_nxtval: barrier failed",0);
}


void finalize_nxtval()
{
   usdetach(arena_ptr);
   arena_ptr = NULL;
   (void)unlink(arena_name);
}
