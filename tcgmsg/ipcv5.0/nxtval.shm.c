#include "tcgmsgP.h"


#define LEN 2
long nxtval_counter=0;
#define INCR 1                 /* increment for NXTVAL */
#define BUSY -1L               /* indicates somebody else updating counter*/
#define LOCK
#define UNLOCK


Integer NXTVAL_(mproc)
     Integer  *mproc;
/*
  Get next value of shared counter.

  mproc > 0 ... returns requested value
  mproc < 0 ... server blocks until abs(mproc) processes are queued
                and returns junk
  mproc = 0 ... indicates to server that I am about to terminate

*/
{
  long shmem_swap();
  long local;
  long sync_type= INTERNAL_SYNC_TYPE;

  int  server = (int)NNODES_() -1;         /* id of server process */

  if (NNODES_()==1) {
     if (DEBUG_) {
       (void) printf("%2ld: nxtval: mproc=%ld\n",NODEID_(), *mproc);
       (void) fflush(stdout);
     }

     if (*mproc < 0) {
           SYNCH_(&sync_type);
           /* reset the counter value to zero */
           if( NODEID_() == server) nxtval_counter = 0;
           SYNCH_(&sync_type);
     }
     if (*mproc > 0) {
/* Cray T3D/E
           while((local = shmem_swap(&nxtval_counter, BUSY, server)) == BUSY);
           shmem_swap(&nxtval_counter, (local+INCR), server);
*/
           LOCK;
             local = nxtval_counter;
             nxtval_counter += INCR;
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
      Error("nxtval: sequential version with silly mproc ", (Integer) *mproc);
  }

  return (Integer)local;
}

