#include <mpi.h>
#include "tcgmsgP.h"
#include "../config/fujitsu-vpp.h"

#define BARRIER_BROKEN 1

#ifdef BARRIER_BROKEN
#  undef NATIVE_BARRIER
#  define  NATIVE_BARRIER MPI_Barrier(MPI_COMM_WORLD)
#endif

#define LEN 2
long nxtval_counter=0;
long *pnxtval = &nxtval_counter;
#define INCR 1                 /* increment for NXTVAL */
#define BUSY -1L               /* indicates somebody else updating counter*/



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
  long oldval;

  long  server = NNODES_() -1;         /* id of server process */

  if (SR_parallel) {
     if (DEBUG_) {
       (void) printf("%2ld: nxtval: mproc=%ld\n",NODEID_(), *mproc);
       (void) fflush(stdout);
     }

     if (*mproc < 0) {
           NATIVE_BARRIER;
           /* reset the counter value to zero */
           if( NODEID_() == server) nxtval_counter = 0;
           NATIVE_BARRIER;
     }
     if (*mproc > 0) {
        long newval;

        NATIVE_LOCK(server,NXTV_SEM);
           CopyElemFrom(pnxtval,&oldval,1,server);
           newval = oldval +INCR;
           CopyElemTo(&newval, pnxtval, 1, server);
        NATIVE_UNLOCK(server,NXTV_SEM);
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

  return (Integer)oldval;
}

/*\ initialization for nxtval -- called in PBEGIN
\*/
void install_nxtval()
{
}

void finalize_nxtval(){};

