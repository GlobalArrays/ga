#include <mpi.h>
#include <mpp/shmem.h>
#include "tcgmsgP.h"

#define NXTVAL_GUARD 63
#define LEN 2
long nxtval_counter=0;
#define INCR 1                 /* increment for NXTVAL */
#define BUSY -1L               /* indicates somebody else updating counter*/

/* on j90 shmem barrier appaers to be broken */
#if defined(CRAY_T3D) || defined(_CRAYMPP)
#define SYNC barrier()
#else
#define SYNC MPI_Barrier(MPI_COMM_WORLD)
#endif



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
  long local;

  int  server = (int)NNODES_() -1;         /* id of server process */

  if (SR_parallel) {
     if (DEBUG_) {
       (void) printf("%2ld: nxtval: mproc=%ld\n",NODEID_(), *mproc);
       (void) fflush(stdout);
     }

     if (*mproc < 0) {
           SYNC;
           /* reset the counter value to zero */
           if( NODEID_() == server) nxtval_counter = 0;
           SYNC;
     }
     if (*mproc > 0) {

#       if defined(CRAY_T3D) || defined(_CRAYMPP)

           /* use atomic swap operation to increment nxtval counter */
           while((local = shmem_swap(&nxtval_counter, BUSY, server)) == BUSY);
           shmem_swap(&nxtval_counter, (local+INCR), server);

#       else

           /* only a subset of shemem available */
#          pragma _CRI guard NXTVAL_GUARD 
           shmem_get(&local,&nxtval_counter,1,0);
           local +=INCR;
           shmem_put(&nxtval_counter,&local,1,0);
           shmem_quiet();
#          pragma _CRI endguard NXTVAL_GUARD

#       endif

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

/*\ initialization for nxtval -- called in PBEGIN
\*/
void install_nxtval()
{
}

void finalize_nxtval(){};
