#include <mpi.h>
#include "tcgmsgP.h"
#include "armci.h"


#define LEN 2
static long *pnxtval_counter;
#define INCR 1                 /* increment for NXTVAL */
#define BUSY -1L               /* indicates somebody else updating counter*/
#define NXTV_SERVER ((int)NNODES_() -1)

long NXTVAL_(mproc)
     long  *mproc;
/*
  Get next value of shared counter.

  mproc > 0 ... returns requested value
  mproc < 0 ... server blocks until abs(mproc) processes are queued
                and returns junk
  mproc = 0 ... indicates to server that I am about to terminate

*/
{
  long local;
  int rc;

  int  server = NXTV_SERVER;         /* id of server process */

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

       rc = ARMCI_Rmw(ARMCI_FETCH_AND_ADD_LONG,(int*)&local,(int*)pnxtval_counter,1,server);
     
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
      Error("nxtval: sequential version with silly mproc ", (long) *mproc);
  }

  return local;
}

/*\ initialization for nxtval -- called in PBEGIN
\*/
void install_nxtval()
{
   int rc;
   int me = (int)NODEID_(), bytes, server;

   void *ptr_ar[MAX_PROCESS];

   rc = ARMCI_Init();
   if(rc)Error("nxtv: armci_init failed",rc);
   server = NXTV_SERVER;

   if(me== server) bytes = sizeof(long);
   else bytes =0;

   rc = ARMCI_Malloc(ptr_ar,bytes);
   if(rc)Error("nxtv: armci_malloc failed",rc);

   pnxtval_counter = (long*) ptr_ar[server];
   if(me==server)*pnxtval_counter = (long)0;
    
   rc=MPI_Barrier(MPI_COMM_WORLD); 
   if(rc!=MPI_SUCCESS)Error("init_nxtval: barrier failed",0);
}


void finalize_nxtval()
{
    if(NODEID_() == NXTV_SERVER)ARMCI_Free(pnxtval_counter);
    ARMCI_Finalize();
}
