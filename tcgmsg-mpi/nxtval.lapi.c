#include <stdio.h>
#include <mpproto.h>
#include <pthread.h>
#include <stdio.h>
#include <lapi.h>
#include <mpi.h>

#include "tcgmsgP.h"

lapi_handle_t lapi_handle;
lapi_info_t   lapi_info;

int nxtval_counter=0;
int *nxtval_cnt_adr = &nxtval_counter;

#define LEN 2
#define INCR 1                 /* increment for NXTVAL */
#define BUSY -1L               /* indicates somebody else updating counter*/



/*\ initialize lapi
\*/
void install_nxtval()
{

     int myid, numtasks, rc;
     lapi_info.protocol = TB3_DEV;

     rc = LAPI_Init(&lapi_handle, &lapi_info);
     if(rc) Error("lapi_init failed",rc);
     
     rc=LAPI_Qenv(lapi_handle, TASK_ID, &myid);
     if(rc) Error("lapi_qenv failed",rc);
     rc=LAPI_Qenv(lapi_handle, NUM_TASKS, &numtasks);
     if(rc) Error("lapi_qenv failed 2",rc);

     if(myid>= numtasks || myid <0 || numtasks <0 || numtasks > 512)
            Error("lapi_initialize: invalid env",0); 

     /* disable LAPI internal error checking */
     LAPI_Senv(lapi_handle, ERROR_CHK, 0);

     /* broadcast nxtval counter address to everybody */
     MPI_Bcast(&nxtval_cnt_adr, sizeof(int*), MPI_BYTE, numtasks-1, MPI_COMM_WORLD);

#ifdef DEBUG
     printf("me=%d initialized %d processes\n", myid, numtasks);
#endif
     fflush(stdout);
     
}


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
#define INC 1
  long local;
  static lapi_cntr_t req_id=(lapi_cntr_t)0;
  static int val;
  int rc, inc = INC;

  int  server = (int)NNODES_() -1;         /* id of server process */

  if (server>0) { 
     /* parallel execution */
     if (DEBUG_) {
       (void) printf("%2ld: nxtval: mproc=%ld\n",NODEID_(), *mproc);
       (void) fflush(stdout);
     }

     if (*mproc < 0) {
           MPI_Barrier(MPI_COMM_WORLD);
           /* reset the counter value to zero */
           if( NODEID_() == server) nxtval_counter = 0;
           MPI_Barrier(MPI_COMM_WORLD);
     }
     if (*mproc > 0) {
           /* use atomic swap operation to increment nxtval counter */
           rc = LAPI_Rmw(lapi_handle, FETCH_AND_ADD, server, nxtval_cnt_adr,
                                      &inc, &local, &req_id);
           if(rc)Error("nxtval: rmw failed",rc);
           rc = LAPI_Waitcntr(lapi_handle, &req_id, 1, &val);
           if(rc)Error("nxtval: waitcntr failed",rc);
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

void finalize_nxtval(){};
