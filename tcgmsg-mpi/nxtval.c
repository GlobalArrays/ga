#include "mpi.h"
#include "tcgmsgP.h"


#define LEN 2
long nxtval_counter=0;
#define INCR 1                 /* increment for NXTVAL */
#define BUSY -1L               /* indicates somebody else updating counter*/


void NextValueServer()
{
  Int  cnt     = 0;            /* actual counter */
  Int  ndone   = 0;            /* no. finished for this loop */
  int  type    = TYPE_NXTVAL;  /* message type */
  Int  buf[LEN];               /* buffer to get values */
  Int  mproc;                  /* no. of processes running loop */
  Int  nval;                   /* no. of values requested */
  int  done_list[MAX_PROCESS]; /* list of processes finished with this loop */
  int  lenmes, nodefrom;
  int  node;
  Int  ntermin=0;
  MPI_Status status;


  while (1) {

    /* Wait for input from any node */
    
    MPI_Recv(buf, LEN, TCG_INT, MPI_ANY_SOURCE, type, MPI_COMM_WORLD, &status); 
    MPI_Get_count(&status, TCG_INT, &lenmes);
    nodefrom = status.MPI_SOURCE;

    if (lenmes != LEN) {
      Error("NextValueServer: lenmes != LEN", (Int) lenmes);
      return;   /* Never actually gets here as does long jump */
    }

    mproc = buf[0];
    nval = buf[1];
    if (DEBUG_)
      (void) printf("NVS: from=%ld, mproc=%ld, ndone=%ld\n",
		    nodefrom, mproc, ndone);

    if (mproc == 0) {


      /* Sending process is about to terminate. Send reply and disable
       * sending to him. If all processes have finished return.
       *
       * All processes block on waiting for message
       * from nxtval server before terminating. nxtval only lets
       * everyone go when all have registered termination.
       */

      if (++ntermin == NNODES_()) {
        for (node=0; node<NNODES_(); node++) {
          MPI_Send(&cnt, 1, TCG_INT,  node, type, MPI_COMM_WORLD); 
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(0);
      }

    }
    else if (mproc > 0) {
      
      /* This is what we are here for */

      MPI_Send(&cnt, 1, TCG_INT,  nodefrom, type, MPI_COMM_WORLD); 
      cnt += nval;
    }
    else if (mproc < 0) {

      /* This process has finished the loop. Wait until all mproc
	 processes have finished before releasing it */

      done_list[ndone++] = nodefrom;

      if (ndone == -mproc) {
	while (ndone--) {
	  nodefrom = done_list[ndone];
          MPI_Send(&cnt, 1, TCG_INT,  nodefrom, type, MPI_COMM_WORLD); 
	}
	cnt = 0;
	ndone = 0;
      }
    }
  }
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
  Int  buf[2];
  MPI_Status status;
  int  type = TYPE_NXTVAL;

#  ifdef NXTVAL_SERVER
     int  server = (int)NNODES_();         /* id of server process */
#  else
     int  server = (int)NNODES_() -1;         /* id of server process */
#  endif

  if (SR_parallel) {
     buf[0] = *mproc;
     buf[1] = INCR;

     if (DEBUG_) {
       (void) printf("%2ld: nxtval: mproc=%ld\n",NODEID_(), *mproc);
       (void) fflush(stdout);
     }

#    ifdef NXTVAL_SERVER
       MPI_Send(buf, LEN, TCG_INT,  server, type, MPI_COMM_WORLD); 
       MPI_Recv(buf, 1,   TCG_INT,  server, type, MPI_COMM_WORLD, &status); 
       return buf[0];
#    endif
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

  return 0;
}

/*\ initialization for nxtval -- called in PBEGIN
\*/
void install_nxtval()
{
int numprocs, myid;

   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);

#  ifdef NXTVAL_SERVER
       /* in this mode one process is hidden from the application */
       if(SR_parallel && myid == numprocs -1) {
#        ifndef QUIET
           printf("TCGMSG-MPI info: excluding one process for nxtval server\n");
           fflush(stdout);
#        endif
         NextValueServer();
       }
#  else
        Do not know how to implement nxtval !
#  endif
}

void finalize_nxtval(){}
