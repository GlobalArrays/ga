/* Use Integerel NX hrecv to implement NXTVAL */ 

#include <stdio.h>
#include "sndrcv.h"


/* default is Paragon */
#ifdef DELTA
#  include <mesh.h>
#elif defined(IPSC)
#  include <cube.h>
#else
#  include <nx.h>
#endif


static long nxtval_buffer[2];    /* Used by handler for nxtval service */
static void nxtval_handler();
static long nxtval_server;
#define TYPE_NXTVAL 32768      /* Type of messages for next value    */
#define TYPE_NXTVAL_REPLY 32769

#define DEBUG_ DEBUG
extern long DEBUG=0;           /* debug flag ... see setdbg */



/*\ initialization for nxtval -- called in PBEGIN
\*/
void install_nxtval()
{
  /* Register the handler for NXTVAL service */

  nxtval_server = numnodes() - 1;

  if (mynode() == nxtval_server)
    hrecv(TYPE_NXTVAL, nxtval_buffer, sizeof nxtval_buffer, nxtval_handler);

  masktrap(0);  /* Ensure trap is enabled */

  gsync();
}


Integer NXTVAL_(mproc)
     long *mproc;
/*
  Get next value of shared counter.

  mproc > 0 ... returns requested value
  mproc < 0 ... server blocks until abs(mproc) processes are queued
                and returns junk
  mproc = 0 ... indicates to server that I am about to terminate

  this needs to be extended so that clusters of processes with
  shared memory collectively get a bunch of values from the server
  thus reducing the overhead of calling nextvalue.
*/
{
  long buf[2];
  long lenbuf = sizeof buf;
  long lenmes, nodefrom;
  long sync = 1;
  long msgid;

  buf[0] = *mproc;
  buf[1] = 1;

  if (DEBUG_) {
    (void) printf("nxtval: me=%d, mproc=%d\n",mynode(), *mproc);
    (void) fflush(stdout);
  }

  msgid = irecv(TYPE_NXTVAL_REPLY, (char *) buf, lenbuf);
  csend(TYPE_NXTVAL, (char *) buf, lenbuf, nxtval_server, 0);
  msgwait(msgid);

  return buf[0];
}


static void nxtval_handler(msgtype, msglen, requesting_node, pid)
      long msgtype, msglen, requesting_node, pid;
{
  long oldmask = masktrap(1);
  static long cnt     = 0;     /* actual counter */
  static long ndone   = 0;     /* no. finished for this loop */
  static long done_list[4096];  /* list of processes finished with this loop */
  long lencnt = sizeof cnt;    /* length of cnt */
  long node   = -1;            /* select any node */
  long type   = TYPE_NXTVAL_REPLY;   /* message type */
  long mproc;                  /* no. of processes running loop */
  long nval;                   /* no. of values requested */
  long sync = 1;               /* all info goes synchronously */
  long lenbuf = sizeof nxtval_buffer;    /* length of buffer */

  if (msglen != lenbuf) 
    Error("NextValueServer: lenmsg != lenbuf", msglen);

  mproc = nxtval_buffer[0];
  nval  = nxtval_buffer[1];
  if (DEBUG_) {
    (void) printf("NVS: from=%d, mproc=%d, ndone=%d\n",
                  requesting_node, mproc, ndone);
  }

  if (mproc == 0)
    Error("NVS: invalid mproc ", mproc);
  else if (mproc > 0) {
      
    /* This is what we are here for */

    csend(type, (char *) &cnt, sizeof cnt, requesting_node, 0);
    cnt += nval;
  }
  else if (mproc < 0) {

    /* This process has finished the loop. Wait until all mproc
       processes have finished before releasing it */

    done_list[ndone++] = requesting_node;

    if (ndone == -mproc) {
      while (ndone--) {
        long nodeto = done_list[ndone];
        csend(type, (char *) &cnt, sizeof cnt, nodeto, 0);
      }
      cnt = 0;
      ndone = 0;
    }
  }
  hrecv(TYPE_NXTVAL, nxtval_buffer, sizeof nxtval_buffer, nxtval_handler);
  oldmask = masktrap(oldmask);
}

void finalize_nxtval(){};
