/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/ipsc.c,v 1.6 1995-02-24 02:17:22 d3h325 Exp $ */

/*
   Toolkit interface for the iPSC-2, i860, DELTA and Paragon
*/

#include <stdio.h>

#include "sndrcv.h"

#ifdef EVENTLOG
#include "evlog.h"
#endif

extern char* malloc();

/* Declarations of routines used from the iPSC communications library */

#ifdef DELTA
#include <mesh.h>
#elif defined(PARAGON)
#include <nx.h>
#else
#include <cube.h>
#endif

extern char *memalign();

static long nxtval_buffer[2];    /* Used by handler for nxtval service */
static void nxtval_handler();
static long nxtval_server;
#define TYPE_NXTVAL 32768      /* Type of messages for next value    */
#define TYPE_NXTVAL_REPLY 32769

/*
  So that can easily wait for message of specific type and
  node the sender packs the user type and requested node
  to form an internal type with MAKETYPE. GETTYPE and GETNODE
  extract the user type and originating node, respectively.
*/

#define MAKETYPE(type, node)	((long) ((type) | ((node) << 17)))
#define GETTYPE(ttype)		((long) ((ttype) & 0xffff))
#define GETNODE(ttype)		((long) (((ttype) >> 17) & 0xffff))

/* Macros for checking node and type - true if out of range */

#define CHKNODE(node) ( ((node) < 0) || ((node) > NNODES_()) )
#define CHKTYPE(type) ( ((type) < 1) || ((type) > 32767) )


/* Global variables */

#define DEBUG_ DEBUG
static long DEBUG=0;           /* debug flag ... see setdbg */

#define MAX_Q_LEN 64         /* Maximum no. of outstanding messages */

static long n_in_msg_q = 0;    /* No. in the message q */

static struct msg_q_struct{
  long   msg_id;
  long   node;
  long   type;
  long   lenbuf;
  long   snd;
  long   from;
} msg_q[MAX_Q_LEN];

/***********************************************************/

long NODEID_()
/*
  Return number of the calling process ... at the moment this is
  just the same as the Intel numbering
*/
{
  return mynode();
}

void Error(string, code)
     char *string;
     long code;
/*
  Error handler
*/
{
  (void) fflush(stdout);
  (void) fflush(stderr);

  (void) fprintf(stderr, "%3d:%s %ld(%x)\n", NODEID_(), string, code, code);
  (void) perror("system message");

  (void) fflush(stdout);
  (void) fflush(stderr);

#if defined(DELTA)
  killproc((long) -1, (long) 0);
#elif defined(PARAGON)
  kill(0L,9L);
#else
  killcube((long) -1, (long) -1);
#endif
}

long NNODES_()
/*
  Return number of USER processes.
*/
{
  return numnodes();
}

static long firsttime=0;    /* Reference for timer */

static void MtimeReset()    /* Sets timer reference */
{
  firsttime = (long) mclock();
}

long MTIME_()
/*
  Return elapsed time in CENTI seconds from some abitrary origin
*/
{
  static long firstcall = 1;

  if (firstcall) {
    firstcall = 0;
    MtimeReset();
  }
    
  return (long) ((mclock() - firsttime) / 10);
}

void SND_(type, buf, lenbuf, node, sync)
     long *type;
     char *buf;
     long *lenbuf;
     long *node;
     long *sync;
/*
  long *type     = user defined integer message type (input)
  char *buf      = data buffer (input)
  long *lenbuf   = length of buffer in bytes (input)
  long *node     = node to send to (input)
  long *sync     = flag for sync(1) or async(0) communication (input)
*/
{
  long me = NODEID_();
  long ttype = MAKETYPE(*type, me);

  if (CHKTYPE(*type))
    Error("SND_: invalid type specified",*type);

  if (CHKNODE(*node))
    Error("SND_: invalid node specified",*node);

  if (DEBUG) {
    (void) printf("SND_: node %ld sending to %ld, len=%ld, type=%ld, sync=%ld\n",
		  me, *node, *lenbuf, *type, *sync);
    (void) fflush(stdout);
  }

#ifdef EVENTLOG
  evlog(EVKEY_BEGIN,     EVENT_SND,
	EVKEY_MSG_LEN,  *lenbuf,
	EVKEY_MSG_FROM,  me,
	EVKEY_MSG_TO,   *node,
	EVKEY_MSG_TYPE, *type,
	EVKEY_MSG_SYNC, *sync,
	EVKEY_LAST_ARG);
#endif

  if (*sync)
    csend(ttype, buf, *lenbuf, *node, 0);
  else {
    if (n_in_msg_q >= MAX_Q_LEN)
      Error("SND: overflowing async Q limit", n_in_msg_q);

    msg_q[n_in_msg_q].msg_id = isend(ttype, buf, *lenbuf, *node, 0);
    msg_q[n_in_msg_q].snd = 1;
    if (DEBUG) {
      (void) printf("SND: me=%ld, to=%ld, len=%ld, msg_id=%ld, ninq=%ld\n",
		    me, *node, *lenbuf, msg_q[n_in_msg_q].msg_id,
		    n_in_msg_q);
      (void) fflush(stdout);
    }
    n_in_msg_q++;
  }
#ifdef EVENTLOG
  evlog(EVKEY_END, EVENT_SND, EVKEY_LAST_ARG);
#endif
}

void RCV_(type, buf, lenbuf, lenmes, nodeselect, nodefrom, sync)
     long *type;
     char *buf;
     long *lenbuf;
     long *lenmes;
     long *nodeselect;
     long *nodefrom;
     long *sync;
/*
  long *type        = user defined type of received message (input)
  char *buf         = data buffer (output)
  long *lenbuf      = length of buffer in bytes (input)
  long *lenmes      = length of received message in bytes (output)
                      (exceeding receive buffer is hard error)
  long *nodeselect  = node to receive from (input)
                      -1 implies that any pending message of the specified
                      type may be received
  long *nodefrom    = node message is received from (output)
  long *sync        = flag for sync(1) or async(0) receipt (input)
*/
{
  long me = NODEID_();
  long ttype;
  static long node = 0;		/* For fairness in rcv from anyone */
  
  if (CHKTYPE(*type))
    Error("RCV_: invalid type specified",*type);
  
  if (*nodeselect != -1) {
    if (CHKNODE(*nodeselect))
      Error("RCV_: invalid node specified",*nodeselect);
    
    ttype = MAKETYPE(*type, *nodeselect); 
  }
  else {
    ttype = -1;                    /* Any node, check type later */
  }
  
  if (DEBUG) {
    (void) printf("RCV_: node %ld receiving from %ld, len=%ld, type=%ld, sync=%ld\n",
		  me, *nodeselect, *lenbuf, *type, *sync);
    (void) fflush(stdout);
  }
  
#ifdef EVENTLOG
  evlog(EVKEY_BEGIN,     EVENT_RCV,
	EVKEY_MSG_FROM, *nodeselect,
	EVKEY_MSG_TO,    me,
	EVKEY_MSG_TYPE, *type,
	EVKEY_MSG_SYNC, *sync,
	EVKEY_LAST_ARG);
#endif
  
  if (*sync) {
    if (*nodeselect == -1) {
      /* Receive from anyone. To do this efficiently and preserve the
	 type matching first wait for a message of any type to have arrived.
	 Then if it is of the correct type we are in business. Otherwise we
	 have an O(P) process where we loop thru all processors looking
	 for messages ... ugh. */
      
      cprobe((long) -1);	/* Yuck ... NX makes no attempt at fairness 
				   ... it always searches starting at node 0 
				   ... gotta live with it or accept an O(P) cost
				   even if only one message is pending */
      ttype = infotype();
      if (GETTYPE(ttype) != *type) {
	long nn = NNODES_();
	
	while (1) {
	  ttype = MAKETYPE(*type, node);
	  if (iprobe(ttype))
	    break;
	  else {
	    node = (node + 1) % nn;
	    if (node == 0)
	      {flick(); flick(); flick();}
	  }
	}
      }
    }
    
    crecv(ttype, buf, *lenbuf);
    *nodefrom = infonode();          /* Get source node  */
    ttype = infotype();              /* Get type */
    *lenmes = infocount();           /* Get length */
    
    if (*lenmes > *lenbuf)
      Error("RCV_: out of range length on received message",*lenmes);
    
    if (GETTYPE(ttype) != *type)
      Error("RCV_: type mismatch for received message",GETTYPE(ttype));
    
    if (GETNODE(ttype) != *nodefrom)
      Error("RCV_: mismatch of nodefrom and node packed in received type",
	    GETNODE(ttype));
    
    if (*nodeselect != -1 && *nodefrom != *nodeselect)
	Error("RCV_: received message from wrong node!",*nodefrom);
  }
  else {

    /* Note that essentially NO checking can be done here
       as the info is just not available. Set the unknown
       return values to invalid values. Checking is done later
       when the asynch I/O completes */

    /* !!!! Note that that asynchronous receive from anyone breaks unless an
       explicit syncronization is done if there is any possibility of
       a message of a different type arriving.  The synchronous code
       does not suffer this problem !!! */

    if (n_in_msg_q >= MAX_Q_LEN)
      Error("RCV: overflowing async Q limit", n_in_msg_q);

    msg_q[n_in_msg_q].msg_id = irecv(ttype, buf, *lenbuf);
    msg_q[n_in_msg_q].node   = *nodeselect;
    msg_q[n_in_msg_q].type   = *type;
    msg_q[n_in_msg_q].lenbuf = *lenbuf;
    msg_q[n_in_msg_q].snd = 0;
    if (DEBUG) {
      (void) printf("RCV: me=%ld, from=%ld, len=%ld, msg_id=%ld, ninq=%ld\n",
		    me, *nodeselect, *lenbuf, msg_q[n_in_msg_q].msg_id,
		    n_in_msg_q);
      (void) fflush(stdout);
    }
    *lenmes = (long) -1;
    n_in_msg_q++;
    *nodefrom = (long) *nodeselect;
  }
#ifdef EVENTLOG
  evlog(EVKEY_END, EVENT_RCV,
	EVKEY_MSG_FROM, *nodefrom,
	EVKEY_MSG_LEN, *lenmes,
	EVKEY_LAST_ARG);
#endif
}

void PBEGINF_()
{
  PBEGIN_();
}

long PROBE_(long *type, long *node)
{
  long ttype;
  
  if (*node >= 0) {
    /* Receive from specific node ... simple map to iprobe */
    
    ttype = MAKETYPE(*type, *node);
    return iprobe(ttype);
  }
  else if (iprobe(-1L)) {
    /* Receive from anyone ... find the first available message ... if that
       is not OK then manually examine everything */
    
    ttype = infotype();
    if (GETTYPE(ttype) == *type) {
      return 1;
    }
    else {
      long nn = NNODES_();
      long me = NODEID_();
      long p;
      
      for (p=0; p<nn; p++) {
	if (p != me) {
	  ttype = MAKETYPE(*type, p);
	  if (iprobe(ttype))
	    return 1;
	}
      }
      return 0;
    }
  }
}

void PBEGIN_()
{
  char workdir[256], *eventfile;
  long start = MTIME_();
  long type = 1;
  DEBUG = 0;

  if (DEBUG) {
    (void) printf("node %ld called pbeginf\n",NODEID_());
    (void) fflush(stdout);
  }

#if !(defined(DELTA) || defined(PARAGON))
  led((long) 1);		/* Green LED on */

  /* recv work directory from host */

  if(_crecv(2, workdir, 256))
    Error("PBEGIN: error receving workdir", (long) -1);
  if(chdir(workdir) != 0)
    Error("PBEGIN: failed to switch to work directory", (long) -1);

  if (DEBUG) {
    (void) printf("node=%ld, nproc=%ld, workdir=%s\n",
		  NODEID_(), NNODES_(), workdir);
    (void) fflush(stdout);
  }
#endif

  /* Register the handler for NXTVAL service */

  nxtval_server = NNODES_() - 1;

  if (mynode() == nxtval_server)
    hrecv(TYPE_NXTVAL, nxtval_buffer, sizeof nxtval_buffer, nxtval_handler);

  /* Synchronize processes and zero all timers on return to user code */

  SYNCH_(&type);
  start = MTIME_() - start;
  MtimeReset();

  /* If logging events make the file events.<nodeid> */

#ifdef EVENTLOG
  if (eventfile=malloc((unsigned) 11)) {
    (void) sprintf(eventfile, "events.%03ld", NODEID_());
    evlog(EVKEY_ENABLE, EVKEY_FILENAME, eventfile,
	  EVKEY_BEGIN, EVENT_PROCESS,
	  EVKEY_STR_INT, "Startup used (cs)", start,
	  EVKEY_STR_INT, "No. of processes", NNODES_(),
	  EVKEY_DISABLE,
	  EVKEY_LAST_ARG);
    (void) free(eventfile);
  }
#endif

  masktrap(0);  /* Ensure trap is enabled */

  SYNCH_(&type);
}

void PEND_()
/*
  Zero effect for ipsc version ... for flash switch off green LED.
*/
{
#ifdef EVENTLOG
  long start=MTIME_();
#endif

  if (DEBUG) {
    (void) printf("node %ld called pend\n",NODEID_());
    (void) fflush(stdout);
  }

  /* If logging events log end of process and dump trace */
#ifdef EVENTLOG
  evlog(EVKEY_ENABLE,
	EVKEY_END, EVENT_PROCESS,
	EVKEY_STR_INT, "Time (cs) waiting to finish", MTIME_()-start,
	EVKEY_DUMP,
	EVKEY_LAST_ARG);
#endif

#if !(defined(DELTA) || defined(PARAGON))
  led((long) 0);
#endif
}

void SETDBG_(onoff)
     long *onoff;
/*
  Define value of debug flag
*/
{
  DEBUG = *onoff;
}

/*ARGSUSED*/
void SYNCH_(type)
     long *type;
/*
  Synchronize processes
*/
{
  gsync();
}

long NXTVAL_(mproc)
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
    (void) printf("nxtval: me=%d, mproc=%d\n",NODEID_(), *mproc);
    (void) fflush(stdout);
  }

  msgid = irecv(TYPE_NXTVAL_REPLY, (char *) buf, lenbuf);
  csend(TYPE_NXTVAL, (char *) buf, lenbuf, nxtval_server, 0);
  msgwait(msgid);

  return buf[0];
}

void PBFTOC_()
/*
  should never call this on ipsc
*/
{
  Error("PBFTOC_: what the hell are we doing here?",(long) -1);
}

void PARERR_(code)
  long *code;
/*
  Handle request for application error termination
*/
{
  Error("FORTRAN error detected", *code);
}

void STATS_()
/*
  Print out statistics for communications ... not yet implemented
*/
{
  (void) fprintf(stderr,"STATS_ not yet supported\n");
  (void) fflush(stderr);
}

void WAITCOM_(nodesel)
     long *nodesel;
/*
  Wait for all messages (send/receive) to complete between
  this node and node *nodesel or everyone if *nodesel == -1.

  !! CURRENTLY ALWAYS WAIT FOR ALL COMMS TO FINISH ... IGNORES NODESEL !!
  
  long *node = node with which to ensure communication is complete
*/
{
  long i;
#ifdef EVENTLOG
  evlog(EVKEY_BEGIN,     "Waitcom",
	EVKEY_STR_INT,   "n_in_msg_q", (int) n_in_msg_q,
	EVKEY_LAST_ARG);
#endif

  for (i=0; i<n_in_msg_q; i++) {
    if (DEBUG) {
      (void) printf("WAITCOM: %ld waiting for msgid %ld, #%ld\n",NODEID_(),
		    msg_q[i].msg_id, i);
      (void) fflush(stdout);
    }
    msgwait(msg_q[i].msg_id);
  }
  n_in_msg_q = 0;
#ifdef EVENTLOG
  evlog(EVKEY_END, "Waitcom", EVKEY_LAST_ARG);
#endif
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

void BRDCST_(type, buf, lenbuf, originator)
     long *type;
     char *buf;
     long *lenbuf;
     long *originator;
/*
  broadcast buffer to all other processes from process originator
  ... all processes call this routine specifying the same
  orginating process.
*/
{
  long me = NODEID_();
  long ttype = MAKETYPE(*type, *originator);

  if (CHKTYPE(*type))
    Error("BRDCST_: invalid type specified",*type);

  if (me == *originator)
    csend(ttype, buf, *lenbuf, (long) -1, (long) 0);
  else
    crecv(ttype, buf, *lenbuf);
}

#define GOP_BUF_SIZE 10000
#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a) (((a) >= 0) ? (a) : (-(a)))

static double gop_work[GOP_BUF_SIZE];

/*ARGSUSED*/
void DGOP_(ptype, x, pn, op)
     double *x;
     long *ptype, *pn;
     char *op;
{
  double *work = gop_work;
  long nleft  = *pn;
  long buflen = MIN(nleft,GOP_BUF_SIZE); /* Try to get even sized buffers */
  long nbuf   = (nleft-1) / buflen + 1;

  buflen = (nleft-1) / nbuf + 1;

  if (strncmp(op,"abs",3) == 0) {
    long n = *pn;
    while(n--)
      x[n] = ABS(x[n]);
  }
  
  while (nleft) {
    long ndo = MIN(nleft, buflen);

    if (strncmp(op,"+",1) == 0)
      gdsum(x, ndo, work);
    else if (strncmp(op,"*",1) == 0)
      gdprod(x, ndo, work);
    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)
      gdhigh(x, ndo, work);
    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmax",6) == 0)
      gdlow(x, ndo, work);
    else
      Error("DGOP: unknown operation requested", (long) *pn);

    nleft -= ndo; x+= ndo;
  }
}

/*ARGSUSED*/
void IGOP_(ptype, x, pn, op)
     long *x;
     long *ptype, *pn;
     char *op;
{
  long *work = (long *) gop_work;
  long nleft  = *pn;
  long buflen = MIN(nleft,2*GOP_BUF_SIZE); /* Try to get even sized buffers */
  long nbuf   = (nleft-1) / buflen + 1;

  buflen = (nleft-1) / nbuf + 1;

  if (strncmp(op,"abs",3) == 0) {
    long n = *pn;
    while(n--)
      x[n] = ABS(x[n]);
  }
  
  while (nleft) {
    long ndo = MIN(nleft, buflen);

    if (strncmp(op,"+",1) == 0)
      gisum(x, ndo, work);
    else if (strncmp(op,"*",1) == 0)
      giprod(x, ndo, work);
    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)
      gihigh(x, ndo, work);
    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmax",6) == 0)
      gilow(x, ndo, work);
    else if (strncmp(op,"or",2) == 0)
      gior(x, ndo, work);
    else
      Error("IGOP: unknown operation requested", (long) *pn);

    nleft -= ndo; x+= ndo;
  }
}

double TCGTIME_()
{
  static int first_call = 1;
  static double first_time;
  double diff;

  if (first_call) {
    first_time = dclock();
    first_call = 0;
  }

  diff = dclock() - first_time;

  return diff;			/* Add logic here for clock wrap */
}
