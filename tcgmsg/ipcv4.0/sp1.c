/*$Id: sp1.c,v 1.2 1995-02-02 23:25:50 d3g681 Exp $*/
/*\
 *       TCGMSG INTERFACE FOR THE IBM SP-1      
 *
 *            Jarek Nieplocha, 30.03.94
\*/

#include <stdio.h>
#ifdef EUIH
#  include "mpctof.c"
#  define mp_rcvncall rcvncall
#  define mp_lockrnc  lockrnc
#else
#  include <mpproto.h>
#endif

#include "srftoc.h"

#ifdef EVENTLOG
#include "evlog.h"
#endif

#define MAXPROC 1024

/* By posting multiple interrupt receives (one for each process)
 * we can improve performance of NXTVAL server by 10 -100% factor 
 */

#ifdef POST_MULT_RCV
#define NXTVAL_BUF_SIZE MAXPROC
#else
#define NXTVAL_BUF_SIZE 1 
#endif

#define INCR 1                 /* increment for NXTVAL */
#define TYPE_NXTVAL 32768      /* Type of messages for next value    */
#define TYPE_NXTVAL_REPLY 32769	/* Type for NXTVAL response */
#define SYNC_TYPE 32770		/* Type for synchronization */

long mperrno=-1;       /* EUI error code, for some reason not in current EUIH
                          remove the statement when found/fixed */

#define DEBUG_ DEBUG
static long DEBUG=0;           /* debug flag ... see setdbg */

/* Global variables */

static long dontcare, allmsg, nulltask,allgrp; /*values for EUI/EUIH wildcards*/
static long nxtval_buffer[NXTVAL_BUF_SIZE];    /* Used by handler for nxtval */
static void nxtval_handler();
static long nxtval_server;

static volatile long n_in_msg_q = 0;    /* No. in the message q */

#define MAX_Q_LEN 2048         /* Maximum no. of outstanding messages */
static struct msg_q_struct{
  long   msg_id;
  long   node;
  long   type;
  long   lenbuf;
  long   snd;
  long   from;
} msg_q[MAX_Q_LEN];


extern char* malloc();
extern char *memalign();

#ifdef INTR_SAFE

/* global variables to implement interrupt safe synchronization */
char sync[MAXPROC];
long sync_msgid[MAXPROC];
#endif


/***** debug *****/
static long handler_entered=-1;
/***********************************************************/

#ifdef EUIH
/*\ mpc_probe is missing in the EUIH
\*/
mpc_probe(node, type, bytes)
    long *node, *type, *bytes;
{
   return mp_probe(node, type, bytes);
}
#endif


/*\ Error handler
\*/
void Error(string, code)
     char *string;
     long code;
{
  (void) fflush(stdout);
  (void) fflush(stderr);

  (void) fprintf(stderr, "%3d:%s %ld(%x)\n", NODEID_(), string, code, code);
  (void) perror("system message");

  (void) fflush(stdout);
  (void) fflush(stderr);

  mpc_stopall(code);
}



/*\ Return number of the calling process ... at the moment this is
 *  just the same as the EUIH task numbering in allgrp
\*/
long NODEID_()
{
long numtask, taskid, r;
  r= mpc_environ(&numtask, &taskid);
  return (taskid);
}



/*\ Return number of USER tasks/processes (in allgrp).
\*/
long NNODES_()
{
long numtask, taskid, r;
  r= mpc_environ(&numtask, &taskid);
  return (numtask);
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
  long status, msgid;
  long me = NODEID_();
  long ttype = *type;


  if (DEBUG) {
    (void)printf("SND_: node %ld sending to %ld, len=%ld, type=%ld, sync=%ld\n",
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

  if (*sync){

    /* blocking communication in EUI-H disables dispatching interrupt handlers*/
    /* mpc_snd + mpc_status are interruptable  */

#ifdef INTR_SAFE
    status = mpc_send(buf, *lenbuf, *node, ttype, &msgid);
    if(status == -1) 
      Error("SND: mperrno error code ", mperrno);
    while((status=mpc_status(msgid)) == -1); /* nonblocking probe */
    if(status < -1) Error("blocking SND: invalid message ID ", msgid );
#else
    status = mpc_bsend(buf, *lenbuf, *node, ttype); 
#endif

  }else {
    if (n_in_msg_q >= MAX_Q_LEN)
      Error("SND: overflowing async Q limit", n_in_msg_q);

    status = mpc_send(buf, *lenbuf, *node, ttype, &msgid);
    if(status == -1) 
      Error("async. SND: mperrno error code ", mperrno);

    msg_q[n_in_msg_q].msg_id =  msgid;
    msg_q[n_in_msg_q].snd = 1;
    fprintf(stderr,"nonblocking send: MSGID: %d\n",msgid);
    if (DEBUG) {
      (void) printf("SND: me=%ld, to=%ld, len=%ld, msg_id=%ld, ninq=%ld\n",
		    me, *node, *lenbuf, msgid,
		    n_in_msg_q);
      (void) fflush(stdout);
    }
    n_in_msg_q++;
  }
#ifdef EVENTLOG
  evlog(EVKEY_END, EVENT_SND, EVKEY_LAST_ARG);
#endif
}


/*\ gets values of EUI/MPL wildcards
\*/
void wildcards()
{
long buf[4], qtype, nelem, status;

	qtype = 3;
	nelem = 4;
	status = mpc_task_query(buf,nelem,qtype);
	if(status==-1)
           Error("TCGMSG: wildcards: mpc_task_query error", -1L);

        dontcare = buf[0];
	allmsg   = buf[1];
	nulltask = buf[2];
	allgrp   = buf[3];
        /* fprintf(stderr,"dontcare=%d, allmsg=%d, nulltask=%d, allgrp=%d\n",
        dontcare, allmsg, nulltask, allgrp); */
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
  static long ttype, nbytes;
  static long node;		
  static long status, msgid;
  
  
  if (*nodeselect == -1) 
    node = dontcare; 
  else 
    node = *nodeselect;

  ttype = *type; 
  
  if (DEBUG) {
     printf("RCV_: node %ld receiving from %ld, len=%ld, type=%ld, sync=%ld\n",
		  me, *nodeselect, *lenbuf, *type, *sync);
     fflush(stdout);
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
    /* blocking communication in EUI-H disables dispatching interrupt handlers*/
    /* mpc_rcv + mpc_status are interruptable  */

#ifdef INTR_SAFE
    status = mpc_recv(buf, *lenbuf, &node, &ttype, &msgid); 
    if(status == -1) Error("RCV: mperrno error code ", mperrno);

    while((status=mpc_status(msgid)) == -1); /* nonblocking probe */
    if(status < -1) Error("blocking RCV: invalid message ID ", msgid );
    *lenmes = status; 
#else
    status = mpc_brecv(buf, *lenbuf, &node, &ttype, lenmes); 
#endif

    *nodefrom = node;          /* Get source node  */
    
    if (*lenmes > *lenbuf)
      Error("RCV_: out of range length on received message",*lenmes);

    if (ttype != *type)
      Error("RCV_: type mismatch for received message",ttype);
    
    if (*nodeselect != -1 && *nodefrom != *nodeselect)
	Error("RCV_: received message from wrong node!",*nodefrom);
  }
  else {

    if (n_in_msg_q >= MAX_Q_LEN)
      Error("RCV: overflowing async Q limit", n_in_msg_q);

    status = mpc_recv(buf, *lenbuf, &node, &ttype, &msgid); 
    if(status == -1) Error("nonblocking RCV: mperrno error code ", mperrno);

    *nodefrom = node;          /* Get source node  */
    *lenmes =  -1L;
    msg_q[n_in_msg_q].msg_id = msgid;
    msg_q[n_in_msg_q].node   = *nodeselect;
    msg_q[n_in_msg_q].type   = *type;
    msg_q[n_in_msg_q].lenbuf = *lenbuf;
    msg_q[n_in_msg_q].snd = 0;
    n_in_msg_q++;

  }

  if (DEBUG) {
      (void) printf("RCV: me=%ld, from=%ld, len=%ld, msg_id=%ld, ninq=%ld\n",
                    me, *nodeselect, *lenbuf, msg_q[n_in_msg_q].msg_id,
                    n_in_msg_q);
      (void) fflush(stdout);
  }

#ifdef EVENTLOG
  evlog(EVKEY_END, EVENT_RCV,
	EVKEY_MSG_FROM, *nodefrom,
	EVKEY_MSG_LEN, *lenmes,
	EVKEY_LAST_ARG);
#endif
}



long PROBE_(long *type, long *node)
{
long ttype, nbytes, nnode, rc ;
  
  nnode =  (*node < 0) ? dontcare : *node; 
  ttype = *type;
  rc = mpc_probe(&nnode, &ttype, &nbytes);
  if (DEBUG) 
     fprintf(stderr," %d in PROBE ret. code=%d from=%d type=%d bytes=%d\n",
             NODEID_(), rc, nnode, *type, nbytes); 
  
  return (nbytes==-1 ? 0 : 1);
}




static long requesting_node;     /* interrupting processor */
static long int_rcv_id;

void PBEGIN_()
{
  char workdir[256], *eventfile;
  long start = MTIME_();
  static long type = SYNC_TYPE, htype = TYPE_NXTVAL, msgid,node;
  static long newflag, oldflag, len_buf,status;
  void SYNCH_();
  DEBUG = 0;

  if (DEBUG) {
    (void) printf("node %ld called pbeginf\n",NODEID_());
    (void) fflush(stdout);
  }

  /* get the system wildcards */
  wildcards();

  /* Register the handler for NXTVAL service */

  nxtval_server = NNODES_() - 1;
  len_buf = sizeof(nxtval_buffer[0]);

  if (NODEID_() == nxtval_server)
#ifdef POST_MULT_RCV
  for(requesting_node=0; requesting_node<=nxtval_server; requesting_node++){
     status = mp_rcvncall(nxtval_buffer+requesting_node, &len_buf, 
                &requesting_node, &htype, &msgid, nxtval_handler); 
#else
  {
     requesting_node = dontcare;
     status = mpc_rcvncall(nxtval_buffer, len_buf, &requesting_node,
                         &htype, &int_rcv_id, nxtval_handler); 
#endif
    if(status == -1) 
      Error("PBEGIN: rcvncall failed:  mperrno error code ", mperrno);
  }

#ifdef INTR_SAFE
    /* post rcv for synchronization message */
    if(NODEID_() == 0)
       for(node=1;node<NNODES_() ;node++){
          status = mpc_recv(sync+node,sizeof(char),&node,&type,sync_msgid+node);
          if(status == -1) Error("PBEGIN: trouble with mpc_recv", mperrno);
    }else{
       node = 0;
       status = mpc_recv(sync,sizeof(char),&node,&type,sync_msgid);
       if(status == -1) Error("PBEGIN: trouble with mpc_recv", mperrno);
    }
#endif         

  /* Synchronize processes and zero all timers on return to user code */

  mpc_sync(allgrp);     /* it blocks interrupts but it's OK this time */

  /* start timer */

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

  /* Ensure trap is enabled */
  newflag = 0; mp_lockrnc(&newflag, &oldflag);
  SYNCH_(&type);
}


void PBEGINF_()
{
  PBEGIN_();
}



void PEND_()
/*
  Zero effect for sp1 version ... 
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
}


/*\ Define value of debug flag
\*/
void SETDBG_(onoff)
     long *onoff;
{
  DEBUG = *onoff;
}


/*\ Synchronize processes
\*/
void SYNCH_(type)
     long *type;
{
#ifdef INTR_SAFE
    static long ttype = SYNC_TYPE,node,inode, status;

    /* Dumb synchronization ... all send to 0 then 0 responds */
 
    /* post rcv for synchronization message */
    if(NODEID_() == 0){
      for(inode=1;inode<NNODES_();inode++)
         while (mpc_status(sync_msgid[inode]) == -1);
      for(inode=1;inode<NNODES_();inode++){
         node = inode;
         status = mpc_bsend(sync+node,sizeof(char),node,ttype);
         status = mpc_recv(sync+node,sizeof(char),&node,&ttype,sync_msgid+node);
      } 
    }else{
       node = 0;
       status = mpc_bsend(sync+node,sizeof(char),node,ttype);
       while (mpc_status(sync_msgid[node]) == -1);
       status = mpc_recv(sync,sizeof(char),&node,&ttype,sync_msgid);
    }
#else
    mpc_sync(allgrp);
#endif
}




/*\ Get next value of shared counter.
\*/
long NXTVAL_(mproc)
     long *mproc;
/*
  mproc > 0 ... returns requested value
  mproc < 0 ... server blocks until abs(mproc) processes are queued
                and returns junk
  mproc = 0 ... indicates to server that I am about to terminate
*/
{
static  long buf[1];
static  long lenbuf = sizeof(buf);
static  long lenmes, nodefrom, nodeto;
static  long sync = 1;
static  long msgid, status, type;
static  long rtype  = TYPE_NXTVAL_REPLY;   /* reply message type */
static  long  ret_val;

  buf[0] = *mproc;

  if (DEBUG_) {
    (void) printf("nxtval: me=%d, mproc=%d\n",NODEID_(), *mproc);
    (void) fflush(stdout);
  }


    type = TYPE_NXTVAL_REPLY;
    nodefrom = nxtval_server;
    status = mpc_recv((char*)buf, lenbuf, &nodefrom, &type, &msgid);
    if(status < -1) Error("NXTVAL: recv failed ", -1);

    if (DEBUG_) 
       fprintf(stderr,"nxtval: me=%d, waiting for reply type=%d from=%d\n",
         NODEID_(),type,nodefrom); 

    type = TYPE_NXTVAL;
    status = mpc_bsend((char*)buf, lenbuf, nxtval_server, type);
    if(status < -1) Error("NXTVAL: send failed ", -1);

    while((status=mpc_status(msgid)) == -1);  /* spin using nonblocking probe */
    if(status < -1) Error("NXTVAL: invalid message ID ", msgid);

    ret_val = buf[0];
  
  if (DEBUG_) 
   fprintf(stderr,"nxtval: me=%d, got reply, nextval= %d \n",
     NODEID_(),ret_val); 

  return(ret_val);
}




void PBFTOC_()
{
  Error("PBFTOC_: what the hell are we doing here?",(long) -1);
}


/*\ Handle request for application error termination
\*/
void PARERR_(code)
  long *code;
{
  Error("FORTRAN error detected", *code);
}


/*\ Print out statistics for communications ... not yet implemented
\*/
void STATS_()
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
  long i, status, nbytes;
#ifdef EVENTLOG
  evlog(EVKEY_BEGIN,     "Waitcom",
	EVKEY_STR_INT,   "n_in_msg_q",  n_in_msg_q,
	EVKEY_LAST_ARG);
#endif

  for (i=0; i<n_in_msg_q; i++) {
    if (DEBUG) {
      (void) printf("WAITCOM: %ld waiting for msgid %ld, #%ld\n",NODEID_(),
		    msg_q[i].msg_id, i);
      (void) fflush(stdout);
    }
#ifdef WAIT_BLOCKING
    status = mpc_wait(&msg_q[i].msg_id, &nbytes);
    if(status == -1) 
      Error("WAITCOM failed:  mperrno error code ", mperrno);
#else
      /* fprintf(stderr,"%d messages, now: %d\n",n_in_msg_q,i); */
      while((status=mpc_status(msg_q[i].msg_id)) == -1);    /* interruptable*/
      if(status < -1) Error("WAITCOM: invalid message ID ", msg_q[i].msg_id );
      /* fprintf(stderr,"message %d completed\n",i); */
#endif
  }
  n_in_msg_q = 0;
#ifdef EVENTLOG
  evlog(EVKEY_END, "Waitcom", EVKEY_LAST_ARG);
#endif
}



/*\ Interrupt handler
\*/
static void nxtval_handler(pid)
       long *pid;
{
static long cnt     = 0;          /* actual counter */
volatile static long ndone = 0;   /* no. finished for this loop */
static long done_list[MAXPROC];   /* list of processes finished with this loop*/
static  long lencnt = sizeof cnt;    /* length of cnt */
static  long node   = -1;            /* select any node */
static  long rtype  = TYPE_NXTVAL_REPLY;   /* reply message type */
static  long mproc;                  /* no. of processes running loop */
static  long nval;                   /* no. of values requested */
static  long sync   = 1;             /* all info goes synchronously */
static  long lenbuf = sizeof(nxtval_buffer[0]);    /* length of buffer */
static  long status, htype = TYPE_NXTVAL, msglen, id;
static  long new=1, old;

  mpc_wait(pid, &msglen);
  if (msglen != lenbuf) 
    Error("NextValueServer: lenmsg != lenbuf", msglen);

#ifdef POST_MULT_RCV
  mproc = nxtval_buffer[requesting_node];
#else
  mproc = nxtval_buffer[0];
#endif

  nval  = INCR;
  if (DEBUG_) {
    (void) printf("NVS: from=%d  mproc=%d, counter=%d, ndone=%d\n",
                    requesting_node, mproc, cnt, ndone);
  }

  if (mproc == 0)
    Error("NVS: invalid mproc ", mproc);
  else if (mproc > 0) {
      
    /* This is what we are here for */
  
    status = mpc_bsend((char*) &cnt, sizeof(cnt), requesting_node, rtype);
    cnt += nval;

  } else if (mproc < 0) {

    /* This process has finished the loop. */

    done_list[ndone++] = requesting_node;

    if (ndone == -mproc) {
      /*  all processes have finished so release them */
      while (ndone--) {
        long nodeto = done_list[ndone];
        status = mpc_bsend((char*) &cnt, sizeof(cnt), nodeto, rtype);
      }
      cnt = 0;
      ndone = 0;
    }
  }

#ifdef POST_MULT_RCV
  status = mp_rcvncall(nxtval_buffer+requesting_node, &lenbuf, &requesting_node,
                   &htype, &id, nxtval_handler);
#else
  requesting_node = dontcare;
  status = mp_rcvncall(nxtval_buffer, &lenbuf, &requesting_node, &htype, &id, 
                    nxtval_handler);
#endif
  if(status == -1)
      Error("NXTVAL handler: rcvncall failed:  mperrno error code ", mperrno);

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
  long status;
  long me = NODEID_();
  long ttype = *type;

  if (DEBUG){
    fprintf(stderr,"type=%d, (%d,%d) int=%d, long=%d\n",*type, *originator,
            ttype, sizeof(long), sizeof(long)); 
  }
  status = mpc_bcast(buf, *lenbuf, *originator, allgrp);
  if(status == -1) 
      Error("BRDCST failed: mperrno error code ", mperrno);
}




/* global operation stuff */

#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a) (((a) >= 0) ? (a) : (-(a)))

#define GOP_BUF_SIZE 50000                         /* global ops buffer size */
static double gop_work[GOP_BUF_SIZE];

/*\ d_vmul in the current EUIH is missing: had to implement our own 
\*/
void my_d_vmul(double *a, double *b, double *out, long *len)
{
long i,n;
    n = *len/sizeof(double);
    for(i=0;i<n;i++) out[i] = a[i]*b[i];
}



void DGOP_(ptype, x, pn, op)
     double *x;
     long *ptype, *pn;
     char *op;
{
  void  d_vadd(), d_vmul(),  d_vmax(),  d_vmin();
  long originator = 0, status;
  double *work = gop_work;
  long nleft  = *pn;
  long buflen = MIN(nleft,GOP_BUF_SIZE); /* Try to get even sized buffers */
  long nbuf   = (nleft-1) / buflen + 1;
  long n;

  buflen = (nleft-1) / nbuf + 1;

  if (strncmp(op,"abs",3) == 0) {
    n = *pn;
    while(n--) x[n] = ABS(x[n]);
  } 
  
  while (nleft) {
    long ndo = MIN(nleft, buflen);

    if (strncmp(op,"+",1) == 0)
      mpc_reduce(x, work, ndo*sizeof(double), originator, d_vadd, allgrp);
    else if (strncmp(op,"*",1) == 0)
      mpc_reduce(x, work, ndo*sizeof(double), originator, my_d_vmul, allgrp);
    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)
      mpc_reduce(x, work, ndo*sizeof(double), originator, d_vmax, allgrp);
    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmin",6) == 0)
      mpc_reduce(x, work, ndo*sizeof(double), originator, d_vmin, allgrp);
    else
      Error("DGOP: unknown operation requested", (long) *pn);

    status = mpc_bcast(work, ndo*sizeof(double),  originator, allgrp);
    if(status == -1) 
      Error("DGOP: broadcast failed:  mperrno error code ", mperrno);
    
    n = ndo;
    while(n--)  x[n] = work[n];

    nleft -= ndo; x+= ndo;
  }
}


void IGOP_(ptype, x, pn, op)
     long *x;
     long *ptype, *pn;
     char *op;
{
  void i_vadd(), i_vmul(), i_vmax(), i_vmin(),  b_vor();
  long originator = 0, status;
  long *work = (long *) gop_work;
  long nleft  = *pn;
  long buflen = MIN(nleft,2*GOP_BUF_SIZE); /* Try to get even sized buffers */
  long nbuf   = (nleft-1) / buflen + 1;
  long n;

  buflen = (nleft-1) / nbuf + 1;

  if (strncmp(op,"abs",3) == 0) {
    n = *pn;
    while(n--) x[n] = ABS(x[n]);
  } 
  
  while (nleft) {
    long ndo = MIN(nleft, buflen);

    if (strncmp(op,"+",1) == 0)
      mpc_reduce(x, work, ndo*sizeof(long), originator, i_vadd, allgrp);
    else if (strncmp(op,"*",1) == 0)
      mpc_reduce(x, work, ndo*sizeof(long), originator, i_vmul, allgrp);
    else if (strncmp(op,"max",3) == 0 || strncmp(op,"absmax",6) == 0)
      mpc_reduce(x, work, ndo*sizeof(long), originator, i_vmax, allgrp);
    else if (strncmp(op,"min",3) == 0 || strncmp(op,"absmax",6) == 0)
      mpc_reduce(x, work, ndo*sizeof(long), originator, i_vmin, allgrp);
    else if (strncmp(op,"or",2) == 0)
      mpc_reduce(x, work, ndo*sizeof(long), originator, b_vor, allgrp);
    else
      Error("IGOP: unknown operation requested", (long) *pn);

    status = mpc_bcast(work, ndo*sizeof(long),  originator, allgrp);
    if(status == -1) 
      Error("IGOP: broadcast failed:  mperrno error code ", mperrno);
    
    n = ndo;
    while(n--) x[n] = work[n];

    nleft -= ndo; x+= ndo;
  }
}

