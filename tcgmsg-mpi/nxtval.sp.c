#include <stdio.h>
#include <mpproto.h>

#include "srftoc.h"


#define MAXPROC 1024
#define TYPE_NXTVAL 32768      /* Type of messages for next value    */
#define TYPE_NXTVAL_REPLY 32769


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
extern long DEBUG=0;           /* debug flag ... see setdbg */

/* Global variables */

static long dontcare, allmsg, nulltask,allgrp; /*values for EUI/EUIH wildcards*/
static long nxtval_buffer[NXTVAL_BUF_SIZE];    /* Used by handler for nxtval */
static void nxtval_handler();
static long nxtval_server;

#ifdef INTR_SAFE

/* global variables to implement interrupt safe synchronization */
char sync[MAXPROC];
long sync_msgid[MAXPROC];
#endif


/***** debug *****/
static long handler_entered=-1;
/***********************************************************/


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


static long requesting_node;     /* interrupting processor */
static long int_rcv_id;


/*\ initialization for nxtval -- called in PBEGIN
\*/
void install_nxtval()
{
int  nodes, mynode;
int len_buf = sizeof(nxtval_buffer[0]);
int status, htype=TYPE_NXTVAL, oldflag, newflag;

  wildcards();

  mpc_environ(&nodes, &mynode);
  nxtval_server = nodes - 1;


  if (mynode == nxtval_server)
  {
     requesting_node = dontcare;
     status = mpc_rcvncall(nxtval_buffer, len_buf, &requesting_node,
                         &htype, &int_rcv_id, nxtval_handler); 
    if(status == -1) 
      Error("install_nxtval: rcvncall failed:  mperrno error code ", mperrno);
  }

#ifdef INTR_SAFE
    /* post rcv for synchronization message */
    if(NODEID_() == 0)
       for(node=1;node<NNODES_() ;node++){
          status = mpc_recv(sync+node,sizeof(char),&node,&type,sync_msgid+node);
          if(status == -1)Error("install_nxval: trouble with mpc_recv",mperrno);
    }else{
       node = 0;
       status = mpc_recv(sync,sizeof(char),&node,&type,sync_msgid);
       if(status == -1) Error("install_nxtval: trouble with mpc_recv", mperrno);
    }
#endif         

  /* Synchronize processes and zero all timers on return to user code */

  mpc_sync(allgrp);     /* it blocks interrupts but it's OK this time */

  /* Ensure trap is enabled */
  newflag = 0; mp_lockrnc(&newflag, &oldflag);
  mpc_sync(allgrp);     /* it blocks interrupts but it's OK this time */
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
static  long status, htype = TYPE_NXTVAL, id;
static  long new=1, old;
static  size_t  msglen;

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

