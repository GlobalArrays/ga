#include "srftoc.h"
#include "tcgmsg.h"
#include "tcgmsgP.h"

extern long MatchShmMessage();
extern long DEBUG_;

static msgids[MAX_PROC];

long ProbeNode(type, node)
     long *type, *node;
     /*
       Return 1/0 (TRUE/FALSE) if a message of the given type is available
       from the given node.  If the node is specified as -1, then all nodes
       will be examined.  Some attempt is made at ensuring fairness.

       If node is specified as -1 then this value is overwritten with the
       node that we got the message from.

       */
{
  static long  next_node = 0;

  long  nproc = NNODES_();
  long  me = NODEID_();
  long  found = 0;
  long  cur_node;
  int   i, proclo, prochi;

  if (*node == me)
    Error("PROBE_ : cannot recv message from self, msgtype=", *type);

  if (*node == -1) {                /* match anyone */

        proclo = 0;
        prochi = nproc-1;
        cur_node = next_node;

  } else
        proclo = prochi = cur_node =  *node;

  for(i = proclo; i<= prochi; i++) {

    if (cur_node != me){                /* can't receive from self */
        found = MatchShmMessage(cur_node, *type); 
        if (found) break; 
    }
    cur_node = (cur_node +1)%nproc;
       
  }

  if(found) *node = cur_node;

  /* if wildcard node, determine which node we'll start with next time */
  if(*type == -1) next_node = (cur_node +1)%nproc;
  return(found);
}



long PROBE_(type, node)
     long *type, *node;
     /*
       Return 1/0 (TRUE/FALSE) if a message of the given type is available
       from the given node.  If the node is specified as -1, then all nodes
       will be examined.  Some attempt is made at ensuring fairness.

      */
{
    long nnode = *node;
    return(ProbeNode(type, &nnode));
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

  static long ttype, nbytes;
  static long node;
  static long status, msgid;
  long   me = NODEID_();
  void msg_rcv();


  node = *nodeselect;

  ttype = *type;

  if (DEBUG_) {
     printf("RCV_: node %ld receiving from %ld, len=%ld, type=%ld, sync=%ld\n",
                  me, *nodeselect, *lenbuf, *type, *sync);
     fflush(stdout);
  }

  /* wait for a matching message */
  if(node==-1)   while(ProbeNode(type, &node) == 0);
/*  fprintf(stderr,"me=%d out of ProbeNode %d\n",me, node);*/
  msg_rcv(ttype, buf, *lenbuf, lenmes, node); 
  *nodefrom = node;  

  if (DEBUG_) {
      (void) printf("RCV: me=%ld, from=%ld, len=%ld\n",
                    me, *nodeselect, *lenbuf);
      (void) fflush(stdout);
  }
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
  void msg_wait();
  long msg_async_snd();

  if (DEBUG_) {
    (void)printf("SND_: node %ld sending to %ld, len=%ld, type=%ld, sync=%ld\n",
                  me, *node, *lenbuf, *type, *sync);
    (void) fflush(stdout);
  }

  if (*sync)
    msg_wait(msg_async_snd(*type, buf, *lenbuf, *node));
  else
    msgids[*node] = msg_async_snd(*type, buf, *lenbuf, *node);

  if (DEBUG_) {
      (void) printf("SND: me=%ld, to=%ld, len=%ld \n",
                    me, *node, *lenbuf);
      (void) fflush(stdout);
  }
}

void WAITCOM_(nodesel)
     long *nodesel;
/*
  Wait for all messages (send/receive) to complete between
  this node and node *nodesel or everyone if *nodesel == -1.
*/
{
  if (*nodesel == -1) {
    long node;
    for (node=0; node<TCGMSG_nnodes; node++)
      WAITCOM_(&node);
  }
  else if (msgids[*nodesel]) {
    msg_wait(msgids[*nodesel]);
    msgids[*nodesel] = 0;
  }
}  
