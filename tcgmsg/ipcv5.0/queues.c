/* $$ */
#include "tcgmsgP.h"

static const long false = 0;
static const long true  = 1;

extern void USleep(long);
extern void Busy(int);

extern long async_send(SendQEntry *);

static long NextMsgID(long node)
/*
  Given a nodeid return a unqiue integer constructed by
  combining it with the value of a counter
*/
{
  static long id = 0;
  static long mask = (1<<20)-1;

  id = (id + 1) & mask;
  if (id == 0) id = 1;
  
  return (node << 20) + id;
}

static long NodeFromMsgID(long msgid)
/*
  Given an id from NextMsgID extract the node
*/
{
  long node = msgid >> 20;

  if (node < 0 || node > NNODES_())
    Error("NodeFromMsgID: invalid msgid", msgid);

  return node;
}

static void flush_send_q_node(long node)
/*
  Flush as many messages as possible without blocking from
  the send q to the specified node.
*/
{
  while (TCGMSG_proc_info[node].sendq) {
    
    if (!async_send(TCGMSG_proc_info[node].sendq)) {
      /* Send is incomplete ... stop processing this q*/
      break;
    }
    else {
      SendQEntry *tmp = TCGMSG_proc_info[node].sendq;
      
      TCGMSG_proc_info[node].sendq = (SendQEntry *) TCGMSG_proc_info[node].sendq->next;
      if (tmp->free_buf_on_completion)
	(void) free(tmp->buf);
      tmp->active = false;	/* Matches NewSendQEntry() */
    }
  }
}

void flush_send_q()
/*
  Flush as many messages as possible without blocking
  from all of the send q's.
*/
{
  long node;
  long nproc = NNODES_();

  for (node=0; node<nproc; node++)
    if (TCGMSG_proc_info[node].sendq)
      flush_send_q_node(node);
}    

long msg_status(msgid)
     long msgid;
/*
  Return 0 if the message operation is incomplete.
  Return 1 if the message operation is complete.
*/
{
  long node = NodeFromMsgID(msgid);
  SendQEntry *entry;
  long status = 1;

  flush_send_q();

  /* Attempt to find the msgid in the message q.  If it is not
     there then the send is complete */

  for (entry=TCGMSG_proc_info[node].sendq; entry; entry=(SendQEntry *) entry->next) {
    if (entry->msgid == msgid) {
      status = 0;
      break;
    }
  }

  return status;
}

void msg_wait(long msgid)
/*
  Wait for the operation referred to by msgid to complete.
*/
{
  long nspin = 0;
#ifdef NOSPIN
  long spinlim = 100;
# ifdef CRAY
  long waittim = 10000;
# endif
#else
  long spinlim = 1000000;
# ifdef CRAY
  long waittim = 100000;
# endif
#endif  

  while (!msg_status(msgid)) {
    nspin++;
    if (nspin < spinlim)
      Busy(100);
    else 
#ifdef CRAY
      USleep(waittim);
#else
      usleep(1);
#endif
  }
}

static SendQEntry *NewSendQEntry(void)
{
  SendQEntry *new = TCGMSG_sendq_ring;

  if (new->active)
    Error("NewSendQEntry: too many outstanding sends\n", 0L);

  TCGMSG_sendq_ring = (SendQEntry *) TCGMSG_sendq_ring->next_in_ring;

  new->active = true;

  return new;
}

long msg_async_snd(type, buf, lenbuf, node)
     long type;
     char *buf;
     long lenbuf;
     long node;
{
  long msgid;
  SendQEntry *entry;

  if (node<0 || node>=TCGMSG_nnodes)
    Error("msg_async_send: node is out of range", node);

  if (node == TCGMSG_nodeid)
    Error("msg_async_send: cannot send to self", node);

  msgid = NextMsgID(node);
  entry = NewSendQEntry();

  /* Insert a new entry into the q */

  entry->tag   = TCGMSG_proc_info[node].n_snd++; /* Increment tag */
  entry->msgid = msgid;
  entry->type  = type;
#ifdef CRAY_T3D
  /* allignment is critical on T3D (shmem library) */
  if (((unsigned long) buf) & 7) {
    printf("%2ld: mallocing unalinged buffer len=%ld\n",
           TCGMSG_nodeid, lenbuf);
    fflush(stdout);
    if (!(entry->buf = malloc((size_t) lenbuf)))
       Error("msg_sync_send: malloc failed", lenbuf);
    (void) memcpy(entry->buf, buf, lenbuf);
    entry->free_buf_on_completion = 1;
  }
  else 
#endif
  {
    entry->buf   = buf;
    entry->free_buf_on_completion = 0;
  }
  entry->lenbuf= lenbuf;
  entry->node  = node;
  entry->next  = (SendQEntry *) 0;
  entry->written = 0;
  entry->buffer_number = 0;

  /* Attach to the send q */

  if (!TCGMSG_proc_info[node].sendq)
    TCGMSG_proc_info[node].sendq = entry;
  else {
    SendQEntry *cur = TCGMSG_proc_info[node].sendq;
    
    while (cur->next)
      cur = cur->next;
    cur->next = entry;
  }

  /* Attempt to flush the send q */

  flush_send_q();

  return msgid;
}

void msg_snd(long type, char *buf, long lenbuf, long node)
/*
  synchronous send of message to a process

  long *type     = user defined integer message type (input)
  char *buf      = data buffer (input)
  long *lenbuf   = length of buffer in bytes (input)
  long *node     = node to send to (input)

  for zero length messages only the header is sent
*/
{
  msg_wait(msg_async_snd(type, buf, lenbuf, node));
}
