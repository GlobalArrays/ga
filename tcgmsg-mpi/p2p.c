#include <mpi.h>
#include "tcgmsgP.h"

/************************ nonblocking message list ********************/
#define MAX_Q_LEN 1024         /* Maximum no. of outstanding messages */
static struct msg_q_struct{
  MPI_Request request;
  long         node;
  long         type;
  long         lenbuf;
  long         snd;
  long         from;
} msg_q[MAX_Q_LEN];

static int n_in_msg_q=0;
/**********************************************************************/



void SND_(type, buf, lenbuf, node, sync)
     long  *type;
     Void *buf;
     long  *lenbuf;
     long  *node;
     long  *sync;
{
int ierr;
int ttype = (int)*type;

  if (DEBUG_) {
      printf("SND_: node %ld sending to %ld, len=%ld, type=%ld, sync=%ld\n",
              NODEID_(), *node, *lenbuf, *type, *sync);
      fflush(stdout);
  }

  if (*sync){

    ierr = MPI_Send(buf, (int)*lenbuf, MPI_CHAR, (int)*node, ttype,TCGMSG_Comm);
    tcgmsg_test_statusM("SND_:", ierr);
  }else{

    if (n_in_msg_q >= MAX_Q_LEN)
        Error("SND:overflowing async Q limit", n_in_msg_q);
    ierr = MPI_Isend(buf, (int)*lenbuf, MPI_CHAR,(int)*node, ttype,TCGMSG_Comm,
                     &msg_q[n_in_msg_q].request);
    tcgmsg_test_statusM("nonblocking SND_:", ierr);

    msg_q[n_in_msg_q].node   = *node;
    msg_q[n_in_msg_q].type   = *type;
    msg_q[n_in_msg_q].lenbuf = *lenbuf;
    msg_q[n_in_msg_q].snd = 1;
  }
}



void RCV_(type, buf, lenbuf, lenmes, nodeselect, nodefrom, sync)
     long  *type;
     Void *buf;
     long  *lenbuf;
     long  *lenmes;
     long  *nodeselect;
     long  *nodefrom;
     long  *sync;
{
int ierr;
int node, count = (int)*lenbuf;
MPI_Status status;
MPI_Request request;

    if (*nodeselect == -1)
      node = MPI_ANY_SOURCE;
    else
      node = (int)*nodeselect;

    if (DEBUG_) {
      printf("RCV_: node %ld receiving from %ld, len=%ld, type=%ld, sync=%ld\n",
              NODEID_(), *nodeselect, *lenbuf, *type, *sync);
      fflush(stdout);
    }

    if(*sync==0){

      if (n_in_msg_q >= MAX_Q_LEN)
         Error("nonblocking RCV_: overflowing async Q limit", n_in_msg_q);

      ierr = MPI_Irecv(buf, count, MPI_CHAR, node,(int)*type,TCGMSG_Comm,
             &request);
      tcgmsg_test_statusM("nonblocking RCV_:", ierr);

      *nodefrom = node;          /* Get source node  */
      *lenmes =  -1L;
      msg_q[n_in_msg_q].request = request;
      msg_q[n_in_msg_q].node   = *nodeselect;
      msg_q[n_in_msg_q].type   = *type;
      msg_q[n_in_msg_q].lenbuf = *lenbuf;
      msg_q[n_in_msg_q].snd = 0;
      n_in_msg_q++;

    }else{

      ierr = MPI_Recv(buf, count, MPI_CHAR, node, (int)*type,TCGMSG_Comm,
             &status);
      tcgmsg_test_statusM("RCV_:", ierr);
      ierr = MPI_Get_count(&status, MPI_CHAR, &count);
      tcgmsg_test_statusM("RCV:Get_count:", ierr);
      *nodefrom = (long)status.MPI_SOURCE; 
      *lenmes   = (long)count;
    }
}

/* ignores nodesel !! */
void WAITCOM_(nodesel)
     long *nodesel;
{
int ierr, i;
MPI_Status status;

  for (i=0; i<n_in_msg_q; i++){
    if (DEBUG_) {
      (void) printf("WAITCOM: %ld waiting for msg to/from node %ld, #%d\n",
             NODEID_(), msg_q[i].node, i);
      (void) fflush(stdout);
    }
    ierr = MPI_Wait(&msg_q[i].request, &status);
    tcgmsg_test_statusM("WAITCOM:", ierr);
  }
  n_in_msg_q = 0;
}



long PROBE_(type, node)
    long *type;
    long *node;  
{
int flag, source, ierr ;
MPI_Status status;

    source = (*node < 0) ? MPI_ANY_SOURCE  : (int) *node;
    ierr   = MPI_Iprobe(source, (int)*type, TCGMSG_Comm, &flag, &status);
    tcgmsg_test_statusM("PROBE:", ierr);

    return (flag == 0 ? 0 : 1);
}
