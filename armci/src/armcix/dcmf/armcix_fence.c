/* begin_generated_IBM_copyright_prolog                             */
/*                                                                  */
/* ---------------------------------------------------------------- */
/* (C)Copyright IBM Corp.  2007, 2008                               */
/* IBM CPL License                                                  */
/* ---------------------------------------------------------------- */
/*                                                                  */
/* end_generated_IBM_copyright_prolog                               */
/**
 * \file armci/src/x/dcmf/armcix_fence.c
 * \brief DCMF ARMCI Extension for fence operations.
 */

#include "armcix_impl.h"
#include <stdio.h>

DCMF_Protocol_t __fence_protocol;


/**
 * \brief Send a fence complete acknowledgement message to the origin node.
 *
 * \param[in] connection The armci connection data for this peer.
 *
 * \see ARMCIX_DCMF_ReceiveFenceMessage
 * \see ARMCIX_DCMF_SignalReceiveComplete
 */
void ARMCIX_DCMF_AcknowledgeFence (ARMCIX_DCMF_Connection_t * connection)
{
  //fprintf (stderr, "ARMCIX_DCMF_AcknowledgeFence() >> connection->fence.target == %d\n", connection->fence.target);
  connection->fence.target = 0;

  ARMCIX_DCMF_Control_t response;
  response.isRequest = 0;

  DCMF_Control (&__fence_protocol,
                DCMF_MATCH_CONSISTENCY,
                connection->peer,
                (DCMF_Control_t *) &response);
  //fprintf (stderr, "ARMCIX_DCMF_AcknowledgeFence() <<\n");
}


/**
 * \brief Signal that a receive operation has completed.
 *
 * If there is an active fence operation this will complete the fence
 * and send an acknowledgement to the fence operation origin node.
 *
 * \param[in] connection The armci connection data for this peer.
 *
 * \see ARMCIX_DCMF_AcknowledgeFence
 * \see ARMCIX_DCMF_RecvPut1
 * \see ARMCIX_DCMF_RecvPut2
 */
void ARMCIX_DCMF_SignalReceiveComplete (ARMCIX_DCMF_Connection_t * connection)
{
  //fprintf (stderr, "ARMCIX_DCMF_SignalReceiveComplete() >> connection->sequence.target=%d -> %d\n", connection->sequence.target, (connection->sequence.target)+1);
  connection->sequence.target++;
  //fprintf (stderr, "ARMCIX_DCMF_SignalReceiveComplete() -- connection->fence.target=%d\n", connection->fence.target);
  if (connection->fence.target)
  {
    //fprintf (stderr, "ARMCIX_DCMF_SignalReceiveComplete() -- connection->fence.watermark=%d == connection->sequence.target=%d\n", connection->fence.watermark, connection->sequence.target);
    if (connection->fence.watermark == connection->sequence.target)
    {
      ARMCIX_DCMF_AcknowledgeFence (connection);
    }
  }
  //fprintf (stderr, "ARMCIX_DCMF_SignalReceiveComplete() <<\n");
}


/**
 * \brief Receive a fence control message.
 *
 * The fence message type is either a fence \e request or a fence \e ack.
 *
 * \param[in] clientdata Registered clientdata, the armci connection array
 * \param[in] info       Fence control information
 * \param[in] peer       Rank of the node that sent this control message
 *
 * \see ARMCIX_Fence
 * \see ARMCIX_AllFence
 * \see ARMCIX_DCMF_AcknowledgeFence
 * \see ARMCIX_DCMF_Connection_t
 * \see ARMCIX_DCMF_Control_t
 * \see DCMF_RecvControl
 */
void ARMCIX_DCMF_ReceiveFenceMessage (void                 * clientdata,
                                      const DCMF_Control_t * info,
                                      unsigned               peer)
{
  ARMCIX_DCMF_Connection_t * connection = (ARMCIX_DCMF_Connection_t *) clientdata;
  ARMCIX_DCMF_Control_t    * control    = (ARMCIX_DCMF_Control_t *)    info;

  //fprintf (stderr, "ARMCIX_DCMF_ReceiveFenceMessage() >> peer == %d, control->isRequest == %d\n", peer, control->isRequest);
  
  if (control->isRequest)
  {
    /* Received a fence request message.                                     */
    //fprintf (stderr, "ARMCIX_DCMF_ReceiveFenceMessage() -- connection[%d].sequence.target == %d, control->watermark == %d\n", peer, connection[peer].sequence.target, control->watermark);
    if (connection[peer].sequence.target == control->watermark)
    {
      /* The sequence and watermark numbers already match. Send an           */
      /* immediate fence acknowledgement message to the peer.                */
      ARMCIX_DCMF_AcknowledgeFence (&connection[peer]);
    }
    else
    {
      /* The sequence and watermark numbers do not match. Set the target     */
      /* fence flag and store the fence watermark. This information will be  */
      /* used by subsequent receive message completion events to determine   */
      /* the fence operation is complete and send a fence acknowledgement    */
      /* message.                                                            */
      connection[peer].fence.target = 1;
      connection[peer].fence.watermark = control->watermark;
    }
  }
  else
  {
    /* Received fence acknowledgement message. Complete the fence operation  */
    /* on the origin node by unsetting the origin fence flag.                */
    //fprintf (stderr, "ARMCIX_DCMF_ReceiveFenceMessage() -- connection[%d].fence.origin == %d -> 0\n", peer, connection[peer].fence.origin);
    connection[peer].fence.origin = 0;
  }
  //fprintf (stderr, "ARMCIX_DCMF_ReceiveFenceMessage() <<\n");
}


/**
 * \brief Register the DCMF ARMCI Extention fence operation.
 *
 * \param[in]  connection_array Connection array
 *
 * \see DCMF_Control_register
 */
void ARMCIX_DCMF_Fence_register (ARMCIX_DCMF_Connection_t * connection_array)
{
  DCMF_CriticalSection_enter (0);

  DCMF_Control_Configuration_t configuration = {
    DCMF_DEFAULT_CONTROL_PROTOCOL,
    ARMCIX_DCMF_ReceiveFenceMessage,
    connection_array
  };
  DCMF_Control_register (&__fence_protocol, &configuration);

  DCMF_CriticalSection_exit (0);
}


/**
 * \brief Point-to-point fence operation.
 *
 * Blocks until all active messages between the local node and the remote
 * node have completed and acknowledged by the remote node.
 *
 * \param[in] proc       Rank of the remote node to fence
 *
 * \see ARMCIX_AllFence
 * \see ARMCIX_DCMF_AcknowledgeFence
 * \see ARMCIX_DCMF_ReceiveFenceMessage
 * \see ARMCIX_DCMF_SignalReceiveComplete
 */
void ARMCIX_Fence (int proc)
{
  //fprintf (stderr, "ARMCIX_Fence(%d)\n", proc);
  DCMF_CriticalSection_enter (0);

  /* Begin the fence operation by setting the origin fence flag. The remote  */
  /* fence operation is acknowledged when this flag is unset.                */
  __connection[proc].fence.origin = 1;

  ARMCIX_DCMF_Control_t info;
  info.watermark = __connection[proc].sequence.origin;
  info.isRequest = 1;

  DCMF_Control (&__fence_protocol,
                DCMF_MATCH_CONSISTENCY,
                proc,
                (DCMF_Control_t *) &info);

  /* Advance until all messages to the peer node have completed.             */
  while (__connection[proc].active) DCMF_Messager_advance ();

  /* Advance until the remote fence operation is completed.                  */
  while (__connection[proc].fence.origin) DCMF_Messager_advance ();

  DCMF_CriticalSection_exit (0);
  //fprintf (stderr, "ARMCIX_Fence(%d) <<\n", proc);
}

/**
 * \brief Global fence operation.
 *
 * Blocks until all active messages between the local node and all remote
 * nodes have completed and acknowledged by the remote node.
 *
 * \see ARMCIX_Fence
 * \see ARMCIX_DCMF_AcknowledgeFence
 * \see ARMCIX_DCMF_ReceiveFenceMessage
 * \see ARMCIX_DCMF_SignalReceiveComplete
 */
void ARMCIX_AllFence ()
{
  DCMF_CriticalSection_enter (0);

  //fprintf (stderr, "ARMCIX_AllFence() >>\n");
  //fprintf (stderr, "ARMCIX_AllFence() --\n");

  unsigned size = DCMF_Messager_size ();
  unsigned me   = DCMF_Messager_rank ();
  unsigned peer;

  ARMCIX_DCMF_Control_t info;
  info.isRequest = 1;

  /* Begin the fence operation by setting the origin fence flag. The remote  */
  /* fence operation is acknowledged when this flag is unset.                */
  for (peer = 0; peer < size; peer++)
  {
    __connection[peer].fence.origin = 1;
    info.watermark = __connection[peer].sequence.origin;

    DCMF_Control (&__fence_protocol,
                  DCMF_MATCH_CONSISTENCY,
                  peer,
                  (DCMF_Control_t *) &info);
    //fprintf (stderr, "ARMCIX_AllFence() -- sent fence request from %d to %d (info.watermark=%d)\n", me, peer,info.watermark);
  }

  /* Advance until all messages to all peer nodes have completed.            */
  //fprintf (stderr, "ARMCIX_AllFence() -- -> __global_connection.active == %d\n", __global_connection.active);
  while (__global_connection.active) DCMF_Messager_advance ();
  //fprintf (stderr, "ARMCIX_AllFence() -- <- __global_connection.active == %d\n", __global_connection.active);

  /* Advance until all remote fence operations have completed.               */
  for (peer = 0; peer < size; peer++)
  {
    //fprintf (stderr, "ARMCIX_AllFence() -- -> __connection[%d].fence.origin == %d\n", peer, __connection[peer].fence.origin);
    while (__connection[peer].fence.origin) DCMF_Messager_advance ();
    //fprintf (stderr, "ARMCIX_AllFence() -- <- __connection[%d].fence.origin == %d\n", peer, __connection[peer].fence.origin);
  }

  //fprintf (stderr, "ARMCIX_AllFence() <<\n");
  DCMF_CriticalSection_exit (0);
}

void ARMCIX_Barrier ()
{
#warning implement ARMCIX_Barrier ?
assert (0);
}
