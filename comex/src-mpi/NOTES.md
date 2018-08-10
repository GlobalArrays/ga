# MPI Two-Sided (MPI-TS)

These are notes describing the MPI two-sided runtime. These notes are intended to help developers navigate the contents of these files and to locate specific functionality.

MPI-TS aims to be the basic reference implementation for other one-sided runtimes based solely on MPI_Send/MPI_Recv.  It is strictly compatible with the MPI-1 standard.  It does not use shared memory within a node.  Practically *all* use of MPI takes advantage of the non-blocking interfaces.  There is no asynchronous progress.

## Message Queues

There are three linked lists representing message queues for outgoing messages (`_mq_push`, `_mq_test`, `_mq_pop`), incoming get notifications (`_gq_push`, `_gq_test`), and mutex lock requests (`_lq_progress`).

### Active Messages

MPI-TS uses an active message concept.  The `header_t` type contains attributes for the operation, remote and local addresses, length of the data payload, and an optional notification address.  The `typedef enum op_t` represents the types of operations, for example `OP_PUT`, `OP_ACC_INT`, `OP_BARRIER_REQUEST`/`OP_BARRIER_RESPONSE`.  Messages are sent as a single contiguous buffer containing the header as well as the data payload.  The progress function (see below) queries for and responds to incoming requests from all other MPI ranks.  The `op_t` used in the active message header triggers a callback function to handle the operation.  Some operations require sending a notification message back to the originator of the request.

### MQ Message Queue

The `_mq_push` function calls `MPI_Isend` and adds the MPI_Request to the end of the linked list.  The `_mq_test` function calls `MPI_Test` but only on the head of the linked list of requests.  This function is used in conjunction with `_mq_pop` to remove the first request from the linked list head when the test of completion succeeded.

### GQ Get Queue

The GQ is for get requests only.  It functions similar to the outgoing message queue above.  A get request can be thought of as a reverse put operation.  But since all MPI-TS requests must be non-blocking, we must repeatedly call the progress engine while waiting for the `comex_get` operation to complete.  Completion is indicated by toggling a variable associated with the request.  The get queue contains a linked list of these status indicators.  The progress engine will loop until all outstanding get requests are satisfied. 

## Progress

Progress is made by aggressively calling the progress engine function `comex_make_progress` any time any other comex function is called.  This function is not externally part of the comex API.  This function checks for any incoming messages from any MPI rank.  The operation associated with the message is interpreted to a callback funtion.  The data payload is always after `sizeof(header_t)` bytes.

The progress function will loop until all outstanding incoming get requests as well as outgoing sends are satisfied.  This significantly affects performance since progress is effectively synchronous.

A special note about `comex_barrier()`.  The barrier is effectively a `comex_fence_all()` followed by an `MPI_Barrier()`.  However, if we were to use the `MPI_Barrier` directly, it would not be able to call the comex progress function -- a late fence request message is not the same as no message at all.  The comex_barrier is an expensive all-to-all communication.  This is only the case for MPI-TS; other MPI-based comex runtimes are able to use `MPI_Barrier` directly.

## Special Note About MPI-TS and MPI Interoperability

The information about the `comex_barrier` above indicates a complicated interaction between the comex and MPI runtimes.  Users *must not* make any MPI calls within a comex phase of communication, otherwise comex progress might stall and deadlock the application.  Users must take care to call `comex_barrier` before calling any other MPI function.
