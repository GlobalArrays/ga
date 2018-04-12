# MPI Two-Sided (MPI-TS)

These are notes describing the MPI two-sided runtime. These notes are intended to help developers navigate the contents of these files and to locate specific functionality.

The MPI-TS impementation is not intended to be high-performing but instead represent the simplest MPI-compatible ARMCI/ComEx runtime.  It uses only features from the MPI-1 standard.  It does not provide asynchronous progress.  Progress is only made if a ComEx function is called.  Further, **all use of MPI is non-blocking** -- such as MPI_Isend, MPI_Irecv, MPI_Iprobe -- unless a `comex_barrier` is issued prior to an MPI collective operation.

The trick to maintaining our location consistency model is to use one MPI communicator and one MPI tag.  That way, all MPI point-to-point messages are strictly ordered according to the MPI standard.

The `comex_barrier` is an expensive, non-blocking all-to-all operation.  It is the equivalent of a global fence operation followed by a global barrier.  However, we cannot use `MPI_Barrier` directly since the MPI rank would block and not make progress on outstanding ComEx requests.  The `comex_barrier` instead issues non-blocking pings to all ranks and loops on the progress engine until all responses are received.  This guarantees all previous communication operations have completed *and* that all ranks have reached and completed the barrier.

To facilitate all of the non-blocking sends and receives, three linked lists of messages are maintained for put/acc, get, and mutexes.  The progress engine will not proceed if any request is queued.  Additionally, the progress engine is greedy and will  not proceed if any incoming IProbe is detected.