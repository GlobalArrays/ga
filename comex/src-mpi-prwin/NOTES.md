# MPI Progress Ranks (MPI-PR)
NOTES on generalization of number of Progress-Ranks per computing node
------
Number of Progress-Ranks per node can be suitably 
User has the flexibility to choose the number of Progress-Ranks per
node during the execution of a program. 
An environment variable GA_NUM_PROGRESS_RANKS_PER_NODE will capture the
number to be set for number of Progress-Ranks per node. If not set
in the environment, then this number will set to 1 by default. 
e.g. of defining this variable in the environment during execution:
```
$ export GA_NUM_PROGRESS_RANKS_PER_NODE=1
$ export GA_NUM_PROGRESS_RANKS_PER_NODE=2
$ export GA_NUM_PROGRESS_RANKS_PER_NODE=4
$ setenv GA_NUM_PROGRESS_RANKS_PER_NODE 2
```

CAUTION: For optimum performance, number of MPI ranks per node used
to execute a program should be in multiples of value set for
GA_NUM_PROGRESS_RANKS_PER_NODE. 

With the use of two optional environmental variables, it is possible to set
the GA-Ranks in one group managed by Progress-Rank in either PACKED or CYCLIC
distribution pattern. 
Following example shows an application run with 8 MPI processes/node with 2-PRs.
Total 8 MPI ranks will be divided into two groups and by default, the highest rank 
will be the Progress-Rank of each group. 

PACKED distribution
```
$ export GA_NUM_PROGRESS_RANKS_PER_NODE=2
$ export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=1
```
or
```
$ export GA_NUM_PROGRESS_RANKS_PER_NODE=2
$ export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=Y
```

Above environmental settings sets 2 Progress-Ranks/node and PACKED distribution
The two groups are (0,1,2,3) and (4,5,6,7)
Here, MPI ranks 3 and 7 are Progress-Ranks on the node

Next, CYCLIC distribution
```
$ export GA_NUM_PROGRESS_RANKS_PER_NODE=2
$ export GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=0 
$ export GA_PROGRESS_RANKS_DISTRIBUTION_CYCLIC=1
```

Please make sure to set GA_PROGRESS_RANKS_DISTRIBUTION_PACKED=0 if previously used.

Above environmental settings sets 2 Progress-Ranks/node and CYCLIC distribution
The two groups are (0,2,4,6) and (1,3,5,7)
Here, MPI ranks 6 and 7 are Progress-Ranks on the node

Other notes on MPI Progress-Rank
------
These are notes describing the MPI Progress ranks runtime. These notes are intended to help developers navigate the contents of these files and to locate specific functionality.

The MPI-PR is intended to be the highest-performing MPI-1 compatible ARMCI/ComEx runtime.  It uses only features from the MPI-1 standard and provides asynchronous progress.  Posix shared memory is used extensively.

Asynchronous progress is made by reserving one MPI rank per compute node, taken
from MPI_COMM_WORLD, and using posix shared memory between the reserved
"progress rank" and the remaining ranks on the associated compute node.  The
world communicator is split using `MPI_Comm_split()` and using the result of
`gethostid()` to separate the user/worker ranks from the progress ranks.  The
progress rank is the largest rank on each compute node, unless you set
`MASTER_IS_SMALLEST_SMP_RANK` in the [comex_impl.h](comex_impl.h) file to 1.
When a user's get/put/acc request is made, it is sent as a small MPI message to
the progress rank.  The progress rank interprets the header and processes the
request, copying from/to the shared memory of one of the MPI ranks the progress
rank is managing on its compute node. The code that implements the progress
engine can be found in the `_progress_server` function located in the
[comex.c](comex.c) file.

Incoming requests to the progress rank are all based on the active message
concept. A 'header' message is sent first to the progress engine indicating the
type of request, e.g., OP_PUT, OP_ACC_INT. A complete listing of the request
types is defined in an enumerated list at the top of [comex.c](comex.c). The header contains enough information to complete the request such as source and destination pointers, source and destination ranks, etc. After a header message is sent, any data payload is sent as a separate message. The intent was to let MPI directly use the buffer pointers in case the buffers were allocated using any special, network-specific allocator. Otherwise, a data payload could have been mem-copied to the end of the header message (this is done in the MPI-PR implementation as an optimization).

Posix shared memory is used between all ranks on a compute node, including the reserved progress rank.  When `comex_malloc` is called (collectively), it calls `comex_malloc_local` that creates the shared memory buffer on each user-level MPI rank.  The posix shmem names associated with each buffer is collectively exchanged with all ranks on the node so that all ranks on the same node can access each other's memory directly.  The progress rank does not allocate memory, but rather attaches to all segments allocated on it's node-local ranks.  The shmem name is guaranteed to be unique to the UID and PID and uses an internal counter.

There are a finite number of user-level non-blocking handles. This is set using the environment variable COMEX_MAX_NB_OUTSTANDING. This controls the size of an allocated array of our non-blocking handle data structure nb_t. The nb_t structure contains linked lists of MPI_Request objects associated with the given user-level handle. It is slightly more complicated than that since get requests might be using the packing optimization where the request is first compressed into a contiguous buffer. The stride information is kept with the nb_t message so that the received buffer can be unpacked. All memory is freed when operations complete.
