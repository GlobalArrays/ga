These are notes describing the MPI RMA runtime. These notes are intended
to help developers navigate the contents of these files and to locate specific
functionality.

The files [groups.h](groups.h) and [groups.c](groups.c) contain functionality
that describes the relation between groups in GA (which are essentially equivalent
to MPI communicators) and MPI windows. Each global array in GA has its own MPI
window associated with it and and any communication involving the global array
must use its corresponding window. On the other hand, collective operations in GA,
particularly sync, are defined on groups. To implement sync, each group must
have a list of all windows that are associated with it and there must be
functionality available to manage the association of windows with the group as
global arrays are created and destroyed. Most of the code that supports
this association is located in these two files.

[reg_win.h](reg_win.h) and [reg_win.c](reg_win.c) contain code for finding the
window corresponding to a point in registered memory. When a global array is
created, a `reg_entry_t` struct
is created for each processor in the group on which the global array is defined.
These structs are grouped into a link list so that the global array can
determine where on a remote processor the data allocated for the global array
resides. The functions in these files allow you to identify which window
contains a given pointer. It allows conversion between the pointers used in
ARMCI and the integer offsets used in MPI RMA calls.

The remainder of the code is located in [comex.c](comex.c), with a few type
declarations in [comex_impl.h](comex_impl.h). The comex.c code is has a number
of preprocessor declarations that
can be used to investigate the performance of different implemenations of the
individual ComEx operations. The USE_MPI_DATATYPES symbol uses MPI
Datatypes to send strided and vector data instead of decomposing the request
into multiple contiguous data transfers. This option should be used if at all
possible, it represents a substantial performance boost over sending multiple
individual messages. If the USE_MPI_REQUESTS variable is defined then request
based calls are used for all ComEx one-sided operations. These calls are cleared
using the `MPI_Wait` function. If USE_MPI_FLUSH_LOCAL is defined, then local
completion of the call is accomplished by using the `MPI_Win_flush_local`
function. If neither of these calls is used, then `MPI_Win_lock` and
`MPI_Win_unlock` are used to guarantee progress for one-sided operations. The
request and flush-based protocols use `MPI_Win_lock_all` on the window that is
created for each GA to create a passive synchronization epoch for each window.
Both the request-basted and flush-based protocols support true non-blocking
operations, for the lock/unlock protocol non-blocking operations default to
blocking operations and the GA wait function is a no-op.
