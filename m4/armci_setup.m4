# ARMCI_SETUP
# -----------
# ARMCI is a sensitive piece of code! It mostly depends on the network
# selection, but on occasion the type of system is also important (aka
# TARGET). A lot of this code was ported from the old GNUmakefile, for
# better or worse.
AC_DEFUN([ARMCI_SETUP],
[AC_REQUIRE([GA_ARMCI_NETWORK])
AS_CASE([$ga_armci_network],
[MPI_TS], [
    AC_DEFINE([PTHREADS], [1], [TODO])
    AC_DEFINE([DATA_SERVER], [1], [TODO])
    AC_DEFINE([SERVER_THREAD], [1], [TODO])
    AC_DEFINE([MPI_TS], [1], [TODO])
    ],
[MPI_MT], [
    AC_DEFINE([PTHREADS], [1], [TODO])
    AC_DEFINE([DATA_SERVER], [1], [TODO])
    AC_DEFINE([SERVER_THREAD], [1], [TODO])
    AC_DEFINE([MPI_MT], [1], [TODO])
    ],
[MPI_PT], [
    AC_DEFINE([PTHREADS], [1], [TODO])
    AC_DEFINE([DATA_SERVER], [1], [TODO])
    AC_DEFINE([SERVER_THREAD], [1], [TODO])
    AC_DEFINE([MPI_PT], [1], [TODO])
    ],
[MPI_PR], [
    AC_DEFINE([PTHREADS], [1], [TODO])
    AC_DEFINE([DATA_SERVER], [1], [TODO])
    AC_DEFINE([SERVER_THREAD], [1], [TODO])
    AC_DEFINE([MPI_PR], [1], [TODO])
    ],
[MPI3], [
    AC_DEFINE([PTHREADS], [1], [TODO])
    AC_DEFINE([DATA_SERVER], [1], [TODO])
    AC_DEFINE([SERVER_THREAD], [1], [TODO])
    AC_DEFINE([MPI3], [1], [TODO])
    ],
[MPI_SPAWN], [
    AC_DEFINE([DATA_SERVER], [1], [TODO])
    AC_DEFINE([MPI_SPAWN], [1], [TODO])
    ],
[OPENIB], [
    AC_DEFINE([PTHREADS], [1], [TODO])
    AC_DEFINE([DATA_SERVER], [1], [TODO])
    AC_DEFINE([SERVER_THREAD], [1], [TODO])
    AC_DEFINE([_REENTRANT], [1], [TODO])
    AC_DEFINE([VAPI], [1], [TODO])
    AC_DEFINE([ALLOW_PIN], [1], [TODO])
    AC_DEFINE([PEND_BUFS], [1], [TODO])
    AC_DEFINE([OPENIB], [1], [TODO])
    ],
[SOCKETS], [
    AC_DEFINE([DATA_SERVER], [1], [TODO])
    AC_DEFINE([SOCKETS], [1], [TODO])
    ]
)
AS_IF([test x$REPORT_SHMMAX != x],
    [AC_DEFINE([REPORT_SHMMAX], [1], [TODO])])
AS_IF([test x$thread_safety = xyes],
    [AC_DEFINE([POSIX_THREADS], [1], [TODO])
     AC_DEFINE([_REENTRANT], [1], [TODO])])
])dnl
