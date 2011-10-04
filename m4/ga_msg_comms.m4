# GA_MSG_COMMS([any text here disables tcgmsg output])
# ----------------------------------------------------
# Establishes all things related to messageing libraries.
# This includes the compilers to use (either standard or MPI wrappers)
# or the proper linker flags (-L), libs (-l) or preprocessor directives (-I).
# Yes, it's a beefy AC macro, but because when MPI is desired it replaces the
# usual compiler the order here is necessary and it is all interdependent.
AC_DEFUN([GA_MSG_COMMS], [
# GA_MP_* vars might exist in environment, but they are really internal.
# Reset them.
GA_MP_LIBS=
GA_MP_LDFLAGS=
GA_MP_CPPFLAGS=
# First of all, which messaging library do we want?
m4_ifblank([$1], [
AC_ARG_WITH([mpi],
    [AS_HELP_STRING([--with-mpi[[=ARG]]],
        [select MPI as the messaging library (default); leave ARG blank to use MPI compiler wrappers])],
    [],
    [with_mpi=maybe])
AC_ARG_WITH([tcgmsg],
    [AS_HELP_STRING([--with-tcgmsg],
        [select TCGMSG as the messaging library; if --with-mpi is also specified then TCGMSG over MPI is used])],
    [],
    [with_tcgmsg=no])
],[
AC_ARG_WITH([mpi],
    [AS_HELP_STRING([--with-mpi[[=ARG]]],
        [select MPI as the messaging library (default); leave ARG blank to use MPI compiler wrappers])],
    [],
    [with_mpi=yes])
])
with_mpi_need_parse=no
m4_ifblank([$1], [
AS_CASE([$with_mpi:$with_tcgmsg],
[maybe:yes],[ga_msg_comms=TCGMSG; with_mpi=no],
[maybe:no], [ga_msg_comms=MPI; with_mpi_wrappers=yes; with_mpi=yes],
[yes:yes],  [ga_msg_comms=TCGMSGMPI; with_mpi_wrappers=yes],
[yes:no],   [ga_msg_comms=MPI; with_mpi_wrappers=yes],
[no:yes],   [ga_msg_comms=TCGMSG],
[no:no],    [AC_MSG_ERROR([select at least one messaging library])],
[*:yes],    [ga_msg_comms=TCGMSGMPI; with_mpi_need_parse=yes],
[*:no],     [ga_msg_comms=MPI; with_mpi_need_parse=yes],
[*:*],      [AC_MSG_ERROR([unknown messaging library settings])])
# Hack. If TARGET=MACX and MSG_COMMS=TCGMSG, we really want TCGMSG5.
AS_CASE([$ga_cv_target_base:$ga_msg_comms],
    [MACX:TCGMSG], [ga_msg_comms=TCGMSG5])
],[
AS_CASE([$with_mpi],
    [yes],  [with_mpi_wrappers=yes],
    [no],   [],
    [*],    [with_mpi_need_parse=yes])
])
dnl postpone parsing with_mpi until we know sizeof(void*)
dnl AS_IF([test x$with_mpi_need_parse = xyes],
dnl     [GA_ARG_PARSE([with_mpi], [GA_MP_LIBS], [GA_MP_LDFLAGS], [GA_MP_CPPFLAGS])])
m4_ifblank([$1], [
# PVM is no longer supported, but there is still some code around
# referring to it.
AM_CONDITIONAL([MSG_COMMS_MPI],
    [test x$ga_msg_comms = xMPI || test x$ga_msg_comms = xTCGMSGMPI])
AM_CONDITIONAL([MSG_COMMS_PVM],       [test 1 = 0])
AM_CONDITIONAL([MSG_COMMS_TCGMSG4],   [test x$ga_msg_comms = xTCGMSG])
AM_CONDITIONAL([MSG_COMMS_TCGMSG5],   [test x$ga_msg_comms = xTCGMSG5])
AM_CONDITIONAL([MSG_COMMS_TCGMSGMPI], [test x$ga_msg_comms = xTCGMSGMPI])
AM_CONDITIONAL([MSG_COMMS_TCGMSG],
    [test x$ga_msg_comms = xTCGMSG || test x$ga_msg_comms = xTCGMSG5 || test x$ga_msg_comms = xTCGMSGMPI])
AS_CASE([$ga_msg_comms],
    [MPI],      [AC_DEFINE([MSG_COMMS_MPI], [1],
                    [Use MPI for messaging])],
    [PVM],      [],
    [TCGMSG],   [AC_DEFINE([MSG_COMMS_TCGMSG4], [1],
                    [Use TCGMSG (ipcv4.0) for messaging])
                 AC_DEFINE([MSG_COMMS_TCGMSG], [1],
                    [Use TCGMSG for messaging])
                 AC_DEFINE([TCGMSG], [1],
                    [deprecated, use MSG_COMMS_TCGMSG])],
    [TCGMSG5],  [AC_DEFINE([MSG_COMMS_TCGMSG5], [1],
                    [Use TCGMSG (ipcv5.0) for messaing])
                 AC_DEFINE([MSG_COMMS_TCGMSG], [1],
                    [Use TCGMSG for messaging])
                 AC_DEFINE([TCGMSG], [1],
                    [deprecated, use MSG_COMMS_TCGMSG])],
    [TCGMSGMPI],[AC_DEFINE([MSG_COMMS_TCGMSGMPI], [1],
                    [Use TCGMSG over MPI for messaging])
                 AC_DEFINE([MSG_COMMS_MPI], [1],
                    [Use MPI for messaging])])
])
AC_SUBST([GA_MP_LIBS])
AC_SUBST([GA_MP_LDFLAGS])
AC_SUBST([GA_MP_CPPFLAGS])
])dnl
