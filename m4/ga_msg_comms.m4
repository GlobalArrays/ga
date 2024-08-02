# GA_MSG_COMMS()
# --------------
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
AC_ARG_WITH([mpi],
    [AS_HELP_STRING([--with-mpi[[=ARG]]],
        [select MPI as the messaging library (default); leave ARG blank to use MPI compiler wrappers])],
    [],
    [with_mpi=yes])
with_mpi_need_parse=no
AS_CASE([$with_mpi],
    [yes],  [with_mpi_wrappers=yes; ga_msg_comms=MPI],
    [*],    [with_mpi_need_parse=yes; ga_msg_comms=MPI])
dnl postpone parsing with_mpi until we know sizeof(void*)
dnl AS_IF([test x$with_mpi_need_parse = xyes],
dnl     [GA_ARG_PARSE([with_mpi], [GA_MP_LIBS], [GA_MP_LDFLAGS], [GA_MP_CPPFLAGS])])
AM_CONDITIONAL([MSG_COMMS_MPI],       [test "x$ga_msg_comms" = xMPI])
AS_CASE([$ga_msg_comms],
    [MPI],      [AC_DEFINE([MSG_COMMS_MPI], [1],
                    [Use MPI for messaging])])
AC_SUBST([GA_MP_LIBS])
AC_SUBST([GA_MP_LDFLAGS])
AC_SUBST([GA_MP_CPPFLAGS])
])dnl
