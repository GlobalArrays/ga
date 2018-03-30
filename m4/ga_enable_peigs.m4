# GA_ENABLE_PEIGS
# ---------------
# Whether to enable PeIGS routines.
AC_DEFUN([GA_ENABLE_PEIGS],
[AC_ARG_ENABLE([peigs],
    [AS_HELP_STRING([--enable-peigs],
        [enable Parallel Eigensystem Solver interface])],
    [],
    [enable_peigs=0])
AS_IF([test "x$enable_peigs" = xyes],
    [enable_peigs=1],
    [enable_peigs=0])
AC_SUBST([enable_peigs])
AC_DEFINE_UNQUOTED([ENABLE_PEIGS], [$enable_peigs],
    [Define to 1 if PeIGS is enabled])
AM_CONDITIONAL([ENABLE_PEIGS], [test "x$enable_peigs" = x1])
])dnl
