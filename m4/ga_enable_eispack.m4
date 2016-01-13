# GA_ENABLE_EISPACK
# -----------------
# Whether to enable EISPACK routines.
AC_DEFUN([GA_ENABLE_EISPACK],
[AC_ARG_ENABLE([eispack],
    [AS_HELP_STRING([--enable-eispack],
        [enable Matrix Eigensystem Routines (EISPACK)])],
    [],
    [enable_eispack=no])
AS_IF([test "x$enable_eispack" = xno],
    [AC_DEFINE([ENABLE_EISPACK], [0], [Define to 1 if EISPACK is enabled])],
    [AC_DEFINE([ENABLE_EISPACK], [1], [Define to 1 if EISPACK is enabled])])
AM_CONDITIONAL([ENABLE_EISPACK], [test x$enable_eispack = xyes])
])dnl
