# GA_F77_DISABLE()
# ----------------
# Whether to disable all Fortran code.
AC_DEFUN([GA_F77_DISABLE],
[AC_ARG_ENABLE([f77],
    [AS_HELP_STRING([--disable-f77], [disable Fortran code])],
    [],
    [enable_f77=yes])
AS_IF([test "x$enable_f77" = xyes],
    [AC_DEFINE([NOFORT],     [0], [Define to 1 if not using Fortran])
     AC_DEFINE([ENABLE_F77], [1], [Define to 1 if using Fortran])],
    [AC_DEFINE([NOFORT],     [1], [Define to 1 if not using Fortran])
     AC_DEFINE([ENABLE_F77], [0], [Define to 1 if using Fortran])])
AM_CONDITIONAL([NOFORT],     [test "x$enable_f77" = xno])
AM_CONDITIONAL([ENABLE_F77], [test "x$enable_f77" = xyes])
])# GA_F77_DISABLE
