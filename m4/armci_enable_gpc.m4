# ARMCI_ENABLE_GPC
# ----------------
# Whether to enable GPC calls in ARMCI.
AC_DEFUN([ARMCI_ENABLE_GPC],
[AC_ARG_ENABLE([armci_gpc],
    [AS_HELP_STRING([--enable-armci-gpc], [Enable GPC calls in ARMCI])],
    [enable_armci_gpc=yes
    AC_DEFINE([ARMCI_ENABLE_GPC_CALLS], [1],
        [Define if GPC calls are enabled])],
    [enable_armci_gpc=no])
AM_CONDITIONAL([ARMCI_ENABLE_GPC_CALLS], [test x$enable_armci_gpc = xyes])
])dnl
