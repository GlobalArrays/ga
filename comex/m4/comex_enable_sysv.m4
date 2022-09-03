# COMEX_ENABLE_SYSV
# -----------------
# Whether to enable System V shared memory.
AC_DEFUN([COMEX_ENABLE_SYSV],
[AC_ARG_ENABLE([sysv],
    [AS_HELP_STRING([--enable-sysv], [enable System V shared memory])],
    [],
    [enable_sysv=0])
AS_IF([test "x$enable_sysv" = xyes],
    [enable_sysv=1],
    [enable_sysv=0])

AC_SUBST([enable_sysv])

AC_DEFINE_UNQUOTED([ENABLE_SYSV], [$enable_sysv],
    [Define to 1 if SYSV is enabled])

AM_CONDITIONAL([ENABLE_SYSV], [test "x$enable_sysv" = x1])
])dnl
