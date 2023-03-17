# COMEX_ENABLE_XPMEM
# -----------------
# Whether to enable XPMEM.
AC_DEFUN([COMEX_ENABLE_XPMEM],
[AC_ARG_ENABLE([xpmem],
    [AS_HELP_STRING([--enable-xpmem], [enable XPMEM])],
    [],
    [enable_xpmem=0])
AS_IF([test "x$enable_xpmem" = xyes],
    [enable_xpmem=1],
    [enable_xpmem=0])

AC_SUBST([enable_xpmem])

AC_DEFINE_UNQUOTED([ENABLE_XPMEM], [$enable_xpmem],
    [Define to 1 if XPMEM is enabled])

AM_CONDITIONAL([ENABLE_XPMEM], [test "x$enable_xpmem" = x1])
])dnl
