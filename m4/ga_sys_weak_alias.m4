# GA_SYS_WEAK_ALIAS
# -----------------
# Whether pragma weak is supported.
AC_DEFUN([GA_SYS_WEAK_ALIAS], [
ax_sys_weak_alias=no
_AX_SYS_WEAK_ALIAS_PRAGMA
AM_CONDITIONAL([HAVE_SYS_WEAK_ALIAS_PRAGMA],
    [test "x$ax_cv_sys_weak_alias_pragma" = xyes])
])dnl
