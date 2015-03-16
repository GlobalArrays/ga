# ARMCI_SHMMAX
# ------------
# Allow the upper limit on the ARMCI_DEFAULT_SHMMAX runtime environment
# variable to be configured by user.
AC_DEFUN([ARMCI_SHMMAX], [
AC_CACHE_CHECK([for ARMCI_DEFAULT_SHMMAX upper bound], [armci_cv_shmmax],
    [AS_IF([test "x$ARMCI_DEFAULT_SHMMAX_UBOUND" != x],
        [armci_cv_shmmax=$ARMCI_DEFAULT_SHMMAX_UBOUND],
        [armci_cv_shmmax=8192])])
AS_IF([test $armci_cv_shmmax -le 1],
    [AC_MSG_ERROR([invalid ARMCI_DEFAULT_SHMMAX upper bound; <= 1])])
AC_DEFINE_UNQUOTED([ARMCI_DEFAULT_SHMMAX_UBOUND], [$armci_cv_shmmax],
    [upper bound for ARMCI_DEFAULT_SHMMAX environment variable])
])dnl
