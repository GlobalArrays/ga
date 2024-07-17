# MA_ENABLE_ARMCI_MEM_OPTION
# ------------------------------
# Whether to enable ARMCI in MA.
AC_DEFUN([MA_ENABLE_ARMCI_MEM_OPTION],
[AS_IF([test x$TARGET != xBGQ],
    [AC_DEFINE([ENABLE_ARMCI_MEM_OPTION], [1], [enables ARMCI in MA])])
AM_CONDITIONAL([ENABLE_ARMCI_MEM_OPTION], [test x$TARGET != xBGQ])
])dnl
