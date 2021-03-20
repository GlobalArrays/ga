# ENABLE_CUDA_MEM
# ------------------
# Whether to enable CUDA memory in MA. AC_DEFINEs ENABLE_PROFILING.
AC_DEFUN([MA_ENABLE_CUDA_MEM],
[AC_ARG_ENABLE([cuda-mem],
    [AS_HELP_STRING([--enable-cuda-mem], [enable CUDA memory])],
    [],
    [enable_cuda_mem=no])
AM_CONDITIONAL([ENABLE_CUDA_MEM], [test x$enable_cuda_mem = xyes])
AS_IF([test "x$enable_cuda_mem" = xyes],
    [AC_DEFINE([ENABLE_CUDA_MEM], [1], [set to 1 if CUDA memory is enabled])],
    [AC_DEFINE([ENABLE_CUDA_MEM], [0], [set to 1 if CUDA memory is enabled])])
])dnl
