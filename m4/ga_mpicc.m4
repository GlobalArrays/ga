# GA_PROG_MPICC
# ----------------
# If desired, replace CC with MPICC while searching for a C compiler.
#
# Known C compilers
#  cc       generic compiler name
#  ccc      fujitsu ?? old cray ??
#  cl
#  ecc      Intel on IA64 ??
#  gcc      GNU
#  icc      Intel
#  bgxlc    Intel on BG/P
#  bgxlc_r  Intel on BG/P, thread safe
#  xlc      Intel
#  xlc_r    Intel, thread safe
#  pgcc     Portland Group
#  pathcc   PathScale
#  sxcc     NEC SX
#
# Known MPI C compilers:
#  cmpic++
#  hcp
#  mpcc
#  mpic++
#  mpicc
#  mpicxx
#  mpxlc
#  mpxlc_r
#  mpixlc
#  mpixlc_r
#  sxmpicc  NEC SX
#
AC_DEFUN([GA_PROG_MPICC],
[AC_ARG_VAR([MPICC], [MPI C compiler])
AS_CASE([$ga_cv_target_base],
[BGP],  [ga_mpicc_pref=mpixlc_r; ga_cc_pref=bgxlc_r],
[])
# In the case of using MPI wrappers, set CC=MPICC since CC will override
# absolutely everything in our list of compilers.
# Save CC, just in case.
AS_IF([test x$with_mpi_wrappers = xyes],
    [ga_save_CC="$CC"
     CC="$MPICC"
     AS_IF([test "x$MPICC" != x],
        [AS_IF([test "x$ga_save_CC" != x],
            [AC_MSG_WARN([MPI compilers desired, MPICC is set and CC is set])
             AC_MSG_WARN([Choosing MPICC over CC])])],
        [AS_IF([test "x$ga_save_CC" != x],
            [AC_MSG_WARN([MPI compilers desired but CC is set, ignoring])
             AC_MSG_WARN([Perhaps you meant to set MPICC instead?])])])])
ga_cc="bgxlc_r bgxlc xlc pgcc pathcc icc sxcc gcc cc ecc cl ccc"
ga_mpicc="mpicc mpixlc_r mpixlc hcc mpxlc_r mpxlc sxmpicc mpgcc mpcc cmpicc cc"
AS_IF([test x$with_mpi_wrappers = xyes],
    [CC_TO_TEST="$ga_mpicc_pref $ga_mpicc"],
    [CC_TO_TEST="$ga_cc_pref $ga_cc"])
AC_PROG_CC([$CC_TO_TEST])
# AC_PROG_CC only sets "CC" (which is what we want),
# but override MPICC for the UNWRAP macro.
AS_IF([test x$with_mpi_wrappers = xyes],
    [MPICC="$CC"
     GA_MPI_UNWRAP])
])dnl
