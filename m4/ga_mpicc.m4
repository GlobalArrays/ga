# GA_PROG_MPICC
# ----------------
# If desired, replace CC with MPICC while searching for a C compiler.
#
# Known C compilers
#  cc       generic compiler name
#  ccc      Fujitsu ?? old Cray ??
#  cl
#  gcc      GNU
#  icc      Intel
#  bgxlc    Intel on BG/P
#  bgxlc_r  Intel on BG/P, thread safe
#  xlc      Intel
#  xlc_r    Intel, thread safe
#  pgcc     Portland Group
#  pathcc   PathScale
#  fcc      Fujitsu
#  opencc   AMD's x86 open64
#  suncc    Sun's Studio
#  craycc   Cray
#
# Known MPI C compilers:
#  mpicc
#  mpixlc_r
#  mpixlc
#  hcc
#  mpxlc_r
#  mpxlc
#  mpifcc   Fujitsu
#  mpgcc
#  mpcc
#  cmpicc
#  cc
#
AC_DEFUN([GA_PROG_MPICC],
[AC_ARG_VAR([MPICC], [MPI C compiler])
AS_CASE([$ga_cv_target_base],
[BGQ],  [ga_mpicc_pref=mpixlc_r; ga_cc_pref=bgxlc_r],
[])
# In the case of using MPI wrappers, set CC=MPICC since CC will override
# absolutely everything in our list of compilers.
# Save CC, just in case.
AS_IF([test x$with_mpi_wrappers = xyes],
    [AS_IF([test "x$CC" != "x$MPICC"], [ga_orig_CC="$CC"])
     AS_CASE([x$CC:x$MPICC],
        [x:x],  [],
        [x:x*], [CC="$MPICC"],
        [x*:x],
[AC_MSG_WARN([MPI compilers desired but CC is set while MPICC is unset.])
 AC_MSG_WARN([CC will be ignored during compiler selection, but will be])
 AC_MSG_WARN([tested first during MPI compiler unwrapping. Perhaps you])
 AC_MSG_WARN([meant to set MPICC instead of or in addition to CC?])
 CC=],
        [x*:x*], 
[AS_IF([test "x$CC" != "x$MPICC"],
[AC_MSG_WARN([MPI compilers desired, MPICC and CC are set, and MPICC!=CC.])
 AC_MSG_WARN([Choosing MPICC as main compiler.])
 AC_MSG_WARN([CC will be assumed as the unwrapped MPI compiler.])])
 ga_cv_mpic_naked="$CC"
 CC="$MPICC"],
[AC_MSG_ERROR([CC/MPICC case failure])])])
ga_cc="bgxlc_r bgxlc xlc_r xlc pgcc pathcc icc sxcc fcc opencc suncc craycc gcc cc ecc cl ccc"
ga_mpicc="mpicc mpixlc_r mpixlc hcc mpxlc_r mpxlc sxmpicc mpifcc mpgcc mpcc cmpicc cc"
AS_IF([test x$with_mpi_wrappers = xyes],
    [CC_TO_TEST="$ga_mpicc_pref $ga_mpicc"],
    [CC_TO_TEST="$ga_cc_pref $ga_cc"])
AC_PROG_CC([$CC_TO_TEST])
])dnl

# GA_PROG_CC_NOMPI
# ----------------
# In at least one case, GA needs a standard C compiler i.e. not an MPI
# wrapper. Find one and set it to CC_NOMPI (not CC). Defaults to the CC found
# by GA_PROG_MPICC.
AC_DEFUN([GA_PROG_CC_NOMPI],
[AS_IF([test "x$ga_cv_mpic_naked" != x],
    [CC_NOMPI="$ga_cv_mpic_naked"
     AC_SUBST([CC_NOMPI])],
    [AC_CHECK_PROGS([CC_NOMPI], [$ga_cc], [$CC])])
])dnl
