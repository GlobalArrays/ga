# TASCEL_PROG_MPICXX
# --------------
# If desired, replace CXX with MPICXX while searching for a C++ compiler.
#
# Known C++ compilers:
#  aCC      HP-UX C++ compiler much better than `CC', so test before.
#  c++
#  cc++
#  CC
#  cl.exe
#  cxx
#  FCC      Fujitsu C++ compiler
#  g++
#  gpp
#  icpc     Intel C++ compiler
#  KCC      KAI C++ compiler
#  RCC      Rational C++
#  bgxlC    Intel
#  bgxlC_r  Intel, thread safe
#  xlC      AIX C Set++
#  xlC_r    AIX C Set++ , thread safe
#  pgCC     Portland Group
#  pathCC   PathScale
#  sxc++    NEC SX
#
# Known MPI C++ compilers
#  cmpicc
#  hcc
#  mpcc
#  mpicc
#  mpxlc
#  mpxlc_r
#  sxmpic++ NEC SX
#  mpiFCC   Fujitsu
#
AC_DEFUN([TASCEL_PROG_MPICXX],
[AC_ARG_VAR([MPICXX], [MPI C++ compiler])
AS_CASE([$tascel_cv_target_base],
[BGP],  [tascel_mpicxx_pref=mpixlcxx_r; tascel_cxx_pref=bgxlC_r],
[])
# In the case of using MPI wrappers, set CXX=MPICXX since CXX will override
# absolutely everything in our list of compilers.
AS_IF([test x$with_mpi_wrappers = xyes],
    [tascel_save_CXX="$CXX"
     CXX="$MPICXX"
     AS_IF([test "x$MPICXX" != x],
        [AS_IF([test "x$tascel_save_CXX" != x],
            [AC_MSG_WARN([MPI compilers desired, MPICXX is set and CXX is set])
             AC_MSG_WARN([Choosing MPICXX over CXX])])],
        [AS_IF([test "x$tascel_save_CXX" != x],
            [AC_MSG_WARN([MPI compilers desired but CXX is set, ignoring])
             AC_MSG_WARN([Perhaps you mean to set MPICXX instead?])])])])
tascel_cxx="icpc pgCC pathCC sxc++ g++ c++ gpp aCC CC cxx cc++ cl.exe FCC KCC RCC xlC_r xlC"
tascel_mpicxx="mpic++ mpicxx mpiCC sxmpic++ hcp mpxlC_r mpxlC mpixlcxx_r mpixlcxx mpg++ mpc++ mpCC cmpic++ mpiFCC CC"
AS_IF([test x$with_mpi_wrappers = xyes],
    [CXX_TO_TEST="$tascel_mpicxx_pref $tascel_mpicxx"],
    [CXX_TO_TEST="$tascel_cxx_pref $tascel_cxx"])
AC_PROG_CXX([$CXX_TO_TEST])
AS_IF([test "x$enable_cxx" = xyes], [
    # AC_PROG_CXX only sets "CXX" (which is what we want),
    # but override MPICXX for the UNWRAP macro.
    AS_IF([test x$with_mpi_wrappers = xyes],
        [MPICXX="$CXX"
         TASCEL_MPI_UNWRAP])])
])dnl
