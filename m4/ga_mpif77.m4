# GA_PROG_MPIF77
# -----------------
# If desired, replace F77 with MPIF77 while searching for a Fortran 77 compiler.
# We look for 95/90 compilers first so that we can control the INTEGER size.
# The search order changes depending on the TARGET.
#
# NOTE: We prefer "FC" and "FCFLAGS" over "F77" and "FFLAGS", respectively.
# But our Fortran source is only Fortran 77.  If FC/MPIFC is set, it is
# preferred above all.
#
# Known Fortran 95 compilers:
#  f95          generic compiler name
#  fort         Compaq/HP Fortran 90/95 compiler for Tru64 and Linux/Alpha
#  ftn          native Fortran 95 compiler on Cray
#  g95          original gcc-based f95 compiler (gfortran is a fork)
#  gfortran     GNU Fortran 95+ compiler (released in gcc 4.0)
#  ifc          Intel Fortran 95 compiler for Linux/x86 (now ifort)
#  ifort        Intel Fortran 95 compiler for Linux/x86 (was ifc)
#  lf95         Lahey-Fujitsu F95 compiler
#  pghpf/pgf95  Portland Group F95 compiler
#  xlf95        IBM (AIX) F95 compiler
#  pathf95      PathScale
#  openf95      AMD's x86 open64
#  sunf95       Sun's Studio
#  crayftn      Cray
#
# Known MPI Fortran 95 compilers:
#  cmpifc       ?? not sure if this is even F95
#  ftn          native Fortran 95 compiler on Cray 
#  mpif95       generic compiler name
#  mpixlf95     IBM Blue Gene Fortran 95
#  mpixlf95_r   IBM Blue Gene Fortran 95, reentrant code
#
# Known Fortran 90 compilers:
#  epcf90       "Edinburgh Portable Compiler" F90
#  f90          generic compiler name
#  fort         Compaq/HP Fortran 90/95 compiler for Tru64 and Linux/Alpha
#  pgf90        Portland Group F90 compiler
#  xlf90        IBM (AIX) F90 compiler
#  pathf90      PathScale
#  sxf90        NEC SX Fortran 90
#  openf90      AMD's x86 open64
#  sunf90       Sun's Studio
#
# Known MPI Fortran 90 compilers:
#  cmpif90c     ??
#  mpf90        ??
#  mpif90       generic compiler name
#  sxmpif90     NEC SX Fortran 90
#
# Known Fortran 77 compilers:
#  af77         Apogee F77 compiler for Intergraph hardware running CLIX
#  f77          generic compiler names
#  fl32         Microsoft Fortran 77 "PowerStation" compiler
#  fort77       native F77 compiler on older UNIX systems
#  frt          Fujitsu F77 compiler
#  g77          GNU Fortran 77 compiler
#  pgf77        Portland Group F77 compiler
#  xlf          IBM (AIX) F77 compiler
#  pathf77      PathScale
#
# Known MPI Fortran 77 compilers:
#  hf77         ??
#  mpf77        ??
#  mpif77       generic compiler name
#  mpifrt       Fujitsu
#
AC_DEFUN([GA_PROG_MPIF77],
[AC_ARG_VAR([MPIF77], [MPI Fortran 77 compiler])
AS_CASE([$ga_cv_target_base],
[BGQ],  [ga_mpif77_pref=mpixlf77_r;ga_f77_pref=bgxlf_r],
[])
# If FC is set, override F77.  Similarly for MPIFC/MPIF77 and FCFLAGS/FFLAGS.
AS_IF([test "x$FC" != x],       [F77="$FC"])
AS_IF([test "x$MPIFC" != x],    [MPIF77="$MPIFC"])
AS_IF([test "x$FCFLAGS" != x],  [FFLAGS="$FCFLAGS"])
# In the case of using MPI wrappers, set F77=MPIF77 since F77 will override
# absolutely everything in our list of compilers.
# Save F77, just in case.
AS_IF([test x$with_mpi_wrappers = xyes],
    [AS_IF([test "x$F77" != "x$MPIF77"], [ga_orig_F77="$F77"])
     AS_CASE([x$F77:x$MPIF77],
        [x:x],  [],
        [x:x*], [F77="$MPIF77"],
        [x*:x],
[AC_MSG_WARN([MPI compilers desired but F77 is set while MPIF77 is unset.])
 AC_MSG_WARN([F77 will be ignored during compiler selection, but will be])
 AC_MSG_WARN([tested first during MPI compiler unwrapping. Perhaps you])
 AC_MSG_WARN([meant to set MPIF77 instead of or in addition to F77?])
 F77=],
        [x*:x*], 
[AS_IF([test "x$F77" != "x$MPIF77"],
[AC_MSG_WARN([MPI compilers desired, MPIF77 and F77 are set, and MPIF77!=F77.])
 AC_MSG_WARN([Choosing MPIF77 as main compiler.])
 AC_MSG_WARN([F77 will be assumed as the unwrapped MPI compiler.])])
 ga_cv_mpif77_naked="$F77"
 F77="$MPIF77"],
[AC_MSG_ERROR([F77/MPIF77 case failure])])])
ga_mpif95="mpif95 mpixlf95_r mpixlf95 ftn"
ga_mpif90="mpif90 mpixlf90_r mpixlf90 mpf90 cmpif90c sxmpif90"
ga_mpif77="mpif77 hf77 mpixlf_r mpixlf mpifrt mpf77 cmpifc"
ga_f95="xlf95 pgf95 pathf95 ifort g95 f95 fort ifc efc openf95 sunf95 crayftn gfortran lf95 ftn"
ga_f90="xlf90 f90 pgf90 pghpf pathf90 epcf90 sxf90 openf90 sunf90"
ga_f77="xlf f77 frt pgf77 pathf77 g77 cf77 fort77 fl32 af77"
AS_IF([test x$with_mpi_wrappers = xyes],
    [F77_TO_TEST="$ga_mpif77_pref $ga_mpif95 $ga_mpif90 $ga_mpif77"],
    [F77_TO_TEST="$ga_f77_pref $ga_f95 $ga_f90 $ga_f77"])
AC_PROG_F77([$F77_TO_TEST])
])dnl
