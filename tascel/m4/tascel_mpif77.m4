# TASCEL_PROG_MPIF77
# -----------------
# If desired, replace F77 with MPIF77 while searching for a Fortran 77 compiler.
# We look for 95/90 compilers first so that we can control the INTEGER size.
# The search order changes depending on the TARGET.
# This replaces AC_PROG_F77 because we must verify that the compiler works
# and if not the search continues.
#
# NOTE: We prefer "FC" and "FCFLAGS" over "F77" and "FFLAGS", respectively.
# But our Fortran source is only Fortran 77.  If FC/MPIFC is set, it is
# preferred above all.
#
# Known Fortran 95 compilers:
#  bgxlf95      IBM BlueGene/P F95 cross-compiler
#  blrts_xlf95  IBM BlueGene/L F95 cross-compiler
#  efc          Intel Fortran 95 compiler for IA64
#  f95          generic compiler name
#  fort         Compaq/HP Fortran 90/95 compiler for Tru64 and Linux/Alpha
#  ftn          native Fortran 95 compiler on Cray X1,XT4,XT5
#  g95          original gcc-based f95 compiler (gfortran is a fork)
#  gfortran     GNU Fortran 95+ compiler (released in gcc 4.0)
#  ifc          Intel Fortran 95 compiler for Linux/x86 (now ifort)
#  ifort        Intel Fortran 95 compiler for Linux/x86 (was ifc)
#  lf95         Lahey-Fujitsu F95 compiler
#  pghpf/pgf95  Portland Group F95 compiler
#  xlf95        IBM (AIX) F95 compiler
#  pathf95      PathScale
#
# Known MPI Fortran 95 compilers:
#  cmpifc       ?? not sure if this is even F95
#  ftn          native Fortran 95 compiler on Cray XT4,XT5
#  mpif95       generic compiler name
#  mpixlf95     IBM BlueGene/P Fortran 95
#  mpixlf95_r   IBM BlueGene/P Fortran 95, reentrant code
#  mpxlf95      IBM BlueGene/L Fortran 95
#  mpxlf95_r    IBM BlueGene/L Fortran 95, reentrant code
#
# Known Fortran 90 compilers:
#  blrts_xlf90  IBM BlueGene/L F90 cross-compiler
#  epcf90       "Edinburgh Portable Compiler" F90
#  f90          generic compiler name
#  fort         Compaq/HP Fortran 90/95 compiler for Tru64 and Linux/Alpha
#  pgf90        Portland Group F90 compiler
#  xlf90        IBM (AIX) F90 compiler
#  pathf90      PathScale
#  sxf90        NEC SX Fortran 90
#
# Known MPI Fortran 90 compilers:
#  cmpif90c     ??
#  mpf90        ??
#  mpif90       generic compiler name
#  mpxlf90      IBM BlueGene/L Fortran 90
#  mpxlf90_r    IBM BlueGene/L Fortran 90, reentrant code
#  sxmpif90     NEC SX Fortran 90
#
# Known Fortran 77 compilers:
#  af77         Apogee F77 compiler for Intergraph hardware running CLIX
#  blrts_xlf    IBM BlueGene/L F77 cross-compiler
#  cf77         native F77 compiler under older Crays (prefer over fort77)
#  f77          generic compiler names
#  fl32         Microsoft Fortran 77 "PowerStation" compiler
#  fort77       native F77 compiler under HP-UX (and some older Crays)
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
#  mpxlf        IBM BlueGene/L Fortran 77
#  mpxlf_r      IBM BlueGene/L Fortran 77, reentrant code
#  mpifrt       Fujitsu
#
AC_DEFUN([TASCEL_PROG_MPIF77],
[AC_ARG_VAR([MPIF77], [MPI Fortran 77 compiler])
AS_CASE([$tascel_cv_target_base],
[BGP],  [tascel_mpif77_pref=mpixlf77_r;tascel_f77_pref=bgxlf_r],
[BGL],  [tascel_mpif77_pref=mpxlf95;   tascel_f77_pref=blrts_xlf95],
[])
# If FC is set, override F77.  Similarly for MPIFC/MPIF77 and FCFLAGS/FFLAGS.
AS_IF([test "x$FC" != x],       [F77="$FC"])
AS_IF([test "x$MPIFC" != x],    [MPIF77="$MPIFC"])
AS_IF([test "x$FCFLAGS" != x],  [FFLAGS="$FCFLAGS"])
# In the case of using MPI wrappers, set F77=MPIF77 since F77 will override
# absolutely everything in our list of compilers.
# Save F77, just in case.
AS_IF([test x$with_mpi_wrappers = xyes],
    [tascel_save_F77="$F77"
     F77="$MPIF77"
     AS_IF([test "x$MPIF77" != x],
        [AS_IF([test "x$tascel_save_F77" != x],
            [AC_MSG_WARN([MPI compilers desired, MPIF77 is set and F77 is set])
             AC_MSG_WARN([Choosing MPIF77 over F77])])],
        [AS_IF([test "x$tascel_save_F77" != x],
            [AC_MSG_WARN([MPI compilers desired but F77 is set, ignoring])
             AC_MSG_WARN([Perhaps you meant to set MPIF77 instead?])])])
])
tascel_mpif95="mpif95 mpxlf95_r mpxlf95 ftn"
tascel_mpif90="mpif90 mpxlf90_r mpxlf90 mpf90 cmpif90c sxmpif90"
tascel_mpif77="mpif77 hf77 mpxlf_r mpxlf mpifrt mpf77 cmpifc"
tascel_f95="xlf95 pgf95 pathf95 ifort g95 f95 fort ifc efc gfortran lf95 ftn"
tascel_f90="xlf90 f90 pgf90 pghpf pathf90 epcf90 sxf90"
tascel_f77="xlf f77 frt pgf77 pathf77 g77 cf77 fort77 fl32 af77"
AS_IF([test x$with_mpi_wrappers = xyes],
    [F77_TO_TEST="$tascel_mpif77_pref $tascel_mpif95 $tascel_mpif90 $tascel_mpif77"],
    [F77_TO_TEST="$tascel_f77_pref $tascel_f95 $tascel_f90 $tascel_f77"])
AC_PROG_F77([$F77_TO_TEST])
# AC_PROG_F77 only sets F77 (which is what we want),
# but override MPIF77 for the UNWRAP macro.
AS_IF([test x$with_mpi_wrappers = xyes],
    [MPIF77="$F77"
     TASCEL_MPI_UNWRAP])
])dnl
