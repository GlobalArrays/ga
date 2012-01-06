# Process this file with autoconf to produce a configure script.

###############################################################################
# Init autoconf
###############################################################################
AC_PREREQ([2.67])
AC_INIT([Aggregate Remote Memory Copy Interface (ARMCI)], [1.5],
        [hpctools@pnl.gov], [armci],
        [http://www.emsl.pnl.gov/docs/parsoft/armci/])
AC_CONFIG_AUX_DIR([build-aux])
AC_CONFIG_HEADER([config.h])
AC_CONFIG_LIBOBJ_DIR([compat])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_SRCDIR([src/common/armci.c])

###############################################################################
# Must occur before automake init
###############################################################################
GA_TARGET
GA_CROSS_COMPILING

###############################################################################
# Init automake
###############################################################################
AM_INIT_AUTOMAKE([color-tests foreign parallel-tests silent-rules subdir-objects])

# Don't emit "rebuild rules" for configure, Makefile.ins, etc.
AM_MAINTAINER_MODE

###############################################################################
# Misc. information and package setup.
###############################################################################
GA_WITH_HELP

ARMCI_TOP_BUILDDIR="`pwd`"
cd "$srcdir"
ARMCI_TOP_SRCDIR="`pwd`"
cd "$ARMCI_TOP_BUILDDIR"
AS_IF([test "$ARMCI_TOP_BUILDDIR" != "$ARMCI_TOP_SRCDIR"],
      [AC_MSG_NOTICE([Detected VPATH build])])

# Determine messaging library up front because if MPI is desired we use the
# MPI compiler wrappers instead of the standard compilers.
GA_MSG_COMMS

# Whether to enable F77 source dependency.
# Note that compiling the Fortran 77 source is enabled by default.
GA_F77_DISABLE

# Hack to enable NEW_MALLOC feature
AC_ARG_ENABLE([portals-new-malloc],
    [AS_HELP_STRING([--enable-portals-new-malloc],
        [add -DNEW_MALLOC to CPPFLAGS])])
AS_IF([test "x$enable_portals_new_malloc" = xyes],
    [AC_DEFINE([NEW_MALLOC], [1], [for portals, enable new malloc])])
AM_CONDITIONAL([PORTALS_ENABLE_NEW_MALLOC],
    [test "x$enable_portals_new_malloc" = xyes])

ARMCI_ENABLE_GPC
ARMCI_ENABLE_GROUP
ARMCI_ENABLE_NB_NONCONT
ARMCI_ENABLE_PROFILING
GA_DISABLE_MPI_TESTS
GA_ENABLE_WARNINGS
GA_ENABLE_CHECKPOINT
GA_ENABLE_OPT
GA_ENABLE_TRACE
GA_THREAD_SAFE

# Unless we rename all preprocessor symbols of MPI and MPIPP
# in the source code, this hack is necessary.
AH_BOTTOM([/* DO NOT MODIFY THE FOLLOWING PREPROCESSOR DIRECTIVES */
#ifdef MSG_COMMS_MPI
#   ifdef __cplusplus
#       define MPIPP
#   else
#       define MPI
#   endif /* __cplusplus */
#endif /* MSG_COMMS_MPI */])

#########################################
# C compiler
#########################################
AC_MSG_NOTICE
AC_MSG_NOTICE([C compiler])
AC_MSG_NOTICE

ga_save_CFLAGS="$CFLAGS"
GA_PROG_MPICC
CFLAGS="$ga_save_CFLAGS"
AC_USE_SYSTEM_EXTENSIONS
GA_COMPILER_VENDOR
GA_WARN_FLAGS
ga_save_CFLAGS="$CFLAGS"
CFLAGS="$ga_cv_c_warning_flags $CFLAGS"

AS_IF([test x$with_mpi_wrappers = xyes],
    [GA_MPI_UNWRAP],
    [GA_ARG_PARSE([with_mpi], [GA_MP_LIBS], [GA_MP_LDFLAGS], [GA_MP_CPPFLAGS])])
# If F77 is enabled, only perform a simple compilation test.
AS_IF([test "x$with_mpi" != xno], [
    AS_CASE([$enable_f77:$enable_mpi_tests],
        [yes:*], [GA_MPICC_TEST_COMPILE],
        [no:yes],[GA_MPICC_TEST_LINK],
        [no:no], [GA_MPICC_TEST_COMPILE])])
# Hack to append .x to executables.
AC_SUBST([EXEEXT], [.x$EXEEXT])

GA_TARGET64

# Establish the underlying network infrastructure (SOCKETS, OPENIB, etc)
GA_ARMCI_NETWORK

# Checks for C header files.
AC_HEADER_ASSERT
AC_HEADER_DIRENT
AC_HEADER_STDBOOL
AC_HEADER_STDC
AC_HEADER_SYS_WAIT
GA_CHECK_HEADERS([assert.h c_asm.h errno.h fcntl.h float.h ia64/sys/inline.h malloc.h math.h memory.h mpp/shmem.h netdb.h netinet/in.h netinet/tcp.h process.h setjmp.h signal.h stdarg.h stdint.h stdio.h stdlib.h string.h strings.h sys/types.h sys/atomic_op.h sys/errno.h sys/file.h sys/ipc.h sys/mman.h sys/param.h sys/sem.h sys/shm.h sys/socket.h sys/stat.h sys/syscall.h sys/systemcfg.h sys/time.h sys/uio.h sys/wait.h time.h unistd.h windows.h winsock.h rpc/rpc.h rpc/types.h rpc/xdr.h],
[], [],
[@%:@ifdef HAVE_RPC_TYPES_H
@%:@include <rpc/types.h>
@%:@endif
@%:@ifdef HAVE_SYS_TYPES_H
@%:@include <sys/types.h>
@%:@endif
@%:@ifdef HAVE_SYS_IPC_H
@%:@include <sys/ipc.h>
@%:@endif])

# Checks for C typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_INLINE
AC_C_RESTRICT
AC_C_VOLATILE
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_OFF_T
AC_TYPE_PID_T
AC_TYPE_SIGNAL
AC_TYPE_SIZE_T
AC_TYPE_SSIZE_T
AC_TYPE_UINT8_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_CHECK_TYPES([ptrdiff_t])
GA_FUNCTION
GA_DISABLE_SYS_WEAK_ALIAS
GA_SYS_WEAK_ALIAS
GA_UNION_SEMUN

# Checks for C type sizes.
AC_CHECK_SIZEOF([void*])
AC_CHECK_SIZEOF([char])
AC_CHECK_SIZEOF([short])
AC_CHECK_SIZEOF([int])
AC_CHECK_SIZEOF([long])
AC_CHECK_SIZEOF([long long])
AC_CHECK_SIZEOF([float])
AC_CHECK_SIZEOF([double])
GA_C_POINTER_AS_INTEGER

# Checks for C library functions.
AC_FUNC_FORK
AC_FUNC_MMAP
AC_FUNC_SELECT_ARGTYPES
AC_FUNC_VPRINTF
AC_SEARCH_LIBS([floor], [m],
    [AC_DEFINE([HAVE_FLOOR], [1],
               [Define to 1 if you have the `floor' function.])],
    [AC_DEFINE([HAVE_FLOOR], [0],
               [Define to 1 if you have the `floor' function.])])
AC_SEARCH_LIBS([pow], [m],
    [AC_DEFINE([HAVE_POW], [1],
               [Define to 1 if you have the `pow' function.])],
    [AC_DEFINE([HAVE_POW], [0],
               [Define to 1 if you have the `pow' function.])])
AC_SEARCH_LIBS([sqrt], [m],
    [AC_DEFINE([HAVE_SQRT], [1],
               [Define to 1 if you have the `sqrt' function.])],
    [AC_DEFINE([HAVE_SQRT], [0],
               [Define to 1 if you have the `sqrt' function.])])
GA_CHECK_FUNCS([bzero fastbcopy ftruncate gethostbyname getpagesize gettimeofday memset munmap select socket strchr strdup strerror strstr strtol _lock_try _acquire_lock xdr_char])
GA_CHECK_FUNCS([pthread_getconcurrency pthread_setconcurrency])

# Checks for C libraries.

ARMCI_C_OPT

CFLAGS="$ga_save_CFLAGS"

#########################################
# C++ compiler
#########################################
AC_MSG_NOTICE
AS_IF([test "x$enable_cxx" = xyes],
    [AC_MSG_NOTICE([C++ compiler])],
    [AC_MSG_NOTICE([C++ compiler (disabled, but some tests still required)])])
AC_MSG_NOTICE
AC_LANG_PUSH([C++])

# GA_PROG_MPICXX is required to silence complaints that C++ source exists
# even if C++ support has been disabled.
ga_save_CXXFLAGS="$CXXFLAGS"
GA_PROG_MPICXX
CXXFLAGS="$ga_save_CXXFLAGS"
AS_IF([test x$with_mpi_wrappers = xyes], [GA_MPI_UNWRAP])

AS_IF([test "x$enable_cxx" = xyes], [

GA_COMPILER_VENDOR
GA_WARN_FLAGS
ga_save_CXXFLAGS="$CXXFLAGS"
CXXFLAGS="$ga_cv_cxx_warning_flags $CXXFLAGS"

AS_IF([test "x$with_mpi" != xno], [GA_MPICXX_TEST])

# Checks for C++ header files.

# Checks for C++ typedefs, structures, and compiler characteristics.

# Checks for C++ type sizes.

# Checks for C++ library functions.

ARMCI_CXX_OPT

CXXFLAGS="$ga_save_CXXFLAGS"

])

AC_LANG_POP([C++])

#########################################
# Assembler
#########################################
AC_MSG_NOTICE
AC_MSG_NOTICE([Assembler])
AC_MSG_NOTICE

GA_AS
AM_PROG_AS

ARMCI_TAS
ARMCI_X86COPY
ARMCI_AS_NO_I386ASM

#########################################
# Fortran compiler
#########################################
AC_MSG_NOTICE
AS_IF([test "x$enable_f77" = xyes],
    [AC_MSG_NOTICE([Fortran compiler])],
    [AC_MSG_NOTICE([Fortran compiler (disabled, but some tests still required)])])
AC_MSG_NOTICE
AC_LANG_PUSH([Fortran 77])

ga_save_FFLAGS="$FFLAGS"
GA_PROG_MPIF77
FFLAGS="$ga_save_FFLAGS"

# If Fortran 77 compiler was not found, we disable support for it.
AS_IF([test "x$F77" = x], [enable_f77=no])

# If Fortran 77 compiler does not work, we disable support for it.
AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([],[])], [],
    [AC_MSG_WARN([Fortran compiler does not work, disabling])
     enable_f77=no
     F77=])

GA_F77_DISABLE_RESULTS

# Perform Fortran 77 tests only if enabled.
if test "x$enable_f77" = xyes; then :

AS_IF([test x$with_mpi_wrappers = xyes], [GA_MPI_UNWRAP])

GA_COMPILER_VENDOR
GA_WARN_FLAGS
ga_save_FFLAGS="$FFLAGS"
FFLAGS="$ga_cv_f77_warning_flags $FFLAGS"

# If F77 is enabled, perform a simple link test for MPI.
AS_IF([test "x$with_mpi" != xno],
    [AS_IF([test "x$enable_mpi_tests" = xyes],
        [GA_MPIF77_TEST_LINK],
        [GA_MPIF77_TEST_COMPILE])])

GA_SUPPRESS_MESSAGE
GA_F77_LIBRARY_LDFLAGS
GA_F77_MAYBE_UNDERSCORING
AC_F77_DUMMY_MAIN
AC_F77_WRAPPERS
GA_F77_CPP_SYMBOL
GA_F77_FIXED
GA_F77_MISMATCH_TYPE
GA_F77_INTEGER_SIZE

# Checks for Fortran typedefs, structures, and compiler characteristics.

# Checks for Fortran type sizes.
ga_save_FFLAGS="$FFLAGS"; FFLAGS="$FFLAGS $FFLAG_INT"
GA_F77_CHECK_SIZEOF([INTEGER], [$ga_cv_f77_integer_size])
FFLAGS="$ga_save_FFLAGS"
GA_F77_CHECK_SIZEOF([REAL], [4])
GA_F77_CHECK_SIZEOF([DOUBLE PRECISION], [8])

# Checks for Fortran library functions.
GA_CHECK_FUNCS([dtime etime])
GA_F77_FLUSH

# The rest of the Fortran 77 to C compatibility checks.
GA_F2C_NOMAIN
GA_F2C_CMDARGS
GA_F2C_HIDDEN_STRING_LENGTH_CONVENTION
GA_F2C_MATCH_TYPES([INTEGER], [short, int, long, long long])
GA_F2C_MATCH_TYPES([REAL], [float, double, long double])
GA_F2C_MATCH_TYPES([DOUBLE PRECISION], [float, double, long double])

ARMCI_F77_OPT

FFLAGS="$ga_save_FFLAGS"

else

# enable_f77=no version of AC_F77_WRAPPERS
AC_DEFINE([F77_FUNC(name,NAME)], [name @%:@@%:@ _])
AC_DEFINE([F77_FUNC_(name,NAME)],[name @%:@@%:@ _])
# enable_f77=no version of GA_F77_CHECK_SIZEOF
AC_DEFINE_UNQUOTED(AS_TR_CPP(sizeof_f77_INTEGER), [$ac_cv_sizeof_voidp],
    [The size of 'INTEGER' as computed by C's sizeof.])
AC_DEFINE_UNQUOTED(AS_TR_CPP(sizeof_f77_REAL), [4],
    [The size of 'REAL' as computed by C's sizeof.])
AC_DEFINE_UNQUOTED(AS_TR_CPP(sizeof_f77_DOUBLE PRECISION), [8],
    [The size of 'DOUBLE PRECISION' as computed by C's sizeof.])
# enable_f77=no version of GA_F2C_NOMAIN
AC_SUBST([FLD_NOMAIN], [])
# enable_f77=no version of GA_F2C_CMDARGS
F2C_GETARG="F2C_GETARG"
F2C_IARGC="F2C_IARGC"
AC_SUBST([F2C_GETARG])
AC_SUBST([F2C_IARGC])
# enable_f77=no version of GA_F2C_HIDDEN_STRING_LENGTH_CONVENTION
AC_DEFINE([F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS], [1],
    [whether the hidden string length comes after all other args])
# enable_f77=no version of GA_F2C_MATCH_TYPES
AC_SUBST([F2C_INTEGER_C_TYPE], [long])
AC_SUBST([F2C_REAL_C_TYPE], [float])
AC_SUBST([F2C_DOUBLE_PRECISION_C_TYPE], [double])

fi

# This is required to override what automake does, if needed.
GA_F77_LD_OVERRIDE

AC_LANG_POP([Fortran 77])

###############################################################################
# Checks for BLAS/LAPACK/SCALAPACK require both C and Fortran features to
# have been detected. Further, they assume the current AC_LANG is C.
###############################################################################
AC_MSG_NOTICE
AC_MSG_NOTICE([Checks for BLAS,LAPACK])
AC_MSG_NOTICE

GA_BLAS
GA_LAPACK

###############################################################################
# Checks for programs.
###############################################################################
AC_MSG_NOTICE
AC_MSG_NOTICE([Checks for additional programs])
AC_MSG_NOTICE

AC_PROG_GREP
AC_PROG_SED

###############################################################################
# Libtool setup -- no compiler/linker tests after this
###############################################################################
AC_MSG_NOTICE
AC_MSG_NOTICE([Libtool setup])
AC_MSG_NOTICE
# temporarily restore unwrapped compilers
# this works around a bug where libtool sadly relies on matching compiler
# names in order to determine features (Fortran only, I think)
# libtool doesn't recognize MPI compiler names, nor should it
AS_IF([test x$with_mpi_wrappers = xyes], [GA_MPI_UNWRAP_PUSH])
GA_AR
LT_INIT([disable-shared])
# and now that that's over, put the MPI compilers back
# also, the above hack incorrectly sets the base compiler as the linker
AS_IF([test x$with_mpi_wrappers = xyes],
    [GA_MPI_UNWRAP_POP
     compiler="$CC"
     LTCC="$CC"
     lt_save_CC="$CC"
     compiler_DEFAULT="$CC"
     compiler_CXX="$CXX"
     compiler_F77="$F77"])

###############################################################################
# Remaining setup -- some tests, some individual components
###############################################################################
AC_MSG_NOTICE
AC_MSG_NOTICE([Miscellaneous remaining setup])
AC_MSG_NOTICE

# Establish some preprocessor symbols.
ARMCI_SETUP

# Set up TCGMSG.
TCGMSG_REMOTE_SHELL
TCGMSG_ENABLE_TIMINGS

# TODO not sure, but thus far required for 'dist' target at least
AM_CONDITIONAL([RANDOM_ACCESS], [test x != x]) # always false

# Whether to use PIC for the inline GCC code.
AM_CONDITIONAL([ENABLE_SHARED], [test x$enable_shared = xyes])

AS_IF([test "x$ga_cv_target" = xLINUX64],
    [AS_CASE([$host_cpu],
        [x86_64|ppc64|ia64],
            [AC_DEFINE([NEED_MEM_SYNC], [1], [Creates memfenc macro])])])
AS_IF([test "x$host_cpu" = xPWR4],
    [AC_DEFINE([NEED_MEM_SYNC], [1], [Creates memfenc macro])])

###############################################################################
# Test suite setup
###############################################################################
AC_ARG_VAR([NPROCS], [number of procs to use for parallel tests (default 4)])
AS_IF([test "x$NPROCS" = x], [NPROCS=4])
AC_SUBST([NPROCS])
AC_ARG_VAR([MPIEXEC],
           [how to run parallel tests if built with MPI e.g. "mpiexec -np %NP%"])
AS_CASE([$ga_msg_comms],
    [TCGMSG],  [TCGEXEC="`pwd`/parallel.x"],
    [TCGMSG5], [TCGEXEC=],
    [MPI|TCGMSGMPI],
        [AS_IF([test "x$MPIEXEC" = x],
            [AC_PATH_PROGS([MPIEXEC], [mpirun mpiexec])
             MPIEXEC="$MPIEXEC -n %NP%"])])
AC_SUBST([MPIEXEC])
AC_SUBST([TCGEXEC])

###############################################################################
# The End
###############################################################################
AC_CONFIG_FILES([Makefile
                 gaf2c/farg.h
                 gaf2c/typesf2c.h])
AC_OUTPUT

# Report on what we found.
AC_MSG_NOTICE([])
AC_MSG_NOTICE([**************************************************************])
AC_MSG_NOTICE([ $PACKAGE_NAME configured as follows:])
AC_MSG_NOTICE([**************************************************************])
AC_MSG_NOTICE([])
AC_MSG_NOTICE([                TARGET=$ga_cv_target])
AC_MSG_NOTICE([             MSG_COMMS=$ga_msg_comms])
AC_MSG_NOTICE([            GA_MP_LIBS=$GA_MP_LIBS])
AC_MSG_NOTICE([         GA_MP_LDFLAGS=$GA_MP_LDFLAGS])
AC_MSG_NOTICE([        GA_MP_CPPFLAGS=$GA_MP_CPPFLAGS])
AC_MSG_NOTICE([         ARMCI_NETWORK=$ga_armci_network])
AC_MSG_NOTICE([ ARMCI_NETWORK_LDFLAGS=$ARMCI_NETWORK_LDFLAGS])
AC_MSG_NOTICE([    ARMCI_NETWORK_LIBS=$ARMCI_NETWORK_LIBS])
AC_MSG_NOTICE([ARMCI_NETWORK_CPPFLAGS=$ARMCI_NETWORK_CPPFLAGS])
AS_IF([test "x$enable_f77" = xyes], [
AC_MSG_NOTICE([                   F77=$F77])
AS_IF([test "x$with_mpi_wrappers" = xyes], [
AC_MSG_NOTICE([         unwrapped F77=$ga_cv_mpif77_naked])
])
AC_MSG_NOTICE([                FFLAGS=$FFLAGS])
AC_MSG_NOTICE([             FFLAG_INT=$FFLAG_INT])
AC_MSG_NOTICE([            ARMCI_FOPT=$ARMCI_FOPT])
])
AC_MSG_NOTICE([                    CC=$CC])
AS_IF([test "x$with_mpi_wrappers" = xyes], [
AC_MSG_NOTICE([          unwrapped CC=$ga_cv_mpic_naked])
])
AC_MSG_NOTICE([                CFLAGS=$CFLAGS])
AC_MSG_NOTICE([            ARMCI_COPT=$ARMCI_COPT])
AS_IF([test "x$enable_cxx" = xyes], [
AC_MSG_NOTICE([                   CXX=$CXX])
AS_IF([test "x$with_mpi_wrappers" = xyes], [
AC_MSG_NOTICE([         unwrapped CXX=$ga_cv_mpicxx_naked])
])
AC_MSG_NOTICE([              CXXFLAGS=$CXXFLAGS])
AC_MSG_NOTICE([          ARMCI_CXXOPT=$ARMCI_CXXOPT])
])
AC_MSG_NOTICE([                   CPP=$CPP])
AC_MSG_NOTICE([              CPPFLAGS=$CPPFLAGS])
AC_MSG_NOTICE([               LDFLAGS=$LDFLAGS])
AC_MSG_NOTICE([                  LIBS=$LIBS])
AC_MSG_NOTICE([                 FLIBS=$FLIBS])
AC_MSG_NOTICE([          BLAS_LDFLAGS=$BLAS_LDFLAGS])
AC_MSG_NOTICE([             BLAS_LIBS=$BLAS_LIBS])
AC_MSG_NOTICE([         BLAS_CPPFLAGS=$BLAS_CPPFLAGS])
AC_MSG_NOTICE([                    AR=$AR])
AC_MSG_NOTICE([              AR_FLAGS=$AR_FLAGS])
AC_MSG_NOTICE([                  CCAS=$CCAS])
AC_MSG_NOTICE([            CCAS_FLAGS=$CCAS_FLAGS])
AC_MSG_NOTICE([                  DEFS=$DEFS])
AC_MSG_NOTICE([                 SHELL=$SHELL])
AS_CASE([$ga_msg_comms], [MPI|TCGMSGMPI], [
AC_MSG_NOTICE([               MPIEXEC=$MPIEXEC])
])
AC_MSG_NOTICE([                NPROCS=$NPROCS])
AC_MSG_NOTICE([])
