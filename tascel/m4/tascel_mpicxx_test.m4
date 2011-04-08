# TASCEL_MPICXX_TEST
# --------------
# Attempt to compile a simple MPI program in C++.
#
AC_DEFUN([TASCEL_MPICXX_TEST], [
AS_IF([test "x$with_mpi" != xno], [
    AC_LANG_PUSH([C++])
    AC_CACHE_CHECK([whether a simple C++ MPI program works],
        [tascel_cv_cxx_mpi_test], [
        AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([[#include <mpi.h>]],
[[int myargc; char **myargv; MPI_Init(&myargc, &myargv); MPI_Finalize();]])],
            [tascel_cv_cxx_mpi_test=yes],
            [tascel_cv_cxx_mpi_test=no])
# That didn't work, so now let's try with our TASCEL_MP_* flags.
        AS_IF([test "x$tascel_cv_cxx_mpi_test" = xno], [
        tascel_save_LIBS="$LIBS";           LIBS="$LIBS $TASCEL_MP_LIBS"
        tascel_save_CPPFLAGS="$CPPFLAGS";   CPPFLAGS="$CPPFLAGS $TASCEL_MP_CPPFLAGS"
        tascel_save_LDFLAGS="$LDFLAGS";     LDFLAGS="$LDFLAGS $TASCEL_MP_LDFLAGS"
        AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([[#include <mpi.h>]],
[[int myargc; char **myargv; MPI_Init(&myargc, &myargv); MPI_Finalize();]])],
            [tascel_cv_cxx_mpi_test=yes],
            [tascel_cv_cxx_mpi_test=no])
        LIBS="$tascel_save_LIBS"
        CPPFLAGS="$tascel_save_CPPFLAGS"
        LDFLAGS="$tascel_save_LDFLAGS"
        ])
# That didn't work, so now let's try with our TASCEL_MP_* flags and various libs.
        AS_IF([test "x$tascel_cv_cxx_mpi_test" = xno], [
        for lib in -lmpi -lmpich; do
        tascel_save_LIBS="$LIBS";           LIBS="$LIBS $TASCEL_MP_LIBS $lib"
        tascel_save_CPPFLAGS="$CPPFLAGS";   CPPFLAGS="$CPPFLAGS $TASCEL_MP_CPPFLAGS"
        tascel_save_LDFLAGS="$LDFLAGS";     LDFLAGS="$LDFLAGS $TASCEL_MP_LDFLAGS"
        AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([[#include <mpi.h>]],
[[int myargc; char **myargv; MPI_Init(&myargc, &myargv); MPI_Finalize();]])],
            [tascel_cv_cxx_mpi_test=$lib; break],
            [tascel_cv_cxx_mpi_test=no])
        LIBS="$tascel_save_LIBS"
        CPPFLAGS="$tascel_save_CPPFLAGS"
        LDFLAGS="$tascel_save_LDFLAGS"
        done
        LIBS="$tascel_save_LIBS"
        CPPFLAGS="$tascel_save_CPPFLAGS"
        LDFLAGS="$tascel_save_LDFLAGS"
        ])
        ])
    AC_LANG_POP([C++])
    AS_CASE([$tascel_cv_cxx_mpi_test],
        [yes],  [],
        [no],   [AC_MSG_FAILURE([could not link simple C++ MPI program])],
        [*],    [TASCEL_MP_LIBS="$tascel_cv_cxx_mpi_test"],
        [])
])
])dnl
