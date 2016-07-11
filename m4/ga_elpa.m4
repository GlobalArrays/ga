# GA_F77_ELPA_TEST
# ----------------
# Generate Fortran 77 conftest for ELPA.
AC_DEFUN([GA_F77_ELPA_TEST], [AC_LANG_CONFTEST([AC_LANG_PROGRAM([],
[[      use ELPA1
      implicit none
      logical status
      integer i4
      double precision dscal8,darray8(2)
      status = SOLVE_EVP_REAL(i4,i4,darray8,i4,
     C     darray8,darray8,i4,i4,i4,i4)]])])
])


# GA_C_ELPA_TEST
# --------------
# Generate C conftest for ELPA.
AC_DEFUN([GA_C_ELPA_TEST], [AC_LANG_CONFTEST([AC_LANG_PROGRAM(
[#ifdef __cplusplus
extern "C" {
#endif
char solve_evp_real ();
#ifdef __cplusplus
}
#endif
],
[[char result = solve_evp_real ();
]])])
])

# GA_RUN_ELPA_TEST
# ----------------
# Test the linker.
# Clears ELPA_LIBS on failure.  Sets ga_elpa_ok=yes on success.
AC_DEFUN([GA_RUN_ELPA_TEST], [
AS_IF([test "x$enable_f77" = xno],
   [AC_LANG_PUSH([C])
    GA_C_ELPA_TEST()
    AC_LINK_IFELSE([], [ga_elpa_ok=yes], [ELPA_LIBS=])
    AC_LANG_POP([C])],
   [AC_LANG_PUSH([Fortran 77])
    GA_F77_ELPA_TEST()
    AC_LINK_IFELSE([], [ga_elpa_ok=yes], [ELPA_LIBS=])
    AC_LANG_POP([Fortran 77])])
])dnl

# GA_ELPA([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
# -------------------------------------------------
# Modeled after GA_SCALAPACK.
# Tries to find ELPA library.
# ELPA depends on SCALAPACK, LAPACK, and BLAS.
AC_DEFUN([GA_ELPA],
[AC_REQUIRE([GA_SCALAPACK])
elpa_size=4
AC_ARG_WITH([elpa],
    [AS_HELP_STRING([--with-elpa=[[ARG]]],
        [use ELPA library compiled with sizeof(INTEGER)==4])],
    [elpa_size=4])
AC_ARG_WITH([elpa8],
    [AS_HELP_STRING([--with-elpa8=[[ARG]]],
        [use ELPA library compiled with sizeof(INTEGER)==8])],
    [elpa_size=8; with_elpa="$with_elpa8"])

ga_elpa_ok=no
AS_IF([test "x$with_elpa" = xno], [ga_elpa_ok=skip])

# Parse --with-elpa argument. Clear previous values first.
ELPA_LIBS=
ELPA_LDFLAGS=
ELPA_CPPFLAGS=
GA_ARG_PARSE([with_elpa],
    [ELPA_LIBS], [ELPA_LDFLAGS], [ELPA_CPPFLAGS])

ga_save_LIBS="$LIBS"
ga_save_LDFLAGS="$LDFLAGS"
ga_save_CPPFLAGS="$CPPFLAGS"

LDFLAGS="$ELPA_LDFLAGS $SCALAPACK_LDFLAGS $LAPACK_LDFLAGS $BLAS_LDFLAGS $GA_MP_LDFLAGS $LDFLAGS"
CPPFLAGS="$ELPA_CPPFLAGS $SCALAPACK_CPPFLAGS $LAPACK_CPPFLAGS $BLAS_CPPFLAGS $GA_MP_CPPFLAGS $CPPFLAGS"

# ELPA fortran test uses a module and needs CPPFLAGS
# but CPPFLAGS isn't used with *.f non-preprocessed extension
ga_save_FFLAGS="$FFLAGS"
FFLAGS="$ELPA_CPPFLAGS $FFLAGS"

AC_MSG_NOTICE([Attempting to locate ELPA library])

# First, check environment/command-line variables.
# If failed, erase ELPA_LIBS but maintain ELPA_LDFLAGS and
# ELPA_CPPFLAGS.
AS_IF([test $ga_elpa_ok = no],
    [AC_MSG_CHECKING([for ELPA with user-supplied flags])
     LIBS="$ELPA_LIBS $SCALAPACK_LIBS $LAPACK_LIBS $BLAS_LIBS $GA_MP_LIBS $LIBS"
     GA_RUN_ELPA_TEST()
     LIBS="$ga_save_LIBS"
     AC_MSG_RESULT([$ga_elpa_ok])])

# Generic ELPA library?
AS_IF([test $ga_elpa_ok = no],
    [AC_MSG_CHECKING([for ELPA in generic library])
     ELPA_LIBS="-lelpa"
     LIBS="$ELPA_LIBS $SCALAPACK_LIBS $LAPACK_LIBS $BLAS_LIBS $GA_MP_LIBS $LIBS"
     GA_RUN_ELPA_TEST()
     LIBS="$ga_save_LIBS"
     AC_MSG_RESULT([$ga_elpa_ok])])

CPPFLAGS="$ga_save_CPPFLAGS"
LDFLAGS="$ga_save_LDFLAGS"
FFLAGS="$ga_save_FFLAGS"

AC_SUBST([ELPA_LIBS])
AC_SUBST([ELPA_LDFLAGS])
AC_SUBST([ELPA_CPPFLAGS])
AS_IF([test "x$elpa_size" = x8],
    [AC_DEFINE([ELPA_I8], [1], [ELPA is using 8-byte integers])])

# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
AS_IF([test $ga_elpa_ok = yes],
    [have_elpa=1
     $1],
    [AC_MSG_WARN([ELPA library not found, interfaces won't be defined])
     have_elpa=0
     $2])
AC_DEFINE_UNQUOTED([HAVE_ELPA], [$have_elpa],
    [Define to 1 if you have ELPA library.])
AM_CONDITIONAL([HAVE_ELPA], [test $ga_elpa_ok = yes])
])dnl GA_ELPA
