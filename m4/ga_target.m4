# GA_TARGET()
# -----------
# Attempt to determine the old TARGET variable automatically.

AC_DEFUN([GA_TARGET],
[# AH_TEMPLATE for all known TARGETs
AH_TEMPLATE([CYGNUS],       [Define to 1 on Cygnus systems])
AH_TEMPLATE([CYGWIN],       [Define to 1 on Cygwin systems])
AH_TEMPLATE([IBM],          [Define to 1 on IBM SP systems])
AH_TEMPLATE([IBM64],        [Define to 1 on 64bit IBM SP systems])
AH_TEMPLATE([LINUX],        [Define to 1 on generic Linux systems])
AH_TEMPLATE([LINUX64],      [Define to 1 on generic 64bit Linux systems])
AH_TEMPLATE([MACX],         [Define to 1 on OSX systems])
AH_TEMPLATE([MACX64],       [Define to 1 on 64bit OSX systems])
AH_TEMPLATE([SOLARIS],      [Define to 1 on Solaris systems])
AH_TEMPLATE([SOLARIS64],    [Define to 1 on 64bit Solaris systems])
AC_REQUIRE([AC_CANONICAL_BUILD])
AC_REQUIRE([AC_CANONICAL_HOST])
AC_CACHE_CHECK([for TARGET base (64bit-ness checked later)],
[ga_cv_target_base],
[ga_cv_target_base=UNKNOWN
AS_IF([test "x$ga_cv_target_base" = xUNKNOWN],
    [AS_CASE([$host],
        [*cygwin*],         [ga_cv_target_base=CYGWIN],
        [*ibm*],            [ga_cv_target_base=IBM],
        [*linux*],          [ga_cv_target_base=LINUX],
        [*darwin*],         [ga_cv_target_base=MACX],
        [*apple*],          [ga_cv_target_base=MACX],
        [*mingw32*],        [ga_cv_target_base=MINGW],
        [*solaris*],        [ga_cv_target_base=SOLARIS])])
])dnl
AC_DEFINE_UNQUOTED([$ga_cv_target_base], [1],
    [define if this is the TARGET irregardless of whether it is 32/64 bits])
# A horrible hack that should go away somehow...
dnl # Only perform this hack for ARMCI build.
dnl AS_IF([test "x$ARMCI_TOP_BUILDDIR" != x], [
    AC_CACHE_CHECK([whether we think this system is what we call SYSV],
    [ga_cv_sysv],
    [AS_CASE([$ga_cv_target_base],
        [SUN|SOLARIS|IBM|LINUX],
            [ga_cv_sysv=yes],
        [ga_cv_sysv=no])
    ])
    AS_IF([test x$ga_cv_sysv = xyes],
        [AC_DEFINE([SYSV], [1],
            [Define if we want this system to use SYSV shared memory])])
dnl ])
# Hopefully these will never be used and we can remove them soon.
AM_CONDITIONAL([CYGNUS],       [test "$ga_cv_target_base" = CYGNUS])
AM_CONDITIONAL([CYGWIN],       [test "$ga_cv_target_base" = CYGWIN])
AM_CONDITIONAL([IBM],          [test "$ga_cv_target_base" = IBM])
AM_CONDITIONAL([LINUX],        [test "$ga_cv_target_base" = LINUX])
AM_CONDITIONAL([MACX],         [test "$ga_cv_target_base" = MACX])
AM_CONDITIONAL([MINGW],        [test "$ga_cv_target_base" = MINGW])
AM_CONDITIONAL([SOLARIS],      [test "$ga_cv_target_base" = SOLARIS])
])dnl


# GA_TARGET64()
# -------------
# Checking for 64bit platforms requires checking sizeof void*.
# That's easy, but doing it too soon causes AC_PROG_F77/C/CXX to get expanded
# too soon, and we want to expand those with a better list of compilers
# based on our current TARGET. Therefore, we must do this 64bit test later.
AC_DEFUN([GA_TARGET64],
[AC_REQUIRE([GA_TARGET])
AC_COMPUTE_INT([ga_target64_sizeof_voidp],
    [(long int) (sizeof (void*))])
AC_CACHE_CHECK([for TARGET 64bit-ness], [ga_cv_target],
[AS_IF([test x$ga_target64_sizeof_voidp = x8],
    [ga_cv_target=${ga_cv_target_base}64],
    [ga_cv_target=$ga_cv_target_base])])
AC_DEFINE_UNQUOTED([$ga_cv_target], [1],
    [define if this is the TARGET, 64bit-specific])
])dnl
