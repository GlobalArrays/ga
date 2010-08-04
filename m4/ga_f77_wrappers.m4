# GA_F77_WRAPPERS
# -----------------
# Wrap AC_F77_WRAPPERS in case user disables Fortran 77.
AC_DEFUN([GA_F77_WRAPPERS], [
AS_IF([test "x$enable_f77" = xyes],
    [AC_F77_WRAPPERS],
    [AC_DEFINE([F77_FUNC(name,NAME)], [name])
     AC_DEFINE([F77_FUNC_(name,NAME)],[name [##] _])])
])dnl
