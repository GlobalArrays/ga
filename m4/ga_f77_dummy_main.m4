# GA_F77_DUMMY_MAIN
# -----------------
# Wrap AC_F77_DUMMY_MAIN in case user disables Fortran 77.
AC_DEFUN([GA_F77_DUMMY_MAIN], [
AS_IF([test "x$enable_f77" = xyes], [AC_F77_DUMMY_MAIN])
])dnl
