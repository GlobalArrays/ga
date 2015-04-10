# GA_ENABLE_UNIT_TESTS
# --------------------
AC_DEFUN([GA_ENABLE_UNIT_TESTS], [
AC_ARG_ENABLE([unit-tests],
    [AS_HELP_STRING([--enable-unit-tests],
        [build the unfinished unit tests])],
    [],
    [enable_unit_tests=no])
AM_CONDITIONAL([ENABLE_UNIT_TESTS], [test "x$enable_unit_tests" = xyes])
])dnl
