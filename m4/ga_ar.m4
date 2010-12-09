# GA_AR
# -----
# Libtool doesn't advertise AR nor AR_FLAGS in case the user wishes to
# override them. Further, certain systems require a different archiver.
# Use this prior to LT_INIT.
#
# Known archivers:
# ar    - all known systems
# sxar  - special to NEC/NEC64
#
AC_DEFUN([GA_AR], [
AC_ARG_VAR([AR], [archiver used by libtool (default: ar)])
AC_ARG_VAR([AR_FLAGS], [archiver flags used by libtool (default: cru)])
AS_IF([test "x$AR" = x],
    [AS_CASE([$ga_cv_target], [NEC|NEC64], [AR=sxar])])
])dnl
