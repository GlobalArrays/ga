# COMEX_AR
# -----
# Libtool doesn't advertise AR nor AR_FLAGS in case the user wishes to
# override them. Further, certain systems require a different archiver.
# RANLIB may also be affected.
# Use this prior to LT_INIT.
#
# Known archivers:
# ar    - all known systems
#
AC_DEFUN([COMEX_AR], [
AC_ARG_VAR([AR], [archiver used by libtool (default: ar)])
AC_ARG_VAR([AR_FLAGS], [archiver flags used by libtool (default: cru)])
AC_ARG_VAR([RANLIB], [generates index to archive (default: ranlib)])
])dnl
