# TASCEL_F77_LIBRARY_LDFLAGS
# ----------------------
# Wrap AC_F77_LIBRARY_LDFLAGS in case user disables Fortran 77.
# Also, when mixing gcc and ifort, we sometimes need to add -lgcc_s to the
# FLIBS.
AC_DEFUN([TASCEL_F77_LIBRARY_LDFLAGS], [
AS_IF([test "x$enable_f77" = xyes],
    [TASCEL_MPI_UNWRAP_PUSH()
     AC_F77_LIBRARY_LDFLAGS
     TASCEL_MPI_UNWRAP_POP()
     AC_CACHE_CHECK([whether FLIBS needs -lgcc_s], [tascel_cv_flibs_gcc_s],
        [tascel_save_LIBS="$LIBS";  LIBS="$LIBS $FLIBS"
         tascel_save_FLIBS="$FLIBS"
         AC_LANG_PUSH([C])
         AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([],[])],
            [tascel_cv_flibs_gcc_s=no],
            [LIBS="$LIBS -lgcc_s"
             AC_LINK_IFELSE(
                [AC_LANG_PROGRAM([],[])],
                [FLIBS="$FLIBS -lgcc_s"
                 tascel_cv_flibs_gcc_s=yes],
                [AC_MSG_WARN([FLIBS does not work])
                 FLIBS="$tascel_save_FLIBS"
                 tascel_cv_flibs_gcc_s=no])])
         LIBS="$tascel_save_LIBS"
         AC_LANG_POP([C])])
    ])
])dnl
