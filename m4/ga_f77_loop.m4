# GA_F77_LOOP
# ------------
# enables -fno-tree-slp-vectorize to avoid failures in global/testing/test.F
# since gfortran 12 uses -ftree-vectorize at -O2 level
# https://github.com/GlobalArrays/ga/issues/249
#
AC_DEFUN([GA_F77_LOOP], [
AC_CACHE_CHECK([whether $F77 needs a flag to disable fortran loop vectorization],
[ga_cv_f77_loop_flag],
[AC_LANG_PUSH([Fortran 77])
for testflag in -fno-tree-slp-vectorize; do
    ga_save_FFLAGS=$FFLAGS
    AS_IF([test "x$testflag" != xnone], [FFLAGS="$FFLAGS $testflag"])
    AC_COMPILE_IFELSE([AC_LANG_SOURCE(
[[c some comment
      end program]])],
        [ga_cv_f77_loop_flag=$testflag])
    FFLAGS=$ga_save_FFLAGS
    AS_IF([test "x$ga_cv_f77_loop_flag" != x], [break])
done
AC_LANG_POP([Fortran 77])])
AS_IF([test "x$ga_cv_f77_loop_flag" != xnone],
    [AS_IF([test "x$ga_cv_f77_loop_flag" != x],
        [FFLAGS="$FFLAGS $ga_cv_f77_loop_flag"])])
]) # GA_F77_LOOP
