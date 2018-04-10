# GA_F77_COMPUTE_SIZEOF(TYPE, VARIABLE)
# -------------------------------------
AC_DEFUN([GA_F77_COMPUTE_SIZEOF], [
AS_TR_SH([$2])=no
AC_F77_FUNC([size])
AC_F77_FUNC([fsub])
AC_LANG_PUSH([Fortran 77])
AC_COMPILE_IFELSE([AC_LANG_SOURCE(
[[      subroutine fsub
      external size
      $1 x(2)
      call size(x(1),x(2))
      end]])], [mv conftest.$ac_objext cfortran_test.$ac_objext
    ga_save_LIBS=$LIBS
    LIBS="cfortran_test.$ac_objext $LIBS $[]_AC_LANG_PREFIX[]LIBS"
    AC_LANG_PUSH([C])
    AC_RUN_IFELSE([AC_LANG_SOURCE(
[[#include <stdio.h>
#include <stdlib.h>
#ifdef __cplusplus
extern "C" {
#endif
void $size(char *a, char *b)
{
    long diff = (long) (b - a);
    FILE *f=fopen("conftestval", "w");
    if (!f) exit(1);
    fprintf(f, "%ld\n", diff);
    fclose(f);
}
#ifdef __cplusplus
}
#endif
#ifdef __cplusplus
extern "C"
#else
extern
#endif
void $fsub(void);
int main(int argc, char **argv)
{
    $fsub();
    return 0;
}
]])], [AS_TR_SH([$2])=`cat conftestval`])
    LIBS=$ga_save_LIBS
    rm -f conftest* cfortran_test*
    AC_LANG_POP([C])])
AC_LANG_POP([Fortran 77])
]) # GA_F77_COMPUTE_SIZEOF

# GA_F77_CHECK_SIZEOF(TYPE, CROSS-SIZE)
# -------------------------------------
AC_DEFUN([GA_F77_CHECK_SIZEOF],
[AS_VAR_PUSHDEF([type_var], [ga_cv_f77_sizeof_$1])
AC_CACHE_CHECK([size of Fortran $1], type_var,
    [AS_IF([test x$cross_compiling = xyes],
        [AS_VAR_SET(type_var, [$2])],
        [GA_F77_COMPUTE_SIZEOF([$1], type_var)])])
AS_IF([test x$cross_compiling = xyes],
    [AC_MSG_WARN([Cannot determine size of $1 when cross-compiling.])
     AC_MSG_WARN([Defaulting to $2])])
AC_DEFINE_UNQUOTED(AS_TR_CPP(sizeof_f77_$1), $type_var,
    [The size of '$1' as computed by C's sizeof.])
AS_VAR_POPDEF([type_var])
])# GA_F77_SIZEOF
