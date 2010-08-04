# GA_F77_CPP_SYMBOL([ACTION-WHEN-FOUND])
# --------------------------------------
# Detect how to pass CPP symbols to preprocessed Fortran 77.
# 
# Known:
#  -D       the usual
#  -WF,-D   IBM xlf
#  -Wp,-D   Fujitsu
#
AC_DEFUN([GA_F77_CPP_SYMBOL],
[AC_CACHE_CHECK([how to pass symbols to preprocessed $F77],
[ga_cv_f77_cpp_symbol],
[AC_LANG_PUSH([Fortran 77])
for symbol in -D -WF,-D -Wp,-D ; do
    ga_save_CPPFLAGS=$CPPFLAGS
    CPPFLAGS="$CPPFLAGS ${symbol}GABLAHBLAH"
    AC_LANG_CONFTEST(
[[#ifndef GABLAHBLAH
this is an error
#endif
      end program]])
    mv conftest.$ac_ext conftest.F
    AS_IF([$F77 $CPPFLAGS $FFLAGS -c conftest.F 1>&5 2>&5],
        [ga_cv_f77_cpp_symbol=$symbol])
    rm -f conftest.F conftest.$ac_objext
    CPPFLAGS=$ga_save_CPPFLAGS
    AS_IF([test "x$ga_cv_f77_cpp_symbol" != x], [break])
done
AC_LANG_POP([Fortran 77])
])
AS_IF([test "x$ga_cv_f77_cpp_symbol" = x],
    [AC_MSG_ERROR([don't know how to pass symbols to preprocessed Fortran])])
m4_default([$1],
    [AS_CASE([$ga_cv_f77_cpp_symbol],
        [-D],   [],
        [FFLAGS="$FFLAGS ${ga_cv_f77_cpp_symbol}HAVE_CONFIG_H"])])
]) # GA_F77_CPP_SYMBOL
