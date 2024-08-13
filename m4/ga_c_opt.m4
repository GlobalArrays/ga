# GA_C_OPT()
# ----------
# Determine TARGET-/compiler-specific CFLAGS for optimization.
AC_DEFUN([GA_C_OPT], [
AC_REQUIRE([GA_TARGET64])
AC_REQUIRE([GA_ENABLE_OPT])
AC_ARG_VAR([GA_COPT], [GA C optimization flags])
AC_CACHE_CHECK([for specific C optimizations], [ga_cv_c_opt], [
AS_IF([test "x$GA_COPT" != x], [ga_cv_c_opt="$GA_COPT"], [ga_cv_c_opt=])
AS_IF([test "x$ga_cv_c_opt" = x && test "x$enable_opt" = xyes], [
AS_CASE([$ga_cv_target:$ga_cv_c_compiler_vendor:$host_cpu],
[CYGWIN:*:*],               [ga_cv_c_opt=],
[IBM64:*:*],                [ga_cv_c_opt=],
[IBM:*:*],                  [ga_cv_c_opt=],
[LINUX64:fujitsu:x86_64],   [ga_cv_c_opt="-Kfast"],
[LINUX64:gnu:powerpc64],    [ga_cv_c_opt="-funroll-loops"],
[LINUX64:gnu:ppc64],        [ga_cv_c_opt="-funroll-loops"],
[LINUX64:gnu:x86_64],       [ga_cv_c_opt="-O2 -funroll-loops"],
[LINUX64:ibm:powerpc64],    [ga_cv_c_opt="-qinline=100 -qstrict -qarch=auto -qtune=auto"],
[LINUX64:ibm:ppc64],        [ga_cv_c_opt="-qinline=100 -qstrict -qarch=auto -qtune=auto"],
[LINUX64:ibm:x86_64],       [ga_cv_c_opt=],
[LINUX:fujitsu:*],          [ga_cv_c_opt="-Kfast"],
[LINUX:gnu:786],            [ga_cv_c_opt="-O2 -funroll-loops -malign-double"],
[LINUX:gnu:*],              [ga_cv_c_opt="-O2 -funroll-loops"],
[LINUX:gnu:x86],            [ga_cv_c_opt="-O2 -funroll-loops -malign-double"],
[LINUX:ibm:*],              [ga_cv_c_opt="-q32"],
[LINUX:intel:*],            [ga_cv_c_opt="-O3 -prefetch"],
[MACX64:*:*],               [ga_cv_c_opt=],
[MACX:*:*],                 [ga_cv_c_opt=],
                            [ga_cv_c_opt=])
])])
AC_SUBST([GA_COPT], [$ga_cv_c_opt])
])dnl
