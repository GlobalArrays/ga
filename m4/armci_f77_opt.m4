# ARMCI_F77_OPT()
# ---------------
# Determine TARGET-/compiler-specific FFLAGS for optimization.
AC_DEFUN([ARMCI_F77_OPT], [
AC_REQUIRE([GA_TARGET64])
AC_REQUIRE([GA_ENABLE_OPT])
AC_REQUIRE([GA_ARMCI_NETWORK])
AC_ARG_VAR([ARMCI_FOPT], [ARMCI Fortran 77 optimization flags])
AC_CACHE_CHECK([for specific Fortran optimizations], [armci_cv_f77_opt], [
AS_IF([test "x$ARMCI_FOPT" != x], [armci_cv_f77_opt="$ARMCI_FOPT"], [armci_cv_f77_opt=])
AS_IF([test "x$armci_cv_f77_opt" = x && test "x$enable_opt" = xyes], [
AS_CASE([$ga_cv_target:$ga_cv_f77_compiler_vendor:$host_cpu:$ga_armci_network],
[BGQ:ibm:*:*],                  [armci_cv_f77_opt="-O3 -qstrict"],
[BGQ:gnu:*:*],                  [armci_cv_f77_opt="-O2"],
[CYGWIN:*:*:*],                 [armci_cv_f77_opt=],
[IBM64:*:*:*],                  [armci_cv_f77_opt=],
[IBM:*:*:*],                    [armci_cv_f77_opt="-O4 -qarch=auto -qstrict"],
[LINUX64:gnu:x86_64:*],         [armci_cv_f77_opt="-fstrength-reduce -mfpmath=sse"],
[LINUX64:ibm:powerpc64:*],      [armci_cv_f77_opt="-O4 -qarch=auto -qstrict"],
[LINUX64:ibm:ppc64:*],          [armci_cv_f77_opt="-O4 -qarch=auto -qstrict"],
[LINUX64:intel:x86_64:*],       [armci_cv_f77_opt="-O3 -w -cm -xW -tpp7"],
[LINUX64:pathscale:x86_64:*],   [armci_cv_f77_opt="-O3 -OPT:Ofast"],
[LINUX64:portland:x86_64:*],    [armci_cv_f77_opt="-fast -Mdalign -O3"],
[LINUX:gnu:686:*],              [armci_cv_f77_opt="-O3 -funroll-loops -march=pentiumpro -malign-double"],
[LINUX:gnu:686:MELLANOX],       [armci_cv_f77_opt="-O3 -funroll-loops -march=pentiumpro"],
[LINUX:gnu:686:OPENIB],         [armci_cv_f77_opt="-O3 -funroll-loops -march=pentiumpro"],
[LINUX:gnu:786:*],              [armci_cv_f77_opt="-O3 -funroll-loops -march=pentiumpro -malign-double"],
[LINUX:gnu:786:MELLANOX],       [armci_cv_f77_opt="-O3 -funroll-loops -march=pentiumpro"],
[LINUX:gnu:786:OPENIB],         [armci_cv_f77_opt="-O3 -funroll-loops -march=pentiumpro"],
[LINUX:gnu:x86:*],              [armci_cv_f77_opt="-O3 -funroll-loops -malign-double"],
[LINUX:gnu:x86:MELLANOX],       [armci_cv_f77_opt="-O3 -funroll-loops"],
[LINUX:gnu:x86:OPENIB],         [armci_cv_f77_opt="-O3 -funroll-loops"],
[LINUX:intel:686:*],            [armci_cv_f77_opt="-O4 -prefetch -unroll -ip -xK -tpp6"],
[LINUX:intel:786:*],            [armci_cv_f77_opt="-O4 -prefetch -unroll -ip -xW -tpp7"],
[LINUX:intel:*:*],              [armci_cv_f77_opt="-O4 -prefetch -unroll -ip"],
[LINUX:intel:k7:*],             [armci_cv_f77_opt="-O4 -prefetch -unroll -ip -xM"],
[LINUX:portland:686:*],         [armci_cv_f77_opt="-Mvect -Munroll -Mdalign -Minform,warn -Mnolist -Minfo=loop -Munixlogical -tp p6"],
[LINUX:portland:*:*],           [armci_cv_f77_opt="-Mvect -Munroll -Mdalign -Minform,warn -Mnolist -Minfo=loop -Munixlogical"],
[MACX64:intel:*:*],             [armci_cv_f77_opt="-O3 -prefetch -w -cm"],
[MACX:*:*:*],                   [armci_cv_f77_opt=],
[SOLARIS64:gnu:*:*],            [armci_cv_f77_opt="-dalign"],
[SOLARIS64:gnu:i386:*],         [armci_cv_f77_opt="-dalign -xarch=sse2"],
[SOLARIS:gnu:*:*],              [armci_cv_f77_opt="-dalign"],
[SOLARIS:gnu:i386:*],           [armci_cv_f77_opt="-dalign -xarch=sse2"],
                                [armci_cv_f77_opt=])
])])
AC_SUBST([ARMCI_FOPT],  [$armci_cv_f77_opt])
])dnl
