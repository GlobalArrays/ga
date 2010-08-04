# ARMCI_CXX_OPT()
# ---------------
# Determine TARGET-/compiler-specific CXXFLAGS and FFLAGS for optimization.
AC_DEFUN([ARMCI_CXX_OPT], [
AC_REQUIRE([GA_TARGET64])
AC_REQUIRE([GA_ENABLE_OPT])
AC_REQUIRE([GA_ARMCI_NETWORK])
AC_CACHE_CHECK([for specific C++ optimizations], [armci_cv_cxx_opt],
[AS_IF([test x$enable_opt = xno], [armci_cv_cxx_opt="-O0"],
[AS_CASE([$ga_cv_target:$ax_cv_cxx_compiler_vendor:$host_cpu:$ga_armci_network],
[LINUX:*:*:*],              [armci_cv_cxx_opt="-O0"],
                            [armci_cv_cxx_opt="-O0"])])])
AC_SUBST([ARMCI_CXXOPT],    [$armci_cv_cxx_opt])
])dnl
