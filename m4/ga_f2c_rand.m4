# GA_F2C_RAND
# -----------
# In mixed Fortran/C code, if the C code has implemented a 'rand' function for
# use by Fortran, it may conflict with an existing symbol. This is the fault of
# the API being implemented, but in our case for backwards compatibility it
# can't be avoided...
AC_DEFUN([GA_F2C_SRAND48],
[AC_CACHE_CHECK([whether we can safely implement F77_FUNC(srand48)],
    [ga_cv_f2c_srand48],
    [AC_LANG_PUSH([C])
    AC_COMPILE_IFELSE(
        [AC_LANG_PROGRAM(
[[
#include <stdlib.h>
#define _SRAND48_ F77_FUNC(srand48, SRAND48)
void _SRAND48_(long *seed)
{
    unsigned int aseed = *seed;
    srandom(aseed);
}
]],
[[
long seed=6;
_SRAND48_(&seed);
]]
)],
        [ga_cv_f2c_srand48=yes],
        [ga_cv_f2c_srand48=no])
    AC_LANG_POP([C])])
AS_IF([test "x$ga_cv_f2c_srand48" = xyes], [val=1], [val=0])
AC_DEFINE_UNQUOTED([F2C_SRAND48_OK], [$val],
        [define to 1 if Fortran-callable srand48 does not conflict with system def])
]) # GA_F2C_SRAND48
