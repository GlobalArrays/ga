# GA_GNU_LOOP_OPT
# ---------------
# Add -fno-aggressive-loop-optimizations to the compiler flags if using
# GNU compilers.
AC_DEFUN([GA_GNU_LOOP_OPT], [
AC_PREREQ([2.69]) dnl for _AC_LANG_PREFIX
AC_CACHE_CHECK([for -fno-aggressive-loop-optimizations support in _AC_LANG compiler],
    ga_cv_[]_AC_LANG_ABBREV[]_gnu_loop_opt, [
    ga_cv_[]_AC_LANG_ABBREV[]_gnu_loop_opt=
    save_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
    []_AC_LANG_PREFIX[]FLAGS="$save_[]_AC_LANG_PREFIX[]FLAGS -fno-aggressive-loop-optimizations"
    save_ac_[]_AC_LANG_ABBREV[]_werror_flag="$ac_[]_AC_LANG_ABBREV[]_werror_flag"
    AC_LANG_WERROR
    rm -f a.out
    touch a.out
    AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])],
        ga_cv_[]_AC_LANG_ABBREV[]_gnu_loop_opt=-fno-aggressive-loop-optimizations)
    []_AC_LANG_PREFIX[]FLAGS="$save_[]_AC_LANG_PREFIX[]FLAGS"
    ac_[]_AC_LANG_ABBREV[]_werror_flag="$save_ac_[]_AC_LANG_ABBREV[]_werror_flag"
    rm -f a.out
    ])
    AC_SUBST([]_AC_LANG_PREFIX[]FLAG_NO_LOOP_OPT, $ga_cv_[]_AC_LANG_ABBREV[]_gnu_loop_opt)
])
