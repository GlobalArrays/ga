# GA_GNU_LOOP_VECT
# ---------------
# Add -fno-tree-slp-vectorize to the compiler flags if using
# GNU compilers.
AC_DEFUN([GA_GNU_LOOP_VECT], [
AC_PREREQ([2.69]) dnl for _AC_LANG_PREFIX
AC_CACHE_CHECK([for -fno-tree-slp-vectorize support in _AC_LANG compiler],
    ga_cv_[]_AC_LANG_ABBREV[]_gnu_loop_vect, [
    ga_cv_[]_AC_LANG_ABBREV[]_gnu_loop_vect=
    save_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
    []_AC_LANG_PREFIX[]FLAGS="$save_[]_AC_LANG_PREFIX[]FLAGS -fno-tree-slp-vectorize"
    save_ac_[]_AC_LANG_ABBREV[]_werror_flag="$ac_[]_AC_LANG_ABBREV[]_werror_flag"
    AC_LANG_WERROR
    rm -f a.out
    touch a.out
    AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])],
        ga_cv_[]_AC_LANG_ABBREV[]_gnu_loop_vect=-fno-tree-slp-vectorize)
    []_AC_LANG_PREFIX[]FLAGS="$save_[]_AC_LANG_PREFIX[]FLAGS"
    ac_[]_AC_LANG_ABBREV[]_werror_flag="$save_ac_[]_AC_LANG_ABBREV[]_werror_flag"
    rm -f a.out
    ])
    AC_SUBST([]_AC_LANG_PREFIX[]FLAG_NO_LOOP_VECT, $ga_cv_[]_AC_LANG_ABBREV[]_gnu_loop_vect)
])
