# GA_GNU_STD_GNU17
# ---------------
# Add -std=gnu17 to the compiler flags if using
# GNU compilers to avoid -std=gnu23 issues
AC_DEFUN([GA_GNU_STD_GNU17], [
AC_PREREQ([2.69]) dnl for _AC_LANG_PREFIX
AC_CACHE_CHECK([for -std=gnu17 support in _AC_LANG compiler],
    ga_cv_[]_AC_LANG_ABBREV[]_gnu_std_gnu17, [
    ga_cv_[]_AC_LANG_ABBREV[]_gnu_std_gnu17=
    save_[]_AC_LANG_PREFIX[]FLAGS="$[]_AC_LANG_PREFIX[]FLAGS"
    []_AC_LANG_PREFIX[]FLAGS="$save_[]_AC_LANG_PREFIX[]FLAGS -std=gnu17"
    save_ac_[]_AC_LANG_ABBREV[]_werror_flag="$ac_[]_AC_LANG_ABBREV[]_werror_flag"
    AC_LANG_WERROR
    rm -f a.out
    touch a.out
    AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])],
        ga_cv_[]_AC_LANG_ABBREV[]_gnu_std_gnu17=-std=gnu17)
    []_AC_LANG_PREFIX[]FLAGS="$save_[]_AC_LANG_PREFIX[]FLAGS"
    ac_[]_AC_LANG_ABBREV[]_werror_flag="$save_ac_[]_AC_LANG_ABBREV[]_werror_flag"
    rm -f a.out
    ])
    AC_SUBST([]_AC_LANG_PREFIX[]FLAG_STD_GNU17, $ga_cv_[]_AC_LANG_ABBREV[]_gnu_std_gnu17)
])
