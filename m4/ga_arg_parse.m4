# GA_ARG_PARSE(ARG, VAR_LIBS, VAR_LDFLAGS, VAR_CPPFLAGS)
# ------------------------------------------------------
# Parse whitespace-separated ARG into appropriate LIBS, LDFLAGS, and
# CPPFLAGS variables.
AC_DEFUN([GA_ARG_PARSE],
[AC_COMPUTE_INT([ga_arg_parse_sizeof_voidp], [(long int) (sizeof (void*))])
for arg in $$1 ; do
    AS_CASE([$arg],
        [yes],          [],
        [no],           [],
        [-l*],          [$2="$$2 $arg"],
        [-L*],          [$3="$$3 $arg"],
        [-WL*],         [$3="$$3 $arg"],
        [-Wl*],         [$3="$$3 $arg"],
        [-I*],          [$4="$$4 $arg"],
        [*.a],          [$2="$$2 $arg"],
        [*.so],         [$2="$$2 $arg"],
        [*lib],         [AS_IF([test -d $arg], [$3="$$3 -L$arg"],
                            [AC_MSG_WARN([$arg of $1 not parsed])])],
        [*lib64],       [AS_IF([test -d $arg], [$3="$$3 -L$arg"],
                            [AC_MSG_WARN([$arg of $1 not parsed])])],
        [*include],     [AS_IF([test -d $arg], [$4="$$4 -I$arg"],
                            [AC_MSG_WARN([$arg of $1 not parsed])])],
        [*include64],   [AS_IF([test -d $arg], [$4="$$4 -I$arg"],
                            [AC_MSG_WARN([$arg of $1 not parsed])])],
        [ga_arg_parse_ok=no
         AS_IF([test "x$ga_arg_parse_sizeof_voidp" = x8],
            [AS_IF([test -d $arg/lib64],    [$3="$$3 -L$arg/lib64"; ga_arg_parse_ok=yes],
                   [test -d $arg/lib],      [$3="$$3 -L$arg/lib"; ga_arg_parse_ok=yes])
             AS_IF([test -d $arg/include64],[$4="$$4 -I$arg/include64"; ga_arg_parse_ok=yes],
                   [test -d $arg/include],  [$4="$$4 -I$arg/include"; ga_arg_parse_ok=yes])],
            [AS_IF([test -d $arg/lib],      [$3="$$3 -L$arg/lib"; ga_arg_parse_ok=yes])
             AS_IF([test -d $arg/include],  [$4="$$4 -I$arg/include"; ga_arg_parse_ok=yes])])])
done])dnl
