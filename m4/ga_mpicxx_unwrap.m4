# GA_MPICXX_UNWRAP()
# ------------------
# Attempt to unwrap MPI C compiler and determine the underlying compiler.
# The attempt here is only really needed because some tests (notably those by
# libtool) tend to match against the compiler name in some corner cases and
# gets it wrong.
#
# The strategy is to compare the (stdout) outputs of the version flags using
# a custom perl script.
AC_DEFUN([GA_MPICXX_UNWRAP], [
AC_CACHE_CHECK([for base $MPICXX compiler], [ga_cv_mpicxx_naked], [
versions="--version -v -V -qversion"
inside="$GA_TOP_SRCDIR/build-aux/inside.pl"
AS_CASE([$MPICXX],
    [*_r],  [compilers="bgxlC_r"],
    [*],    [compilers="g++ c++ icpc pgCC pathCC gpp aCC cxx cc++ cl.exe FCC KCC RCC bgxlC_r bgxlC xlC_r xlC CC"])
# Try separating stdout and stderr. Only compare stdout.
AS_IF([test "x$ga_cv_mpicxx_naked" = x], [
for version in $versions; do
    for naked_compiler in $compilers; do
        rm -f mpi.txt mpi.err naked.txt naked.err
        AS_IF([$MPICXX $version 1>mpi.txt 2>mpi.err],
            [AS_IF([$naked_compiler $version 1>naked.txt 2>naked.err],
                [AS_IF([$inside mpi.txt naked.txt >/dev/null],
                    [ga_cv_mpicxx_naked=$naked_compiler; break],
                    [echo "inside.pl failed, skipping" >&AS_MESSAGE_LOG_FD])],
                [echo "$naked_compiler $version failed, skipping" >&AS_MESSAGE_LOG_FD])],
            [echo "$MPICXX $version failed, skipping" >&AS_MESSAGE_LOG_FD])
    done
    AS_IF([test "x$ga_cv_mpicxx_naked" != x], [break])
done
])
# Try by combining stdout/err into one file.
AS_IF([test "x$ga_cv_mpicxx_naked" = x], [
for version in $versions; do
    for naked_compiler in $compilers; do
        rm -f mpi.txt naked.txt
        AS_IF([$MPICXX $version 1>mpi.txt 2>&1],
            [AS_IF([$naked_compiler $version 1>naked.txt 2>&1],
                [AS_IF([$inside mpi.txt naked.txt >/dev/null],
                    [ga_cv_mpicxx_naked=$naked_compiler; break],
                    [echo "inside.pl failed, skipping" >&AS_MESSAGE_LOG_FD])],
                [echo "$naked_compiler $version failed, skipping" >&AS_MESSAGE_LOG_FD])],
            [echo "$MPICXX $version failed, skipping" >&AS_MESSAGE_LOG_FD])
    done
    AS_IF([test "x$ga_cv_mpicxx_naked" != x], [break])
done
])
rm -f mpi.txt mpi.err naked.txt naked.err
AS_IF([test "x$ga_cv_mpicxx_naked" = x], [ga_cv_mpicxx_naked=error])
])
AS_IF([test "x$ga_cv_mpicxx_naked" = xerror],
    [AC_MSG_WARN([Could not determine the C++ compiler wrapped by MPI])
     AC_MSG_WARN([This is usually okay])])
])dnl
