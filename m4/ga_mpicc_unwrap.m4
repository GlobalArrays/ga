# GA_MPICC_UNWRAP()
# -----------------
# Attempt to unwrap MPI C compiler and determine the underlying compiler.
# The attempt here is only really needed because some tests (notably those by
# libtool) tend to match against the compiler name in some corner cases and
# gets it wrong.
#
# The strategy is to compare the (stdout) outputs of the version flags using
# diff.
AC_DEFUN([GA_MPICC_UNWRAP], [
AC_CACHE_CHECK([for base $MPICC compiler], [ga_cv_mpicc_naked], [
versions="--version -v -V -qversion"
AS_CASE([$MPICC],
    [*_r],  [compilers="bgxlc_r xlc_r"],
    [*],    [compilers="bgxlc xlc gcc pgcc pathcc icc ecc cl ccc cc"])
# Try separating stdout and stderr. Only compare stdout.
AS_IF([test "x$ga_cv_mpicc_naked" = x], [
for version in $versions; do
    for naked_compiler in $compilers; do
        rm -f mpi.txt mpi.err naked.txt naked.err
        AS_IF([$MPICC $version 1>mpi.txt 2>mpi.err],
            [AS_IF([$naked_compiler $version 1>naked.txt 2>naked.err],
                [AS_IF([diff mpi.txt naked.txt >/dev/null],
                    [ga_cv_mpicc_naked=$naked_compiler; break],
                    [echo "diff failed, skipping" >&AS_MESSAGE_LOG_FD])],
                [echo "$naked_compiler $version failed, skipping" >&AS_MESSAGE_LOG_FD])],
            [echo "$MPICC $version failed, skipping" >&AS_MESSAGE_LOG_FD])
    done
    AS_IF([test "x$ga_cv_mpicc_naked" != x], [break])
done
])
AS_IF([test "x$ga_cv_mpicc_naked" = x], [
# Try by combining stdout/err into one file.
for version in $versions; do
    for naked_compiler in $compilers; do
        rm -f mpi.txt naked.txt
        AS_IF([$MPICC $version 1>mpi.txt 2>&1],
            [AS_IF([$naked_compiler $version 1>naked.txt 2>&1],
                [AS_IF([diff mpi.txt naked.txt >/dev/null],
                    [ga_cv_mpicc_naked=$naked_compiler; break],
                    [echo "diff failed, skipping" >&AS_MESSAGE_LOG_FD])],
                [echo "$naked_compiler $version failed, skipping" >&AS_MESSAGE_LOG_FD])],
            [echo "$MPICC $version failed, skipping" >&AS_MESSAGE_LOG_FD])
    done
    AS_IF([test "x$ga_cv_mpicc_naked" != x], [break])
done
])
rm -f mpi.txt mpi.err naked.txt naked.err
AS_IF([test "x$ga_cv_mpicc_naked" = x], [ga_cv_mpicc_naked=error])
])
AS_IF([test "x$ga_cv_mpicc_naked" = xerror],
    [AC_MSG_WARN([Could not determine the C compiler wrapped by MPI])
     AC_MSG_WARN([This is usually okay])])
])dnl
