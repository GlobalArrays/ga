#! /bin/bash

# Exit on error
set -ev
os=`uname`
TRAVIS_ROOT="$1"
PORT="$2"
MPI_IMPL="$3"
USE_CMAKE="$4"
FORTRAN_COMPILER="$5"

# Environment variables
export CFLAGS="-std=c99"
#export MPICH_CC=$CC
export MPICC=mpicc
# export autotools path
export PATH=$TRAVIS_ROOT/bin:$PATH

MAKE_JNUM=4

# Capture details of build
case "$MPI_IMPL" in
    mpich)
        export PATH=$TRAVIS_ROOT/mpich/bin:$PATH
        mpichversion
        mpicc -show
        ;;
    openmpi)
        case "$os" in
            Darwin)
                echo "Mac"
                # Open MPI 2.0.x / v2.1.x won't startup otherwise
                # https://www.open-mpi.org/faq/?category=osx
                export TMPDIR=/tmp
            ;;
            Linux)
                echo "Linux"
                export PATH=$TRAVIS_ROOT/open-mpi/bin:$PATH
            ;;
        esac
        # this is missing with Mac build it seems
        #ompi_info --arch --config
        mpicc --showme:command
        ;;
    intel)
	source /opt/intel/oneapi/setvars.sh --force || true
	;;
esac

# Configure and build
if [ "$USE_CMAKE" = "Y" ] ; then
    echo 'nothing to do here for cmake '
else
    ./autogen.sh $TRAVIS_ROOT
fi
case "$os" in
    Darwin)
        echo "Mac CFLAGS" $CFLAGS
        ;;
    Linux)
	if [ $(${CC} -dM -E - </dev/null 2> /dev/null |grep __clang__|head -1|cut -c19) ] ; then
	    export CFLAGS="${CFLAGS} -fPIC "
	fi
        echo "Linux CFLAGS" $CFLAGS
        ;;
esac
if [ "$USE_CMAKE" = "Y" ] ; then
case "x$PORT" in
    xmpi-ts)
        ga_rt="MPI_2SIDED"
        ;;
    xmpi-pr)
        ga_rt="MPI_PROGRESS_RANK"
        ;;
    xmpi-pt)
        ga_rt="MPI_PROGRESS_THREAD"
        ;;
    xmpi-mt)
        ga_rt="MPI_MULTITHREADED"
        ;;
    x)
        ga_rt="MPI_2SIDED"
        ;;
    x*)
	echo PORT = "$PORT" not recognized
	exit 1
        ;;
esac
    mkdir -p build
    cd build
    echo FORTRAN_COMPILER is $FORTRAN_COMPILER
    FC=$FORTRAN_COMPILER cmake -DMPIEXEC_MAX_NUMPROCS=5 -DGA_RUNTIME="$ga_rt" ../
else
case "x$PORT" in
    xofi)
        ./configure --with-ofi=$TRAVIS_ROOT/libfabric
        if [[ "$os" = "Darwin" ]] ; then
            export COMEX_OFI_LIBRARY=$TRAVIS_ROOT/libfabric/lib/libfabric.dylib
        fi
        ;;
    xarmci)
        ./configure --with-armci=$TRAVIS_ROOT/external-armci CFLAGS="${CFLAGS} -pthread " LIBS=-lpthread
        ;;
    x)
        ./configure ${CONFIG_OPTS}
        ;;
    xmpi-pr)
        if [[ "$USE_SICM" = "Y" ]] ; then
            export CFLAGS="-DUSE_SICM=1 -I${HOME}/no_cache/SICM/include -I${HOME}/no_cache/SICM/include/public ${CFLAGS}"
            export LDFLAGS="-L${HOME}/no_cache/jemalloc/lib -ljemalloc -L${HOME}/no_cache/SICM/lib -lsicm ${LDFLAGS}"
            export LD_LIBRARY_PATH="${HOME}/no_cache/SICM/lib:${HOME}/no_cache/jemalloc/lib:${LD_LIBRARY_PATH}"
        fi
        ./configure --with-${PORT} ${CONFIG_OPTS}
        ;;
    x*)
        ./configure --with-${PORT} ${CONFIG_OPTS}
        ;;
esac
fi

# build libga
make V=0 -j ${MAKE_JNUM}

# build test programs
if [ "$USE_CMAKE" = "Y" ] ; then
    cd global/testing
    make
    cd ../..
else
    make V=0 checkprogs -j ${MAKE_JNUM}
fi
# run one test
MAYBE_OVERSUBSCRIBE=
if test "x$os" = "xDarwin" && test "x$MPI_IMPL" = "xopenmpi"
then
    MAYBE_OVERSUBSCRIBE=-oversubscribe
fi

# Determine test name based on whether fortran was supported.
TEST_NAME=./global/testing/test.x
if test -x $TEST_NAME
then
    echo "Running fortran-based test"
else
    TEST_NAME=./global/testing/testc.x
    if test -x $TEST_NAME
    then
        echo "Running C-based test"
    else
        echo "No suitable test was found"
        exit 1
    fi
fi

if test "x$PORT" = "xmpi-pr"
then
    mpirun -n 5 ${MAYBE_OVERSUBSCRIBE} ${TEST_NAME}
else
    mpirun -n 4 ${MAYBE_OVERSUBSCRIBE} ${TEST_NAME}
fi
