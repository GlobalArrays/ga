#! /bin/sh

# Exit on error
set -ev

os=`uname`
TRAVIS_ROOT="$1"
PORT="$2"
MPI_IMPL="$3"

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
        case "$os" in
            Darwin)
                echo "Mac"
            ;;
            Linux)
                echo "Linux"
                export PATH=$TRAVIS_ROOT/mpich/bin:$PATH
            ;;
        esac
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
esac

# Configure and build
./autogen.sh $TRAVIS_ROOT
case "x$PORT" in
    xofi)
        ./configure --with-ofi=$TRAVIS_ROOT/libfabric
        if [[ "$os" == "Darwin" ]]; then
            export COMEX_OFI_LIBRARY=$TRAVIS_ROOT/libfabric/lib/libfabric.dylib
        fi
        ;;
    x)
        ./configure $CONFIG_OPTS
        ;;
    x*)
        ./configure --with-${PORT} $CONFIG_OPTS
        ;;
esac

# Run unit tests
make V=0 -j ${MAKE_JNUM}
make V=0 checkprogs -j ${MAKE_JNUM}
make V=0 check-travis
