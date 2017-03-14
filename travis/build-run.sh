#! /bin/sh

# Exit on error
set -ev

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

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
./configure

# Run unit tests
make V=0 -j ${MAKE_JNUM}
make V=0 checkprogs -j ${MAKE_JNUM}
make V=0 check-travis
