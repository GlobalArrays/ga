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
./configure --disable-static

# Run unit tests
make V=0
make V=0 checkprogs
make V=0 check
