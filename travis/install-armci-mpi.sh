#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

# export MPI/Autotools path
export PATH=$TRAVIS_ROOT/bin:$PATH
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

if [ ! -d "$TRAVIS_ROOT/armci-mpi" ]; then
    cd $TRAVIS_ROOT
    git clone -b 'mpi3rma' --depth 10 https://github.com/jeffhammond/armci-mpi.git armci-mpi-source
    cd armci-mpi-source
    ./autogen.sh
    ./configure CC=mpicc MPICC=mpicc CFLAGS="-std=gnu99" --prefix=$TRAVIS_ROOT/armci-mpi --enable-win-allocate --enable-explicit-progress
    make
    make install
else
    echo "ARMCI-MPI installed..."
    find $TRAVIS_ROOT/armci-mpi -name "armci.h"
fi
