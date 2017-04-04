#!/bin/sh
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

MAKE_JNUM=4

# this is where updated Autotools will be for Linux
export PATH=$TRAVIS_ROOT/bin:$PATH

case "$os" in
    Darwin)
        echo "Mac"
        brew update
        case "$MPI_IMPL" in
            mpich)
                brew install mpich | brew upgrade mpich
                ;;
            openmpi)
                brew info open-mpi
                brew install open-mpi | brew upgrade open-mpi
                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 10
                ;;
        esac
    ;;

    Linux)
        echo "Linux"
        case "$MPI_IMPL" in
            mpich)
                if [ ! -d "$TRAVIS_ROOT/mpich" ]; then
                    wget --no-check-certificate http://www.mpich.org/static/downloads/3.2/mpich-3.2.tar.gz
                    tar -xzf mpich-3.2.tar.gz
                    cd mpich-3.2
                    mkdir build && cd build
                    ../configure CFLAGS="-w" --prefix=$TRAVIS_ROOT/mpich
                    make -j ${MAKE_JNUM}
                    make install
                else
                    echo "MPICH already installed"
                fi
                ;;
            openmpi)
                if [ ! -d "$TRAVIS_ROOT/open-mpi" ]; then
                    wget --no-check-certificate https://www.open-mpi.org/software/ompi/v2.0/downloads/openmpi-2.0.2.tar.bz2
                    tar -xjf openmpi-2.0.2.tar.bz2
                    cd openmpi-2.0.2
                    mkdir build && cd build
                    ../configure CFLAGS="-w" --prefix=$TRAVIS_ROOT/open-mpi \
                                --without-verbs --without-fca --without-mxm --without-ucx \
                                --without-portals4 --without-psm --without-psm2 \
                                --without-libfabric --without-usnic \
                                --without-udreg --without-ugni --without-xpmem \
                                --without-alps --without-munge \
                                --without-sge --without-loadleveler --without-tm \
                                --without-lsf --without-slurm \
                                --without-pvfs2 --without-plfs \
                                --without-cuda --disable-oshmem \
                                --disable-libompitrace \
                                --disable-mpi-io  --disable-io-romio \
                                --enable-mpi-thread-multiple
                    make -j ${MAKE_JNUM}
                    make install
                else
                    echo "Open-MPI already installed"
                fi
                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 20
                ;;
        esac
        ;;
esac
