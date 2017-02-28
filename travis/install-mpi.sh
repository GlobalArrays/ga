#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

case "$os" in
    Linux)
        sudo apt-get update -q
        case "$MPI_IMPL" in
            mpich)
                sudo apt-get install -y -q mpich libmpich-dev
                ;;
            openmpi)
                sudo apt-get install -y -q openmpi-bin libopenmpi-dev
                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 1
                ;;
        esac
        ;;
    Darwin)
        brew update
        case "$MPI_IMPL" in
            mpich)
                brew install mpich
                ;;
            openmpi)
                brew info open-mpi
                brew install openmpi
                ;;
            *)
                echo "Unknown MPI implementation: $MPI_IMPL"
                exit 1
                ;;
        esac
        ;;
esac

