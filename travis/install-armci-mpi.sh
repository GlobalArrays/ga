#!/usr/bin/env bash

set -e
set -x

TRAVIS_ROOT="$1"

if [ "${TRAVIS_OS_NAME}" = "linux" ] ; then
    case "$MPI_IMPL" in
        mpich)
            $TRAVIS_ROOT/mpich/bin/mpichversion
            $TRAVIS_ROOT/mpich/bin/mpicc -show
            export MPICC=$TRAVIS_ROOT/mpich/bin/mpicc
            ;;
        openmpi)
            $TRAVIS_ROOT/open-mpi/bin/mpicc --showme:command
            export MPICC=$TRAVIS_ROOT/open-mpi/bin/mpicc
            ;;
    esac
fi

if [ ! -z "${MPICC}" ] ; then
    echo "Found MPICC=${MPICC} in your environment.  Using that."
    ARMCIMPICC=${MPICC}
else
    ARMCIMPICC=mpicc
fi

ARMCI_MPI_DIR=${TRAVIS_ROOT}/armci-mpi
/bin/rm -rf ${ARMCI_MPI_DIR}
git clone -b master --depth 10 https://github.com/jeffhammond/armci-mpi.git ${ARMCI_MPI_DIR}

if ! [ -f ${ARMCI_MPI_DIR}/configure ] ; then
  cd ${ARMCI_MPI_DIR}
  ./autogen.sh
fi

if ! [ -d ${ARMCI_MPI_DIR}/build ] ; then
  mkdir ${ARMCI_MPI_DIR}/build
fi
cd ${ARMCI_MPI_DIR}/build
${ARMCI_MPI_DIR}/configure CC=$ARMCIMPICC --prefix=${TRAVIS_ROOT}/external-armci --enable-g
make install
