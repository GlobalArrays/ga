#!/usr/bin/env bash

set -e
set -x

TRAVIS_ROOT="$1"
os=`uname`
case "$MPI_IMPL" in
    mpich)
	export PATH=$TRAVIS_ROOT/mpich/bin:$PATH
        mpichversion
        mpicc -show
        export MPICC=mpicc
        ;;
    openmpi)
	if [ "$os" = "Linux" ] ; then
	    $TRAVIS_ROOT/open-mpi/bin/mpicc --showme:command
	    export MPICC=$TRAVIS_ROOT/open-mpi/bin/mpicc
	fi
	;;
esac

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
${ARMCI_MPI_DIR}/configure CC=$ARMCIMPICC --prefix=${TRAVIS_ROOT}/external-armci --enable-g
make install
