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
case "$MPI_IMPL" in
    mpich)
        if [ ! -d "$TRAVIS_ROOT/mpich" ] || [  ! -x "$TRAVIS_ROOT/mpich/bin/mpicc" ]; then
            wget --no-check-certificate http://www.mpich.org/static/downloads/3.4.1/mpich-3.4.1.tar.gz
            tar -xzf mpich-3.4.1.tar.gz
            cd mpich-3.4.1
            mkdir -p build && cd build
	    GNUMAJOR=`$F77 -dM -E - < /dev/null 2> /dev/null | grep __GNUC__ |cut -c18-`	
	    GFORTRAN_EXTRA=$(echo $F77 | cut -c 1-8)
	    if [ "$GFORTRAN_EXTRA" = "gfortran" ]; then
		if [ $GNUMAJOR -ge 10  ]; then
		    FFLAGS_IN="-w -fallow-argument-mismatch -O2"
		else
		    FFLAGS_IN="-w -O2"
		fi
	    elif [ "$F77" = "ifort" ]; then
		case "$os" in
		    Darwin)
			IONEAPI_ROOT=~/apps/oneapi
			;;
		    Linux)
			IONEAPI_ROOT=/opt/intel/oneapi
			;;
		esac
		source "$IONEAPI_ROOT"/setvars.sh --force || true
		ifort -V
		icc -V
	    fi
	    if [ $(${CC} -dM -E - </dev/null 2> /dev/null |grep __clang__|head -1|cut -c19) ] ; then
		CFLAGS_in="-w -fPIC"
	    else
		CFLAGS_in="-w"
	    fi
# --disable-opencl since opencl detection generates -framework opencl on macos that confuses opencl	    
            ../configure CC="$CC" FC="$F77" F77="$F77" CFLAGS="$CFLAGS_in" FFLAGS="$FFLAGS_IN" --prefix=$TRAVIS_ROOT/mpich --with-device=ch3 --disable-opencl
            make -j ${MAKE_JNUM}
            make -j ${MAKE_JNUM} install
        else
            echo "MPICH already installed"
        fi
    
	;;
    openmpi)
	case "$os" in
	    Darwin)
		echo "Mac"
		# Homebrew is at 1.10.2, which is broken for STRIDED/IOV=DIRECT.
		brew info open-mpi
		brew install open-mpi || brew upgrade open-mpi || true
		;;
	    Linux)
                if [ ! -d "$TRAVIS_ROOT/open-mpi" ] || [ ! -x "$TRAVIS_ROOT/open-mpi/bin/mpicc" ] ; then
                    wget --no-check-certificate https://www.open-mpi.org/software/ompi/v2.0/downloads/openmpi-2.0.2.tar.bz2
                    tar -xjf openmpi-2.0.2.tar.bz2
                    cd openmpi-2.0.2
                    mkdir -p build && cd build
		    if [ $(${CC} -dM -E - </dev/null 2> /dev/null |grep __clang__|head -1|cut -c19) ] ; then
			CFLAGS_in="-w -fPIC"
		    else
			CFLAGS_in="-w"
		    fi
                    ../configure CC="$CC" FC="$F77" F77="$F77" CFLAGS="$CFLAGS_in" --prefix=$TRAVIS_ROOT/open-mpi \
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
	esac
	;;
    intel)
        ./travis/install-intel.sh
	;;
    *)
	echo "Unknown MPI implementation: $MPI_IMPL"
	exit 10
	;;
esac
