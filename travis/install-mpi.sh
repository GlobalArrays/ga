#!/usr/bin/env bash
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"
MPI_IMPL="$2"

MAKE_JNUM=2
#check if FC,F77 and CC are defined
if [[ -z "${CC}" ]]; then
    CC=cc
fi
if [[  -z "${FC}" ]]; then
    FC=gfortran
fi
if [[ -z "${F77}" ]]; then
    F77="${FC}"
fi

if [ "$F77" == "gfortran" ] && [ "$os" == "Darwin" ]; then
    if [[ ! -x "$(command -v gfortran)" ]]; then
	echo gfortran undefined
	echo symbolic link gfortran-12
	ln -sf /usr/local/bin/gfortran-12 /usr/local/bin/gfortran
    fi
fi

# this is where updated Autotools will be for Linux
export PATH=$TRAVIS_ROOT/bin:$PATH
case "$MPI_IMPL" in
    mpich)
	if [ "$TRAVIS" == "true" ] && [ "$os" == "Darwin" ]; then
            brew install mpich || brew upgrade mpich || true
	else
        if [ ! -d "$TRAVIS_ROOT/mpich" ] || [  ! -x "$TRAVIS_ROOT/mpich/bin/mpicc" ]; then
	    MPI_VER=3.4.2
            wget --no-check-certificate http://www.mpich.org/static/downloads/"$MPI_VER"/mpich-"$MPI_VER".tar.gz
            tar -xzf mpich-"$MPI_VER".tar.gz
            cd mpich-"$MPI_VER"
            mkdir -p build && cd build
	    GNUMAJOR=`$F77 -dM -E - < /dev/null 2> /dev/null | grep __GNUC__ |cut -c18-`	
	    GFORTRAN_EXTRA=$(echo $F77 | cut -c 1-8)
	    echo MPICH F77 is `which "$F77"`
	    echo F77 version is `"$F77" -v`
	    if [ "$GFORTRAN_EXTRA" = "gfortran" ]; then
		if [ $GNUMAJOR -ge 10  ]; then
		    FFLAGS_IN="-w -fallow-argument-mismatch -O1"
		else
		    FFLAGS_IN="-w -O1"
		fi
	    elif [ "$F77" = "ifort" ]; then
		case "$os" in
		    Darwin)
			IONEAPI_ROOT=~/apps/oneapi
			FFLAGS_IN="-O0"
			;;
		    Linux)
			IONEAPI_ROOT=/opt/intel/oneapi
			;;
		esac
		source "$IONEAPI_ROOT"/setvars.sh --force || true
		ifort -V
		icc -V
	    fi
	    CFLAGS_in="-O1 -w -fPIC"
# --disable-opencl since opencl detection generates -framework opencl on macos that confuses opencl	    
            ../configure CC="$CC" FC="$F77" F77="$F77" CFLAGS="$CFLAGS_in" FFLAGS="$FFLAGS_IN" --prefix=$TRAVIS_ROOT/mpich --with-device=ch3 --disable-shared --enable-static --disable-opencl pac_cv_have_float16=no
            make -j ${MAKE_JNUM}
            make -j ${MAKE_JNUM} install
	    ls -Rlta $TRAVIS_ROOT/mpich/lib
#	    file $TRAVIS_ROOT/mpich/lib/libmpi.*.dylib || true
#	    file $TRAVIS_ROOT/mpich/lib/libpmpi.*.dylib || true
#	    ls -lrt $TRAVIS_ROOT/mpich/lib/libpmpi.*.dylib || true
#	    nm  $TRAVIS_ROOT/mpich/lib/libpmpi.*.dylib || true
        else
            echo "MPICH already installed"
        fi
	fi
	;;
    openmpi)
	case "$os" in
	    Darwin)
		echo "Mac"
		# Homebrew is at 1.10.2, which is broken for STRIDED/IOV=DIRECT.
		brew info open-mpi
		brew update
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
