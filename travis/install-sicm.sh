#!/bin/sh

# This should only be run on Linux machines, as OSX does not have NUMA
# SICM has only been added to mpi-pr
if [ "$TRAVIS_OS_NAME" != "linux" ] || [ "$PORT" != "mpi-pr" ]; then
    exit 1;
fi

set -e
set -x

# install dependencies
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update -qq
sudo apt-get install -qq libhwloc-dev libiomp-dev libnuma-dev libpfm4-dev llvm-3.9-dev numactl

# set install directory to current location to not cache jemalloc/SICM
TRAVIS_ROOT="$1"
export PATH=$TRAVIS_ROOT/bin:$PATH

# get SICM
git clone -b ga-sicm https://github.com/lanl/SICM.git
cd SICM

# install jemalloc
export JEPATH="${TRAVIS_ROOT}/jemalloc"
./install_deps.sh --jemalloc --build_dir "$(pwd)" --install_dir "${TRAVIS_ROOT}"
export LD_LIBRARY_PATH="${JEPATH}/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${JEPATH}/lib/pkgconfig:${PKG_CONFIG_PATH}"

# install SICM
./autogen.sh
mkdir -p build
cd build
../configure --with-jemalloc="${JEPATH}" --prefix="${TRAVIS_ROOT}/SICM" CFLAGS="-std=gnu99 ${CFLAGS}"
make -j $(nproc --all)
make -j $(nproc --all) install
