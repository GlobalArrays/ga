#!/usr/bin/env bash

# This should only be run on Linux machines, as OSX does not have NUMA
# SICM has only been added to mpi-pr
os=`uname`
echo " in sicm install. os is " $os
if [ "$os" != "Linux" ] || [ "$PORT" != "mpi-pr" ]; then
    exit 1;
fi

set -e
set -x

# install dependencies
sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
sudo apt-get update -qq
sudo apt-get install -qq libhwloc-dev libomp5 libomp-dev libnuma-dev libpfm4-dev llvm-dev numactl xsltproc

# set install directory to current location to not cache jemalloc/SICM
TRAVIS_ROOT="$1"
export PATH=$TRAVIS_ROOT/bin:$PATH
#install jemalloc
git clone https://github.com/jemalloc/jemalloc
cd jemalloc
export JEPATH="${TRAVIS_ROOT}/jemalloc"
sh autogen.sh
./configure --with-jemalloc-prefix=je_ --prefix="${JEPATH}"
make -j $(nproc --all)
make -j $(nproc --all) install
export LD_LIBRARY_PATH="${JEPATH}/lib:${LD_LIBRARY_PATH}"
export PKG_CONFIG_PATH="${JEPATH}/lib/pkgconfig:${PKG_CONFIG_PATH}"

# get SICM
git clone  https://github.com/lanl/SICM.git
cd SICM
git checkout 5944a56e0ccf159b72ce6fe980745b021216b580


# install SICM
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX="${TRAVIS_ROOT}/SICM"
#../configure --with-jemalloc="${JEPATH}" --prefix="${TRAVIS_ROOT}/SICM" CFLAGS="-std=gnu99 ${CFLAGS}"
make -j $(nproc --all)
make -j $(nproc --all) install
