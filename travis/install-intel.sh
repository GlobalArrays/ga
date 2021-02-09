#!/bin/bash
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`

MAKE_JNUM=4


case "$os" in
    Darwin)
        echo "Mac not read yet"
        exit 10
        ;;
    Linux)
	export APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1
	wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
            && sudo apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB  \
	    && echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list \
            && sudo add-apt-repository "deb https://apt.repos.intel.com/oneapi all main"  \
	    && sudo apt-get update \
	    && sudo apt-get -y install intel-oneapi-ifort intel-oneapi-compiler-dpcpp-cpp-and-cpp-classic  intel-oneapi-mkl \
	    && sudo apt-get -y install intel-oneapi-mpi-devel
	source /opt/intel/oneapi/setvars.sh --force || true
esac
