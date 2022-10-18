#!/bin/bash
# This configuration file was taken originally from the mpi4py project
# <http://mpi4py.scipy.org/>, and then modified for Julia

set -e
set -x

os=`uname`

MAKE_JNUM=4
case "$os" in
    Darwin)
	IONEAPI_ROOT=~/apps/oneapi
	;;
    Linux)
	IONEAPI_ROOT=/opt/intel/oneapi
	;;
esac
#echo "os oneapi root" $os $IONEAPI_ROOT
#exit 0
echo stev "$IONEAPI_ROOT/setvars.sh"
if [ -f "$IONEAPI_ROOT/setvars.sh" ]; then
    echo "Intel oneapi already installed"
    source "$IONEAPI_ROOT"/setvars.sh --force || true
    exit 0
fi
case "$os" in
    Darwin)
	mkdir -p ~/mntdmg ~/apps/oneapi || true
	cd ~/Downloads
	dir_base="18342"
	dir_hpc="18341"
	base="m_BaseKit_p_2022.1.0.92_offline"
	hpc="m_HPCKit_p_2022.1.0.86_offline"
	curl -LJO https://registrationcenter-download.intel.com/akdlm/irc_nas/"$dir_base"/"$base".dmg
	curl -LJO https://registrationcenter-download.intel.com/akdlm/irc_nas/"$dir_hpc"/"$hpc".dmg
	echo "installing BaseKit"
	hdiutil attach "$base".dmg  -mountpoint ~/mntdmg -nobrowse
	sudo ~/mntdmg/bootstrapper.app/Contents/MacOS/install.sh --cli  --eula accept \
	     --action install --components default  --install-dir ~/apps/oneapi
	hdiutil detach ~/mntdmg
	#
	echo "installing HPCKit"
	hdiutil attach "$hpc".dmg  -mountpoint ~/mntdmg -nobrowse
	sudo ~/mntdmg/bootstrapper.app/Contents/MacOS/install.sh --cli  --eula accept \
	     --action install --components default --install-dir ~/apps/oneapi
	hdiutil detach ~/mntdmg
	ls -lrta ~/apps ||true
	sudo rm -rf "$IONEAPI_ROOT"/intelpython "$IONEAPI_ROOT"/dal "$IONEAPI_ROOT"/advisor \
	     "$IONEAPI_ROOT"/ipp "$IONEAPI_ROOT"/conda_channel 	"$IONEAPI_ROOT"/dnnl \
	     "$IONEAPI_ROOT"/installer "$IONEAPI_ROOT"/vtune_profiler "$IONEAPI_ROOT"/tbb || true
	$GITHUB_WORKSPACE/travis/fix_xcodebuild.sh
	sudo cp xcodebuild "$IONEAPI_ROOT"/compiler/latest/mac/bin/intel64/.
	source "$IONEAPI_ROOT"/setvars.sh || true
	ifort -V
	icc -V
	# get user ownership of /opt/intel to keep caching happy
	my_gr=`id -g`
	my_id=`id -u`
	sudo chown -R $my_id /opt/intel
	sudo chgrp -R $my_gr /opt/intel
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
	source "$IONEAPI_ROOT"/setvars.sh --force || true
	    export I_MPI_F90=ifort
	    export I_MPI_F77=ifort
	which mpif90
	mpif90 -show
esac
which ifort
ifort -V
echo ""##### end of  install-intel.sh ####"
