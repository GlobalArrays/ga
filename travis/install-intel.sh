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
	dir_base="cd013e6c-49c4-488b-8b86-25df6693a9b7"
	dir_hpc="edb4dc2f-266f-47f2-8d56-21bc7764e119"
	base="m_BaseKit_p_2023.2.0.49398"
	hpc="m_HPCKit_p_2023.2.0.49443"
	curl -sS -LJO https://registrationcenter-download.intel.com/akdlm/IRC_NAS/"$dir_base"/"$base".dmg
	curl -sS -LJO https://registrationcenter-download.intel.com/akdlm/IRC_NAS/"$dir_hpc"/"$hpc".dmg
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
	export TERM=dumb
        rm -f l_Base*sh l_HP*sh
	tries=0 ; until [ "$tries" -ge 10 ] ; do \
		      dir_base="20f4e6a1-6b0b-4752-b8c1-e5eacba10e01"
		      dir_hpc="1b2baedd-a757-4a79-8abb-a5bf15adae9a"
		      base="l_BaseKit_p_2024.0.0.49564"
		      hpc="l_HPCKit_p_2024.0.0.49589"
		      wget -nv https://registrationcenter-download.intel.com/akdlm/IRC_NAS/"$dir_hpc"/"$hpc".sh \
			  && wget -nv  https://registrationcenter-download.intel.com/akdlm/IRC_NAS/"$dir_base"/"$base".sh \
			  && break ;\
			  tries=$((tries+1)) ; echo attempt no.  $tries    ; sleep 30 ;  done
            sh ./"$base".sh -a -c -s --action install --components intel.oneapi.lin.mkl.devel --install-dir $IONEAPI_ROOT  --eula accept
	    if [[ "$?" != 0 ]]; then
		df -h
		echo "base kit install failed: exit code " "${?}"
		exit 1
	    fi
	    rm  -rf $IONEAPI_ROOT/mkl/latest/lib/ia32
	    rm  -rf $IONEAPI_ROOT/mkl/latest/lib/intel64/*sycl*
	    rm  -rf $IONEAPI_ROOT/mkl/latest/lib/intel64/*_pgi_*
	    rm  -rf $IONEAPI_ROOT/mkl/latest/lib/intel64/*_gf_*
	    intel_components="intel.oneapi.lin.ifort-compiler:intel.oneapi.lin.dpcpp-cpp-compiler"
	    if [[ "$MPI_IMPL" == "intel" ]]; then
		intel_components+=":intel.oneapi.lin.mpi.devel"
	    fi
            sh ./"$hpc".sh -a -c -s --action install \
               --components  "$intel_components"  \
               --install-dir $IONEAPI_ROOT     --eula accept
	    if [[ "$?" != 0 ]]; then
		df -h
		echo "hpc kit install failed: exit code " "${?}"
		exit 1
	    fi
	    rm  -rf $IONEAPI_ROOT/compiler/latest/linux/lib/oclfpga
	    rm -f ./"$hpc".sh ./"$base".sh
	    rm  -rf $IONEAPI_ROOT/compiler/latest/linux/lib/oclfpga || true
	
	source "$IONEAPI_ROOT"/setvars.sh --force || true
	    export I_MPI_F90=ifort
	    export I_MPI_F77=ifort
	which mpif90
	mpif90 -show
esac
which ifort
ifort -V
echo ""##### end of  install-intel.sh ####"
