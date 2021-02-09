#!/bin/bash
mkdir -p ~/mntdmg || true
cd ~/Downloads
curl -LJO https://registrationcenter-download.intel.com/akdlm/irc_nas/17426/m_BaseKit_p_2021.1.0.2427_offline.dmg
curl -LJO https://registrationcenter-download.intel.com/akdlm/irc_nas/17398/m_HPCKit_p_2021.1.0.2681_offline.dmg
ls -lrt *dmg
#
echo "installing BaseKit"
hdiutil attach m_BaseKit_p_2021.1.0.2427_offline.dmg  -mountpoint ~/mntdmg -nobrowse
df 
sudo ~/mntdmg/bootstrapper.app/Contents/MacOS/install.sh --cli  --eula accept \
 --action install --components default
hdiutil detach ~/mntdmg
sudo du -sh /opt/intel
#
echo "installing HPCKit"
hdiutil attach m_HPCKit_p_2021.1.0.2681_offline.dmg  -mountpoint ~/mntdmg -nobrowse
df
sudo ~/mntdmg/bootstrapper.app/Contents/MacOS/install.sh --cli  --eula accept \
 --action install --components default
#sudo cat /opt/intel/oneapi/logs/* /private/tmp/root/intel_oneapi_installer/*/*log || true
hdiutil detach ~/mntdmg
sudo rm -rf /opt/intel/oneapi/intelpython /opt/intel/oneapi/dal /opt/intel/oneapi/advisor \
     /opt/intel/oneapi/ipp /opt/intel/oneapi/conda_channel 	/opt/intel/oneapi/dnnl \
     /opt/intel/oneapi/installer /opt/intel/oneapi/vtune_profiler /opt/intel/oneapi/tbb || true
df
ls -lrt /opt ||true
ls -lrt /opt/intel/oneapi ||true
sudo du -sh /opt/intel
sudo du -sk /opt/intel/oneapi | sort -n ||true
source /opt/intel/oneapi/setvars.sh || true
ifort -V
icc -V
#exit 1
# get user ownership of /opt/intel to keep caching happy
my_gr=`id -g`
my_id=`id -u`
sudo chown -R $my_id /opt/intel
sudo chgrp -R $my_gr /opt/intel
ls -la /opt/intel/oneapi
sync
