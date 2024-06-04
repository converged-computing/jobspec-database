#! /bin/bash
#PBS -l walltime=1:14:59
#PBS -r n
#PBS -j oe
#PBS -k oe
#PBS -l nodes=1
echo dumping some info on the worker node
hostname
df -h
uptime
free
echo ""
export DCACHE_RA_BUFFER="250000000"
echo "Certificate info:"
echo '$X509_USER_CERT: '$X509_USER_CERT
echo '$X509_USER_PROXY: '$X509_USER_PROXY

echo “starting dir:”
pwd
cd working
WORKDIR=`pwd`
echo "WORKDIR = " $WORKDIR
DESTDIR=`pwd`
cd $WORKDIR
echo "changed dir to :"
pwd
export SCRAM_ARCH=slc6_amd64_gcc491
export LD_LIBRARY_PATH=/user/fblekman/lib:$LD_LIBRARY_PATH
export PATH=/user/fblekman/lib:$PATH
source $VO_CMS_SW_DIR/cmsset_default.sh
eval `scram runtime -sh`
cmsenv

SCRATCHDIR=$TMPDIR
mkdir $SCRATCHDIR
cd $SCRATCHDIR
echo "changed dir to :" 
pwd

cd $SCRATCHDIR
ls -las
