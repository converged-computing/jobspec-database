#!/bin/bash
# ior.pbs
#   Invoke ior with typical arguments and default environment
# vlaues

#PBS -l select=1
#PBS -l walltime=09:30:00
#PBS -A d63
#PBS -m bea
#PBS -M sid.kashyap@ed.ac.uk
#PBS -N TransfeSize


module swap PrgEnv-cray PrgEnv-intel
module swap intel intel/15.0.2.164
module load allinea/5.0.1

export LD_LIBRARY_PATH=/work/d63/d63/skashyap/map_libs/map-5.0.1-intel-dyn/:$LD_LIBRARY_PATH

TARGET="/work/d63/d63/shared/IOR_DiRAC2/TransferSizeSurvey"
IOR="/work/d63/d63/skashyap/sid/ior_install/ior_intel_dyn/ior/ior_build/bin/ior"
IOR_SCRIPT="/work/d63/d63/shared/IOR_DiRAC2/TransferSizeSurvey/con.ior"

date '+%Y%m%d%H%M%S'

[ -d $TARGET ] || mkdir -p $TARGET
[ -d $TARGET ] || { echo "Failed to create $TARGET"; exit 1; }

pushd $TARGET
ARGS="-f $IOR_SCRIPT"
aprun -n 1 /work/d63/d63/shared/IOR_DiRAC2/TransferSizeSurvey/strace_d.sh -vvvv $ARGS
popd
