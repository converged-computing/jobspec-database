#!/bin/bash -l
#SBATCH -p debug

#SBATCH -n 768
#SBATCH -N 24

#SBATCH -t 00:30:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.paul@cyi.ac.cy

module load autohbw/default
module swap craype-haswell craype-mic-knl
MyName=`echo $0 | awk -F\/ '{sub(/^\.\//,"",$0); sub(/\.sh$/,"",$0); print $0}'`
top_level_dir=/scratch2/scratchdirs/srijitp/beta3.31_2hex_24c48_ml-0.09530_mh-0.04/propagators
log=$MyName.log

path_to_qlua=/global/homes/s/srijitp/spaul/projects/pi_nucleon

scripts=/global/homes/s/srijitp/spaul/projects/pi_nucleon

path_to_prog=$scripts

TSize=48
Thalf=$(($TSize / 2))

params=params.qlua
include="$scripts/plaquette.qlua  $scripts/timer.qlua $scripts/stout_smear.qlua $scripts/load_gauge_field.qlua $scripts/random_functions.qlua $scripts/gamma_perm_phase.qlua $scripts/gi_dcovi.qlua $scripts/make_mg_solver.qlua $scripts/write_propagator.qlua $scripts/read_propagator.qlua $scripts/deltapp_piN.qlua"

# gauge configuration number
g=230031

mkdir -p $top_level_dir/$g && cd $top_level_dir/$g  || exit 1

echo "# [$MyName] `date`" > $log
pwd >> $log

nproc=$(( $SLURM_JOB_NUM_NODES * 32 ))
echo "# [$MyName] number of processes = $nproc" >> $log

source_location_string="$(awk 'BEGIN{printf("source_locations = { ")}; $1==0 && $2=='$g' && $NF < '$Thalf' {printf("{t=%d, pos={%d, %d, %d}}, ", $6, $3, $4, $5)}; END{printf("}\n")}' $top_level_dir/source_location.nsrc04perConf )"
echo "# [$MyName] source location string is \"$source_location_string\" "

input="params.$g.qlua"

cat $scripts/params.qlua | awk '
  /^source_locations/ {print "'"$source_location_string"'"; next}
  /^nconf/ {print "nconf = ", '$g'; next}
  {print}' > $input


QLUA_SCRIPT=invert_contract.qlua
srun -n $nproc $path_to_qlua/qlua_edison $include $input $path_to_prog/$QLUA_SCRIPT

exitStatus=$?
if [ $exitStatus -ne 0 ]; then
  echo "[$MyName] Error from srun for $QLUA_SCRIPT, status was $exitStatus"
  exit 1
else
  echo "# [$MyName] srun finished successfully for $QLUA_SCRIPT, status was $exitStatus"
fi

#QLUA_SCRIPT=invert_contract_oet.qlua
#srun -n $nproc $path_to_qlua/qlua $include $input $path_to_prog/$QLUA_SCRIPT
#
#exitStatus=$?
#if [ $exitStatus -ne 0 ]; then
#  echo "[$MyName] Error from srun for $QLUA_SCRIPT, status was $exitStatus"
#  exit 1
#else
#  echo "# [$MyName] srun finished successfully for $QLUA_SCRIPT, status was $exitStatus"
#fi

echo "# [$MyName] `date`" >> $log
exit 0
