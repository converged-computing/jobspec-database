#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:01:00
#SBATCH --partition=workq
#SBATCH --account=mwaeor
#SBATCH --export=MWA_DIR
#SBATCH --array=1-2

CURRENTDATE=`date +%Y_%m_%d`

obsarr=()
while IFS= read -r line || [[ "$line" ]]; do
        obsarr+=("$line")
done < LoBES13_093_corr.txt
OBSID=${obsarr[${SLURM_ARRAY_TASK_ID}]}

obspath="/astro/mwaeor/MWA/data/$OBSID/$CURRENTDATE"
echo $obspath
cd $obspath

slurmfile="rts_patch_$OBSID.sh"
echo $slurmfile

sbatch $slurmfile 

