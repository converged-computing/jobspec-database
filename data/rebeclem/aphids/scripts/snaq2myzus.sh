#!/bin/bash
#SBATCH -J snaq2_myzus
#SBATCH -o snaq_%A_%a.out
#SBATCH -e snaq_%A_%a.err
#SBATCH -p medium
#SBATCH --array=0-10
#SBATCH -c 30 
#SBATCH -t 7-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rebeclem@gmail.com


#--- Start the timer
t1=$(date +"%s")

module load julia/1.5.3 
echo "slurm task ID = $SLURM_ARRAY_TASK_ID used as hmax"
echo "start of SNaQ parallel runs on $(hostname)"
# finally: launch the julia script, using Julia executable appropriate for slurm, with full paths:
julia --project="/home/rebecca.clement/.julia/project1/" /home/rebecca.clement/90day_aphid/aphids_github/scripts/runSNaQ2myzus.jl $SLURM_ARRAY_TASK_ID 30 > net$SLURM_ARRAY_TASK_ID_30runs.screenlog 2>&1
echo "end of SNaQ run ..."

#---Complete job
t2=$(date +"%s")
diff=$(($t2-$t1))
echo "[---$SN---] ($(date)) $(($diff / 60)) minutes and $(($diff % 60)) seconds elapsed."
echo "[---$SN---] ($(date)) $SN COMPLETE."
