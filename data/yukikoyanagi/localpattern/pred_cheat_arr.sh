#!/usr/bin/env bash
#
#SBATCH --account austmathjea_fat
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --time 3:00:00            # max time (HH:MM:SS)
#SBATCH --mail-type=END,FAIL
#
# Author: Yuki Koyanagi
# History:
#

echo Running on "$(hostname)"
echo Running job: "$SLURM_JOB_NAME"
echo Available nodes: "$SLURM_NODELIST"
echo Number of nodes: "$SLURM_NNODES"
echo Slurm_submit_dir: "$SLURM_SUBMIT_DIR"
echo Start time: "$(date)"
start=$(date +%s)

echo Enable modules
module purge
module add python-intel/2.7.14

echo Starting Python program
WORKDIR=/work/austmathjea/cdp
srun python pred_cheat_arr.py ${WORKDIR}/test ${WORKDIR}/assess \
     ${WORKDIR}/opts -o /work/austmathjea/locpat/v49/set4/b1 -n 1

end=$(date +%s)
echo End time: "$(date)"
dur=$(date -d "0 $end sec - $start sec" +%T)
echo Duration: "$dur"

echo Done.
