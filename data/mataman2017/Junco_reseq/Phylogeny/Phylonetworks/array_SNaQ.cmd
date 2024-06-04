#!/bin/bash
#SBATCH --error=/projects/VONHOLDT/jsala/C_Phylo/E_Phylonetworks/OE/runsnaq_slurm%a.err
#SBATCH -o /projects/VONHOLDT/jsala/C_Phylo/E_Phylonetworks/OE/runsnaq_slurm%a.log
#SBATCH -J runsnaq
#SBATCH --array=0-10
#SBATCH -c 30
#SBATCH --mem=32G
#SBATCH --time=60:00:00

## --array: to run multiple instances of this script,
##          one for each value in the array.
##          1 instance = 1 task
## -J job name
## -c number of cores (CPUs) per task

echo "slurm task ID = $SLURM_ARRAY_TASK_ID used as hmax"
echo "start of SNaQ parallel runs on $(hostname)"
# finally: launch the julia script, using Julia executable appropriate for slurm, with full paths:
/home/jg2334/download/julia-1.9.4/bin/julia --history-file=no -- runSNaQ.jl $SLURM_ARRAY_TASK_ID 30 > net$SLURM_ARRAY_TASK_ID_30runs.screenlog 2>&1
echo "end of SNaQ run ..."
