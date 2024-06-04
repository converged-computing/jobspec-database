#!/bin/bash
#SBATCH --job-name=R_DXboots
#SBATCH --output=logs/%x_%j.out 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --mem=12000

## set the second environment variable to get the base directory
BASEDIR=${SLURM_SUBMIT_DIR}



## set up a trap that will clear the ramdisk if it is not cleared
function cleanup_ramdisk {
    echo -n "Cleaning up ramdisk directory /$SLURM_TMPDIR/ on "
    date
    rm -rf /$SLURM_TMPDIR
    echo -n "done at "
    date
}

#trap the termination signal, and call the function 'trap_term' when
# that happens, so results may be saved.
trap "cleanup_ramdisk" TERM

module load R

mkdir -p /home/edickie/R/x86_64-pc-linux-gnu-library/4.1

Rscript ./code/R/running_bootedperm_DXeffects_PINTFC.R $SLURM_ARRAY_TASK_ID
