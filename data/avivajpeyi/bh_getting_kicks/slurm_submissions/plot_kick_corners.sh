#!/bin/bash
#SBATCH --job-name=corner_plotter
#SBATCH --array=0-99
#SBATCH --time=0:20:00
#SBATCH --output=corner_plotter_%a.out
#SBATCH --mem-per-cpu=1000

source ~/.bash_profile
conda activate parallel_bilby

# Print the task id.
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

FNAME=$(printf "/fred/oz117/avajpeyi/projects/phase-marginalisation-test/jobs/out_hundred_injections_gstar/out_injection_${SLURM_ARRAY_TASK_ID}/result/injection_${SLURM_ARRAY_TASK_ID}_0_posterior_samples_with_kicks.dat")

echo "Plotting " "$FNAME"

python plot_corner_weighted_with_kick.py --samples-csv $FNAME --true-file datafiles/injections.csv --true-idx $SLURM_ARRAY_TASK_ID