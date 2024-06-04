#!/bin/bash
#
#SBATCH -J matlab # A single job name for the array
#SBATCH -p janson,janson_cascade,shared # Partition
#SBATCH -n 1 # one core
#SBATCH -N 1 # on one node
#SBATCH -t 0-15:00 # Running time
#SBATCH --mem 4000 # Memory request
#SBATCH -o /n/janson_lab/lab/sma/CompDA_paper/results/simulation/binary_Y/debug/matlab_%A_%a.out # Standard output
#SBATCH -e /n/janson_lab/lab/sma/CompDA_paper/results/simulation/binary_Y/debug/matlab_%A_%a.err # Standard error

module load matlab/R2021a-fasrc01
matlab -nodisplay -nosplash -r "i_job=$SLURM_ARRAY_TASK_ID;binary_y_matlab;quit;"
