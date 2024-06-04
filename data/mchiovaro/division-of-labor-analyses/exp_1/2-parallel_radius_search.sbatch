#!/bin/bash

#SBATCH --job-name=slurm_exp_1                # Name of job
#SBATCH --partition=general		                # Specify the partition
#SBATCH --exclusive                           # Request exclusive use of the node so that we can use more memory
#SBATCH --output=slurm_output/out_exp_1.out   # Output file
#SBATCH --error=slurm_output/err_exp_1.err    # Error file
#SBATCH --array=1-159                         # Number of times we want to run the file (dyad-round-radius combos; observations in radius_grid_search in R script)
#SBATCH --ntasks=1                            # Scripts per job
#SBATCH --time=12:00:00                       # Set run time

# save our array ID as an environmental variable
export SLURM_ARRAY_TASK_ID

# load the modules we need
module load gcc/11.3.0
module load r/4.2.1
module load gdal/3.6.1 gsl
source /gpfs/sharedfs1/admin/hpc2.0/apps/gdal/3.6.1/spack/share/spack/setup-env.sh
spack load gdal

# print to output for confirmation that it's started
echo $SLURM_ARRAY_TASK_ID": Running SLURM task"

# run the program
Rscript $HOME/dissertation/exp_1/2-parallel_radius_search-nested.R

# print to output for confirmation that it's ended
echo $SLURM_ARRAY_TASK_ID ": Job done"
