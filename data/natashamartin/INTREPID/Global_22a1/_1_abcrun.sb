#!/bin/bash
#SBATCH --job-name="bat1"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24
#SBATCH --mem=48GB ## should usually be at least twice the number of gigabytes as processors.
#SBATCH -t 48:00:00

# SBATCH --array=0-23
#SBATCH --array=0-0
#SBATCH -o %x.run_%a.out.txt
#SBATCH -e %x.run_%a.err.txt


# SBATCH -p shared
# SBATCH -A TG-MCB190116
#SBATCH -p ind-shared
#SBATCH -A csd722


#SBATCH --mail-user="achaillon@health.ucsd.edu"
#SBATCH --mail-type="all"

countries=( UA  )

# export OMP_NUM_THREADS=24
export PATH=/expanse/lustre/scratch/jpg/temp_project/matlab_2020b/bin:$PATH


# Create a temporary directory on scratch
mkdir -p .matlab/local_cluster_jobs/R2019b/$SLURM_JOB_ID


module load matlab
matlab -nodisplay -r "Run_ABC('${countries[$SLURM_ARRAY_TASK_ID]}',1);exit"


# Cleanup local work directory
rm -rf .matlab/local_cluster_jobs/R2019b/$SLURM_JOB_ID

