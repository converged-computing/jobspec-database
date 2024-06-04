#!/bin/bash
#SBATCH --job-name=topomovie1      # job name (shows up in the queue)
#SBATCH --account=uoa00424     # Project Account
#SBATCH --time=2:00:00         # Walltime (HH:MM:SS)
#SBATCH --mem=60G      # memory/cpu (in MB)
#SBATCH --cpus-per-task=1       # number of cores per task (e.g. OpenMP)
#SBATCH --partition=prepost        # specify a partition
#SBATCH --hint=nomultithread    # don't use hyperthreading  -Xmx2g
#SBATCH --mail-type=END
#SBATCH --mail-user=hjin317@aucklanduni.ac.nz
#SBATCH --output=Topovideo1_%A_%a.out # Include the job ID in the names of
#SBATCH --array=2-5  # Array definition [1:5, exp code]  2 3 4 5

module load MATLAB/2017b

matlab -nodisplay -r Topovideo1
