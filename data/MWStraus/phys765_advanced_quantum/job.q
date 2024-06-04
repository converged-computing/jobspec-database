#!/bin/bash
#SBATCH -e slurm.err                           # Error file location
#SBATCH -p brownlab-gpu,common,scavenger       # Show who you are and get priority
#SBATCH --array=1-1  			       # How many jobs do you have (the int variable $SLURM_ARRAY_TASK_ID will be incremented in the range [1,9] inclusive)
#SBATCH -c 4				       # The number of cpu cores to use
#SBATCH --mail-type=END			
#SBATCH --mail-user=jiyong.yu@duke.edu	       # It will send you an email when the job is finished. (Change it to your email otherwise I will be buried by emails... Or you can set this as empty)
#SBATCH --mem=10G			       # Memory, keep it as 10G
#SBATCH --job-name=AQM			       # How is your output is called, you name it.
#SBATCH --output=%x-%a.out                     # Your output will be named as Name-#ofJob.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate base                             # Update this field to tne conda environment you wish to use for the run

python problem1_pure_python_parallel_zoom_sections.py # Ex if you have two parameters for every job, so it's ${pA[0]} ${pA[1]}