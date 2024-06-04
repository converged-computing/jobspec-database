#!/bin/sh
#
#SBATCH --account=stats          # The account name for the job.
#SBATCH --job-name=HelloWorld    # The job name.
#SBATCH --gres=gpu:1             # Request 1 gpu (1-4 are valid).
#SBATCH -c 1                     # The number of cpu cores to use.
#SBATCH --time=1:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=1gb        # The memory the job will use per cpu core.

module load cuda80/toolkit
module load gcc
module load anaconda
python Hello.py

# End of script
