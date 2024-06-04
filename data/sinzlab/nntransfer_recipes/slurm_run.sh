#!/bin/bash

#SBATCH --job-name=singularity_test # Name of the job
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=1           # Number of CPU cores per task
#SBATCH --nodes=1                   # Ensure that all cores are on one machine
#SBATCH --time=0-01:00              # Runtime in D-HH:MM
#SBATCH --partition=gpu-2080ti      # Partition to submit to
#SBATCH --gres=gpu:1                # Number of requested GPUs
#SBATCH --mem-per-cpu=1000          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --output=work/logs/hostname_%j.out    # File to which STDOUT will be written
#SBATCH --error=work/logs/hostname_%j.err     # File to which STDERR will be written
#SBATCH --mail-type=ALL             # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=arnenix@gmail.com    # Email to which notifications will be sent

./singularity_run.sh run 0 python3 bias_transfer_recipes/main.py --recipe $1 --experiment $2
