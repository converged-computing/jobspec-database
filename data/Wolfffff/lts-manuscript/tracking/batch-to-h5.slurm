#!/bin/bash
#SBATCH --job-name=bat_inf  # Name of the job
#SBATCH --output=logs/batch_inf_%A_%a.out  # STDOUT file
#SBATCH --error=logs/batch_inf_%A_%a.err   # STDERR file
#SBATCH --nodes=1               # Node count
#SBATCH --ntasks=1          # Number of tasks across all nodes
#SBATCH --cpus-per-task=1      # Cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=128G           # total memory per node
#SBATCH --array=1-190
#SBATCH --time=01:00:00          # Run time limit (HH:MM:SS)        # Job array
#SBATCH --mail-type=all          # Email on job start, end, and fault
#SBATCH --mail-user=swwolf@princeton.edu

module load conda
conda init bash
conda activate sleap_dev
TARGET_DIR="/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_tracks/20220217-lts-cam1"


LIST="1through190-tracked.txt"
LINE_NUMBER=${SLURM_ARRAY_TASK_ID}
FILE=$(sed "${LINE_NUMBER}q;d" ${LIST})

OUTPUT_NAME=${FILE%.slp}

sleap-convert ${FILE} --format analysis
