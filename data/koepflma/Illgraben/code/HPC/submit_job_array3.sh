#!/bin/bash
# Example to submit a Job and get a Nodes exclusivly
#
#SBATCH -J 3feature_files # A single job name for the array
#SBATCH --qos normal  # normal priority level
#SBATCH --mail-user=manuela.koepfli@wsl.ch # Your email
#SBATCH --mail-type=SUBMIT,END,FAIL # notify you at given Events
#SBATCH -e feature_files%j.err # Standard error output
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --partition=node
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G #request of RAM
#SBATCH --time=00-01:00 #run for DD-HH:MM
#SBATCH -t 0-06:00 # Running time of xx hours
#SBATCH --array=123,135,136,137,234,199,207,208,223,233 # create a array from x to y and use it later with variable SLURM_ARRAY_TASK_ID. the numer after the %flag specifies how many jobs to start in parallel
#

#execute your script:
python comp_con_attributes_array3.py "${SLURM_ARRAY_TASK_ID}"
