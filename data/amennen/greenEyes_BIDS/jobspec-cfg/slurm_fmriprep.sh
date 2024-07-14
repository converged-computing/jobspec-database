#!/bin/bash

# Run within BIDS code/ directory: sbatch slurm_fmriprep.sh

# Name of job?
#SBATCH --job-name=fmriprep

# Set partition
#SBATCH --partition=all

# How long is job?
#SBATCH -t 12:00:00

# How much memory to allocate (in MB)?
#SBATCH --cpus-per-task=8 --mem-per-cpu=20000
#SBATCH --array=46
# Where to output log files?
#SBATCH --output='../derivatives/fmriprep/logs/fmriprep-%A_%a.log'
#SBATCH --mail-user=amennen@princeton.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Remove modules because Singularity shouldn't need them
echo "Purging modules"
module purge

# Print job submission info
echo "Slurm job ID: " $SLURM_JOB_ID
echo "Slurm array task ID: " $SLURM_ARRAY_TASK_ID
date

# Set subject ID based on array index
printf -v subj "%03d" $SLURM_ARRAY_TASK_ID

# Run fMRIPrep script with participant argument
echo "Running fMRIPrep on sub-$subj"

# CHECK IF PREVIOUS FILES EXIST FOR THAT SUBJECT
#fmriprep output dir
data_dir=/jukebox/norman/amennen/RT_prettymouth/data #this is my study directory
fmriprep_dir=$data_dir/bids/Norman/Mennen/5516_greenEyes/derivatives/fmriprep/sub-$subj
./run_fmriprep.sh $subj
# if [ "$(ls -A $fmriprep_dir)" ]
# then
# 	session=02
# 	echo "other anat files--not deleting folder"
# 	./run_fmriprep_day2.sh $subj $session

# else
# 	echo "no previous fmriprep data"
# 	./run_fmriprep.sh $subj

# fi

echo "Finished running fMRIPrep on sub-$subj"
date
