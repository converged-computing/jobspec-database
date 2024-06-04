#!/bin/bash
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=isobutane_matlab
#################  
#a file for job output, you can check job progress
#SBATCH --output=isobutane_matlab.out
#################
# a file for errors from the job
#SBATCH --error=isobutane_matlab.err
#################
#time you think you need; default is one hour
#in minutes in this case, hh:mm:ss
#SBATCH --time=7-00:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#SBATCH -p evanreed
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#memory per node; default is 4000 MB per CPU
#SBATCH --mem=6400
#################
#tasks to run per node; a "task" is usually mapped to a MPI processes.
# for local parallelism (OpenMP or threads), use "--ntasks-per-node=1 --cpus-per-task=16" instead
#SBATCH --ntasks-per-node=1 --cpus-per-task=1
#################

#now run normal batch command
ml load matlab

matlab -r main_train_isobutane_new
