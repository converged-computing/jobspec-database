#!/bin/bash 
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling  
#################
#set a job name  
#SBATCH --job-name=rep1
#################  
#a file for job output, you can check job progress
#SBATCH --output=/home/sryali/VB-HMM/Jobs/repjob-%j.out
#################
# a file for errors from the job
#SBATCH --error=/home/sryali/VB-HMM/Jobs/repjob-%j.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=00:35:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#SBATCH -n 16
#SBATCH -N 1
#################
#SBATCH --mem-per-cpu=2000
#you could use --mem-per-cpu; they mean what we are calling cores
#################
#get emailed about job BEGIN, END, and FAIL
#################
#now run normal batch commands
module load matlab/R2014a
module load spm/12b
#Run the jerb

matlab -nojit -nosplash -nodesktop -nodisplay -r "main_vbhmm_HCP_replication("$1");exit;"

