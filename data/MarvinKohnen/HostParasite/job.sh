#!/bin/bash
 
#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node
#SBATCH --cpus-per-task=4          # the number cpus per task
#SBATCH --partition=long          # on which partition to submit the job
#SBATCH --time=5:00:00             # the max wallclock time (time limit your job will run)
 
#SBATCH --job-name=My_Copepod         # the name of your job
#SBATCH --mail-type=ALL             # receive an email when your job starts, finishes normally or is aborted
#SBATCH --mail-user=janayaro@uni-muenster.de # your mail address

# load needed modules
ml palma/2022a
ml Julia/1.8.2-linux-x86_64

# Previously needed for Intel MPI (as we do here) - not needed for OpenMPI
# export I_MPI_PMI_LIBRARY=/usr/lib64/libpmi.so
 
# run the application
julia  /home/j/janayaro/Projects_JM/HostParasite/CopepodParasite2D.jl