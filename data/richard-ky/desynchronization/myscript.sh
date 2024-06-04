#!/bin/bash
#
#SBATCH --job-name=desync
#SBATCH --partition=gpu
#SBATCH --output=desync.txt
#
#SBATCH --cpus-per-task=56
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-user=richard.ky@sjsu.edu
#SBATCH --mail-type=BEGIN,END

module load matlab

matlab -nodisplay -nosplash -nodesktop -r "run('Data_Collection_Desynchronization.m')", exit
