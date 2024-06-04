#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=2-00:15:00     # 1 day and 15 minutes
#SBATCH --output=./output_slurm/PfSep13.o
##SBATCH --mail-user=kchau012@ucr.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name="PfSep13 "
#SBATCH -p batch  # This is the default partition, you can use any of the following; intel, batch, highmem, gpu


# Print current date
date

echo 'PF_Sep13;quit'|matlab -nodesktop



