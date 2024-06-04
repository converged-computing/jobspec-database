#!/bin/bash
#SBATCH --job-name=matlab_job
#SBATCH --ntasks=1 --nodes=1 --cpus-per-task=1
#SBATCH --mem-per-cpu=5G
#SBATCH --partition day
#SBATCH --time=0:05:00
#SBATCH --mail-type=ALL
#SBATCH --output=%x_%j.out

cd /home/adw54/Documents/egc_hpc_manual

# Load matlab module and run master .m file
module load MATLAB/2019a-parallel
matlab -nodisplay < matlab/master.m
