#!/bin/sh
#SBATCH --time=2:30:00    # walltime
#SBATCH --nodes=1
#SBATCH --partition=shared
#SBATCH --mem-per-cpu=4000
#SBATCH --ntasks-per-node=1  # number of processor cores (i.e. tasks)
#SBATCH --job-name=RespTest   # job name
#SBATCH --output=out_resptest.dat      # output file name
#SBATCH --error=err_resptest.dat      # error File
#
#
##### These are shell commands: Please configure them
date
module purge
# module load intel/17.0.4-fasrc01 openmpi/3.1.3-fasrc01
# module load postgresql/10.5-fasrc01
conda activate obspy

python test-ReadandRemoval_Obspy_TA.py
/n/home03/kokubo/packages/julia-1.2.0/bin/julia test-ReadandRemoval_TA.jl

echo all process has been done.
