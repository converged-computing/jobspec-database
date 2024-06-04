#!/bin/bash -l
#SBATCH -N 1
#SBATCH -G 1
#SBATCH -t 12:00:00
#SBATCH -q regular
#SBATCH -J DL4N_full_prod
#SBATCH -L SCRATCH,cfs
#SBATCH -C gpu
#SBATCH -A m2043_g
#SBATCH --output logs/%A_%a  # job-array encodding
#SBATCH --image=nersc/pytorch:ngc-21.08-v2
#SBATCH --array 1-1 #a
#SBATCH --gpus-per-task=1

data_path=/pscratch/sd/k/ktub1999/bbp_May_18_8944917/
srun -n 1 shifter python3 Ntran2.py --data_path $data_path

