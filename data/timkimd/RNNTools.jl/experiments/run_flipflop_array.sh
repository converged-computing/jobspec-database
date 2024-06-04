#!/bin/bash
#SBATCH --job-name=flipflop-array
#SBATCH --output=slurm-%A.%a.out
#SBATCH --error=slurm-%A.%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --array=1-3

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID"
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load julia/1.6.1 cudatoolkit/11.0 cudnn/cuda-11.0/8.0.2

julia n_bit_flipflop/three_bit/original/n_bit_flipflop_disc_time_array.jl