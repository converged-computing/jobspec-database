#!/bin/bash

#SBATCH --job-name=AI-HERO_energy_baseline_prediction
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=76
#SBATCH --time=20:00:00

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=76

data_workspace=/hkfs/work/workspace/scratch/dz4120-energy-train-data
group_workspace=...

module load compiler/gnu/11
module load mpi/openmpi/4.0
module load lib/hdf5/1.12
module load devel/cuda/11.8

source ${group_workspace}/energy_baseline_env/bin/activate
srun python ${group_workspace}/predict.py ${data_workspace}
