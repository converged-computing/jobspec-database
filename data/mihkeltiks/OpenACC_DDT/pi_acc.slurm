#!/bin/bash
#SBATCH --exclusive
#SBATCH -J pi_acc
#SBATCH -o test.out
#SBATCH  --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH -A project_46
#SBATCH -p standard-g
#SBATCH --time=10:00

export ALLINEA_STOP_AT_MAIN=1

module load LUMI/22.12
module load partition/G
module load Linaro_Forge/23.0
module load PrgEnv-cray
module load craype-accel-amd-gfx90a
module load rocm

ddt --connect srun -n 4 pi_openacc
