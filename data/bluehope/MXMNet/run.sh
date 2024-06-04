#!/bin/bash
#SBATCH -J molecule
#SBATCH -p amd_a100_4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time 47:30:00
#SBATCH --gres=gpu:1
#SBATCH --comment pytorch

source /home01/$USER/.bashrc

module purge
module load singularity/3.9.7
module load htop nvtop
module load gcc/10.2.0
module load cuda/10.2
module list

conda activate mxmnet

cd /scratch/$USER/workspace/MXMNet

echo "START"

srun python main.py

echo "DONE"
