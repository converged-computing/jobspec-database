#!/bin/sh
# 
# Simple submit script for Slurm.
#
#
#SBATCH -p gpu
#SBATCH -J dlctest0            # The job name.
#SBATCH --time=24:00:00              # The time the job will take to run.
#SBATCH --gres=gpu:1 -c 1	# --gres=gpu:N -c M to request N GPUs and M CPUs
#SBATCH -o dlcTest0.out -e dlcTest0.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ess2129@columbia.edu

source activate deeplabcut
module load cuda/9.0.176 
module load cudnn/v7.0-cuda-9.0
module load gcc/7.4.0

export PATH=/mnt/home/evanschaffer/anaconda3/bin:$PATH
export PYTHONPATH=/mnt/home/evanschaffer/anaconda3/envs/deeplabcut

conda list

/mnt/home/evanschaffer/anaconda3/envs/deeplabcut/bin/python deeplabcut_train.py

# End of script


