#!/bin/bash
#SBATCH --job-name=barlow_twins.sh_9_1.0_jitter
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1 
#SBATCH -C v100-32g
#SBATCH --qos=qos_gpu-t3
#SBATCH --cpus-per-task=6
#SBATCH --hint=nomultithread
#SBATCH --time=16:40:00
#SBATCH --output=./terminal/barlow_twins.sh_9_1.0_contrast%j.out 
#SBATCH --error=./jitter_hue_errors/barlow_twins.sh_9_1.0_contrast%j.out
module load pytorch-gpu/py3/1.7.1
conda deactivate
conda activate clean
nvidia-smi
set -x
bash bash_files/cifar10/barlow_twins.sh 1.0 9
