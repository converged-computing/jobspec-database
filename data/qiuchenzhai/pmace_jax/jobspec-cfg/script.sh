#!/bin/sh -l
#FILENAME: myjobsubmissionfile

#SBATCH -A standby
#SBATCH --nodes=1 --gpus=1 --cpus-per-gpu=32 --get-user-env=L --mem=48G
#SBATCH --time=01:00:00
#SBATCH -e error.out
#SBATCH -o output.out
#SBATCH --job-name demo_pmace_jax
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qzhai@purdue.edu

module load anaconda/2020.11-py38
#module load cuda/11.7.0
#module load gcc/9.3.0
module load cudnn/cuda-12.1_8.9
conda activate pmace_jax
nvidia-smi
cd tests/synthetic_image/single_mode/
nohup python noisy_data_reconstruction.py

# To submit the script: sbatch script.sh
