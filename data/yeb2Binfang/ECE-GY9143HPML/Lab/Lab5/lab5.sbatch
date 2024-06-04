#!/bin/bash
#SBATCH --job-name=dp2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --output=dp1-5.out
#SBATCH --mem=40GB
#SBATCH --time=05:20:00
#SBATCH --gres=gpu:v100





module purge
module load python/intel/3.8.6
module load anaconda3/2020.07
module load cuda/11.3.1
eval "$(conda shell.bash hook)"
conda activate pytorchGPU
cd /home/jy3690/hw5
python3 ./lab2.py --order 15
