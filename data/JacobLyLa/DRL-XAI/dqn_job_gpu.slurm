#!/bin/bash
#SBATCH --account=ie-idi
#SBATCH --job-name=dqn_job_gpu
#SBATCH --time=00:30:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=out/dqn_job_gpu.txt
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jacob.LLarsen@hotmail.com
#SBATCH --mail-type=ALL

module purge
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module list

python -m cProfile -s cumtime -o program.prof -m src.DRL.train_qrunner
