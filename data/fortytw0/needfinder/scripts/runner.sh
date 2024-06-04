#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=04:00:00
##SBATCH --partition=sgpu
#SBATCH --partition=sgpu
#SBATCH --ntasks=24
#SBATCH --job-name=gpu-job
#SBATCH --output=gpu-job.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dasr8731@colorado.edu

module purge
module load cuda
module load cudnn

nvidia-smi

cd /scratch/summit/dasr8731/needfinder
source activate ./venv/

python -m src.train.train