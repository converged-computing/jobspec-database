#!/bin/bash -l

#SBATCH --job-name=mistral-7b-chat-pdf
#SBATCH --mail-user=asaklani@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=100g
#SBATCH --time=8:00:00
#SBATCH --account=engin1
#SBATCH --partition=spgpu
#SBATCH --gpus=1
#SBATCH --output=/home/asaklani/output.log
module purge
module load python3.10-anaconda
conda activate llm

cd /home/asaklani/llm-recipes/
nvidia-smi
python scripts/train.py --config models/mistral-7b-dolly-5k-rag-split/mistral-7b-dolly-rag.yml
