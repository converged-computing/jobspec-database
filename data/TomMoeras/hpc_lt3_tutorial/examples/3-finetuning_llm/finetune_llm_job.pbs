#!/bin/bash -l
#PBS -l nodes=1:ppn=24:gpus=1
#PBS -l mem=32gb
#PBS -l walltime=01:00:00
#PBS -N FineTuneLLM
#PBS -m abe

conda info --envs

echo "Loading modules"

conda activate hpc_lt3_tutorial_venv

echo "Current Python interpreter:"
which python

echo "Current Conda environment:"
conda env list | grep '*'

# Change to the directory where the Python script is located
cd $PBS_O_WORKDIR

# Check CUDA version
echo "Checking CUDA version for PyTorch and Environment:"
python -c "import torch; print(torch.version.cuda)"

# Check GPU availability
echo "Checking GPU availability:"
nvidia-smi

echo "Starting the Python script..."

# Run the Python script
python src/finetune.py

echo "Python script has finished running."

exit 0