#!/bin/bash -l
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=16gb
#PBS -l walltime=00:25:00
#PBS -N GPU_PyTorch_Job
#PBS -m abe

echo "Starting job on $(hostname)"
echo "Job started on $(date)"

# Loading the PyTorch module which has a Python version included
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Check CUDA version
echo "Checking CUDA version for PyTorch and Environment:"
python -c "import torch; print(torch.version.cuda)"

# Check GPU availability
echo "Checking GPU availability:"
nvidia-smi

# Change to the directory from which the job script was submitted
cd $PBS_O_WORKDIR

echo "PBS working directory is $PBS_O_WORKDIR"

echo "Running Python script"
python gpu_pytorch_script.py

echo "Finished running Python script"