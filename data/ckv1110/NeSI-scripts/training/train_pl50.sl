#!/bin/bash -e
#SBATCH --job-name=resnet50-mc
#SBATCH --time=4:50:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=A100:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4

# Load singularity module
module purge
module load Singularity

# Bind directories and append SLURM job ID to output directory
#export SINGULARITY_BIND="\
#/nesi/project/uoa03709/work-dir/py-data:/var/inputdata"

# Run container %runscript
singularity exec --bind /nesi/project/uoa03709/work-dir/py-data:/var/inputdata --cleanenv --nv /nesi/project/uoa03709/containers/sif/smp-cv_0.2.0.sif python /var/inputdata/train_UPP_resnet50.py
# srun singularity exec --nv smp-cv_0.1.4.sif nvidia-smi
# srun singularity exec --nv smp-cv_0.1.4.sif python -c "import torch; print(torch.cuda.is_available())"
