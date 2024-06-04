#!/bin/sh

#SBATCH --job-name=SIF_SC
#SBATCH --output=logs/gpu_SIF_slide_classification.out
#SBATCH --error=logs/gpu_SIF_slide_classification.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

echo "starting .."
#srun singularity exec /projets/sig/mullah/singularity/sif/ubuntu18_osirim.sif python3 "../programs/main.py" -extract TRUE -level 0 -size 224 -overlap FALSE
echo "done"
