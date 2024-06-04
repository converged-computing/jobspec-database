#!/bin/bash
#SBATCH --ntasks=1               # 1 core(CPU)
#SBATCH --nodes=1                # Use 1 node
#SBATCH --job-name=gpu_tf        # sensible name for the job
#SBATCH --mem=16G                 # Default memory per CPU is 3GB.
#SBATCH --partition=gpu           # Must have - partition with node containing the GPUs
#SBATCH --gres=gpu:1              # Must have - to use the GPU in the node
#SBATCH --output=outputs/gpu-xs-%A.out
#SBATCH --error=outputs/gpu-xs-%A.out


# If you would like to use more please adjust this.

## Below you can put your scripts
# If you want to load module
module load singularity

# Hack to ensure that the GPUs work
nvidia-modprobe -u -c=0

# Run experiment
singularity exec --nv tensorflow_gpu.sif python scripts/run_tensorflow_xs.py
