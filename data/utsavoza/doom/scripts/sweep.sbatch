#!/bin/bash

#SBATCH --job-name=dqn-hyperparams-sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --priority=4294967293
#SBATCH --account=class

module purge;

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;

singularity exec --nv \
  --overlay /scratch/$USER/my_env/overlay-50G-10M.ext3:rw \
  /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
  /bin/bash -c "source /ext3/env.sh; wandb agent fuzzy-enigma/dqn-sweep/${1}"
