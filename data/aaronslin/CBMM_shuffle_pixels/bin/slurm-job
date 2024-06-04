#!/usr/bin/env bash
#SBATCH --array=1-20
#SBATCH --job-name=shuffle_pixels
#SBATCH --output slurm_out/%A_%a.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

set -euxo pipefail
${SLURM_ARRAY_TASK_ID:=3}

# allows us to use modules within a script
source ${MODULESHOME}/init/bash

module add openmind/singularity/2.2.1

singularity exec --bind /om:/om /om/user/aaronlin/py35-tf.img python -u conv_shuffle.py -s $SLURM_ARRAY_TASK_ID -o 1 -d cifar
