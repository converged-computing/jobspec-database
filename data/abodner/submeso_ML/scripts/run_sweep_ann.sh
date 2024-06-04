#!/bin/bash

#SBATCH --job-name=ann_paramsweep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=96GB
#SBATCH --output=slurm_%j.out


# Begin execution
module purge

/scratch/ab10313/pytorch-example/my_pytorch.ext3:ro \
  /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif

singularity exec --nv \
            --overlay /scratch/ab10313/pytorch-example/my_pytorch.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            /bin/bash -c "source /ext3/env.sh; python /home/ab10313/submeso_ML/nn/lightning/ann_hyperparam_sweep.py 10"
