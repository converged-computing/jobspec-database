#!/bin/bash

####### select resources (check https://ubccr.freshdesk.com/support/solutions/articles/13000076253-requesting-specific-hardware-in-batch-jobs)
#SBATCH --constraint=V100

####### make sure no other jobs are assigned to your nodes

####### further customizations
#SBATCH --job-name="PointNet"
#SBATCH --output=out/%j.stdout
#SBATCH --error=err/%j.stderr
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00

hostname
module load tensorflow/1.14.0
. /util/common/tensorflow/1.14.0/py36/anaconda3-5.2.0/etc/profile.d/conda.sh
conda activate tensorflow-gpu
date
python train.py
date
conda deactivate
module unload tensorflow/1.14.0
