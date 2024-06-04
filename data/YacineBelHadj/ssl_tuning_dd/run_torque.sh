#!/bin/bash
#PBS -l gpus=1
#PBS -l walltime=00:10:00
#PBS -e slurm-$PBS_JOBID.err

ENV_NAME=ssl_dd

export MY_APP_ENV=hpc
export NCCL_DEBUG=INFO

# Load modules (make sure these modules or equivalent ones are available on the new cluster)
module load PyTorch-Lightning/2.1.3-foss-2023a
module load Hydra/1.1.1-GCCcore-10.3.0

# Setup virtual environment as Conda is not available
virtualenv --system-site-packages $ENV_NAME
source $ENV_NAME/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

chmod +x src/train.py
python ./src/train.py

deactivate
