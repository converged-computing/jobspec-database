#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=2 
#SBATCH --ntasks-per-node=28 
#SBATCH --gpus-per-node=0
#SBATCH --job-name=parallelSlurm
#SBATCH --account=PAA0202

#slurm starts job in working DIR
cd $SLURM_SUBMIT_DIR

module load miniconda3
#conda create -n vak-env python=3.6
#conda init bash
source activate vak-env
conda install pytorch torchvision cudatoolkit -c pytorch
#conda update -n base -c defaults conda
conda install attrs dask joblib matplotlib pandas scipy toml tqdm
#conda install pip
pip install vak
pip install tweetynet