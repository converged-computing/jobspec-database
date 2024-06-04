#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=train-mdm-new
#SBATCH --mail-user=youremail@email.com     # TODO fill in correct email
#SBATCH --mail-type="ALL"
#SBATCH --time=01-00:00:00
#SBATCH --partition=gpu-long
#SBATCH --output=/your_location_here/logs/%x_%j.out       # TODO fill in log path
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=2

echo "## Starting MDM run on $HOSTNAME"
DIR=$(pwd)
echo "## Current directory $DIR"

# TODO: Load your own venv, modules and settings 
# source env/bin/activate

echo "## using python:"
which python
export PYTHONPATH=".":$PYTHONPATH

echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi
echo "## Running training"

# TODO: fill in checkpoint lications below. 
CHECKPOINT=/motion-diffusion-model/save/humanml_trans_enc_512/model000200000.pt
SAVEDIR=/motion-diffusion-model/save/swdance/models

python train/train_mdm.py --save_dir=${SAVEDIR} --dataset=swdance \
    --resume_checkpoint=${CHECKPOINT} --overwrite \
    --save_interval=5000 --log_interval=500 \
    # --freeze_layers=0

# NOTE:
# default log interval = 1000, save interval = 50000 (from mdm repo).
# /motion-diffusion-model/data_loaders/humanml/utils/get_opt.py <- change datapath here for other dataset! 
# /motion-diffusion-model/model/mdm.py <- here you might want to change some freezing options.. 
