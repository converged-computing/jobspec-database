#!/bin/bash
#PBS -N wandb_run_sf
#PBS -o wandb_run_sf.out
#PBS -j oe
#PBS -lselect=1:ncpus=4:mem=24gb:ngpus=1:gpu_type=RTX6000
#PBS -lwalltime=24:00:00
set -vx

# Need to set the weights and biases API key
export WANDB_API_KEY=$api_key

module purge
module load anaconda3/personal
module load cuda/11.0.1
source activate tf

# Always install newest version of local package
cd /rds/general/user/zrw20/home/DD-GAN-AE
pip install -e .

cd /rds/general/user/zrw20/home/DD-GAN-AE/ddganAE/wandb
nvidia-smi
python train_wandb_sf.py --model="$model" --datafile="$HOME/data/processed/sf_snapshots_200timesteps_rand.npy" --niters=200 --savemodel="False"

mkdir $WORK/$PBS_JOBID
cp * $WORK/$PBS_JOBID

echo "... Run finished $(date) ..."
