#!/bin/bash
#SBATCH --job-name=rl_test
#SBATCH --partition=es1
#SBATCH --qos=es_lowprio
#SBATCH --account=pc_automat
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --constraint=es1_a40
#SBATCH --time=04:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=chenxin0210@lbl.gov
#export CUDA_VISIBLE_DEVICES=0
##python /global/scratch/users/duow/codes/ocp/main.py --mode train --config-yml /global/home/groups/pc_automat/ocp_Se/model_refine/1_dpp_raw/dpp.yml
#source /global/home/users/chenxin0210/miniconda3/etc/profile.d/conda.sh
#conda activate rl
#source activate ml
echo current conda env is $CONDA_DEFAULT_ENV
echo "================"
echo current GPU condition is:
python gpu.py
echo available nCPU is:
nproc
echo "================"
echo start running:


accelerate launch scripts/train.py --config config/dgx.py:gender_equality

