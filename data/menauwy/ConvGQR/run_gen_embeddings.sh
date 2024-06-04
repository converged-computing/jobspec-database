#!/bin/bash
#SBATCH --job-name=gen_topic_adapt_embeddiings
#SBATCH --mail-type="ALL"
#SBATCH --time=7-00:00:00
#SBATCH --partition=amd-gpu-long
#SBATCH --output=%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4g.40gb:1

# making sure we load necessary module and activate right environment
# module restore

#module load ALICE/default # you can access the AMD software stack like this, also works for Intel nodes
conda init bash
source ~/.bashrc
# initialize conda 
# or:: eval "$(conda shell.bash hook)"
conda activate convgqr
echo $CONDA_DEFAULT_ENV
echo $PYTHONPATH
# # move local data to local scratch on the running node
# data_dir="/home/wangym/data1/dataset/qrecc"
# scratch_dir="/scratchdata/${SLURM_JOB_USER}/${SLURM_JOB_ID}"

# mkdir -p $scratch_dir
# echo "## Scratch_dir is: $scratch_dir"
# #cp -r $data_dir/* $scratch_dir

# test path and CUDA device
# # the pwd is the directory where the script is located
TEST_DIR=$(pwd)
echo "## Current dircectory $TEST_DIR"
echo "## Number of available CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "## Checking status of CUDA device with nvidia-smi"
nvidia-smi

# # run script!
echo "## Generating passages embeddings for topiocqa dataset!"
# /data1/wangym/conda/envs/convgqr/bin/python gen_doc_embeddings.py --config Config/gen_doc_embeddings.toml
/data1/wangym/conda/envs/convgqr/bin/python gen_doc_embeddings_adapted.py \
    --config Config/gen_doc_embeddings.toml \
    --saved_block_id=7