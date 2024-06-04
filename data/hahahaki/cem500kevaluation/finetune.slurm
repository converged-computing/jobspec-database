#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=finetunemocounet
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=128000M
#SBATCH --time=00:40:00
#SBATCH --output=/home/codee/scratch/sourcecode/cem-dataset/evaluation/pretrain_500k_test_%j_%N.txt

#export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
#export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

source /home/codee/miniconda3/etc/profile.d/conda.sh
conda activate base
#conda conda install -c conda-forge opencv

#config="/home/codee/scratch/sourcecode/cem-dataset/pretraining/mocov2/mocov2_config.yaml"

log_dir="/home/codee/scratch/sourcecode/cem-dataset/evaluation/finetunesave"
echo log_dir : `pwd`/$log_dir
mkdir -p `pwd`/$log_dir

#echo "$SLURM_NODEID master: $MASTER_ADDR"
echo "$SLURM_NODEID Launching python script"
#echo "$SLURM_NTASKS tasks running"

/home/codee/miniconda3/bin/python /home/codee/scratch/sourcecode/cem-dataset/evaluation/finetune.py > $log_dir/mocofinetune1.8

#python /home/codee/scratch/sourcecode/cem-dataset/evaluation/setup_benchmarks/setup_data.py "/home/codee/scratch/sourcecode/cem-dataset/benchdata"

echo "finetune finished"