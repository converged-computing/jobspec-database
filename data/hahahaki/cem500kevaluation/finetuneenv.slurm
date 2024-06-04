#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=randomfreezeencoder
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128000M
#SBATCH --time=12:00:00
#SBATCH --output=/home/codee/scratch/sourcecode/cem-dataset/evaluation/pretrain_500k_test_%j_%N.txt

module load python/3.10
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -r /home/codee/scratch/sourcecode/cem-dataset/evaluation/requirements.txt
pip install /home/codee/segmentation_models_pytorch-0.3.3-py3-none-any.whl

module load cuda/11.4
#export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
#export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.
#echo "r$SLURM_NODEID master: $MASTER_ADDR"
#echo "r$SLURM_NODEID Launching python script"

log_dir="/home/codee/scratch/sourcecode/cem-dataset/evaluation/finetunesave"
echo log_dir : `pwd`/$log_dir
mkdir -p `pwd`/$log_dir

srun python /home/codee/scratch/sourcecode/cem-dataset/evaluation/finetune.py > $log_dir/randominitfreezeencode1.14

#python /home/codee/scratch/sourcecode/cem-dataset/evaluation/setup_benchmarks/setup_data.py "/home/codee/scratch/sourcecode/cem-dataset/benchdata"

echo "finetune finished"