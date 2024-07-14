#!/bin/bash
#SBATCH --job-name=mup_train
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/network/scratch/n/normandf/mup/logs/slurm-%j.out
#SBATCH --error=/network/scratch/n/normandf/mup/logs/slurm-%j.err

module load miniconda/3
conda activate $SCRATCH/conda/mup
# Set the EXP_NAME only if not already set.
# (For example, it is already set in the sweep script which calls this one.)
EXP_NAME=${EXP_NAME:="gpt2_256"}

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$((${SLURM_JOB_NUM_NODES:=1} * $SLURM_GPUS_ON_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "NODE RANK: $SLURM_NODEID"
echo "SLURM_CPUS_ON_NODE: $SLURM_CPUS_ON_NODE"


# NOTE: Setting this environment variable to a non-zero value makes all call to DatasetDict.map
# ignore the cache files, which isn't what we want. Therefore, we manually use `save_to_disk` and
# `load_from_disk`.
# export HF_DATASETS_IN_MEMORY_MAX_SIZE=$(free -b | grep Mem | awk '{print $2}')

# NOTE: No need to copy the datasets manually from SCRATCH to SLURM_TMPDIR anymore, thanks to the
# save/load_from_disk stuff that's mentioned above!

# Optional: Set some wandb-related environment variables.
#export WANDB_LOG_MODEL=1
export WANDB_WATCH=false
export WANDB_PROJECT=mup_demo
export WANDB_TAGS=$EXP_NAME

# NOTE: Could get pretty much identical behaviour with torchrun directly.
#torchrun --node_rank $SLURM_NODEID --nnodes $SLURM_JOB_NUM_NODES \
#    --nproc_per_node=$SLURM_GPUS_ON_NODE --standalone mup_demo/train.py "$@"

# NOTE: Could be replaced by `mup_demo/train_big_model.py`
training_script=${training_script:="mup_demo/train.py"}

accelerate launch \
     --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT \
     --num_processes $WORLD_SIZE --num_cpu_threads_per_process=$SLURM_CPUS_ON_NODE \
     --num_machines $SLURM_NNODES --machine_rank=$SLURM_NODEID \
     $training_script "$@"
