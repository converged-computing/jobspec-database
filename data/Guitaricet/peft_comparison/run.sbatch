#!/bin/bash
#SBATCH --job-name="reloraF"
#SBATCH --partition=g40x
#SBATCH --mem=400GB
#SBATCH --nodes=1
#SBATCH --hint=nomultithread        # We get physical cores not logical
#SBATCH --cpus-per-task=12          # Number of cores per tasks
#SBATCH --gres=gpu:2                # Number of gpus
#SBATCH --output=/fsx/vlialin/slurmlogs/%j.out  # Set this dir where you want slurm outs to go
#SBATCH --error=/fsx/vlialin/slurmlogs/%j.out  # Set this dir where you want slurm outs to go
#SBATCH --exclusive
#SBATCH --account eleuther
#SBATCH --time=23:59:59
#SBATCH --mail-type=ALL

set -e

source /fsx/vlialin/peft_comparison/.venv/bin/activate
cd /fsx/vlialin/peft_comparison

# ######################################################
# Cluster-specific:
export WANDB_WATCH=false
export WANDB_DIR=/fsx/vlialin/wandb/wandb_dir
export WANDB_CACHE_DIR="/fsx/vlialin/wandb/.cache"
export TMPDIR=/fsx/vlialin/tmp
source /admin/home-vlialin/sbatch_exports.sh  # WANDB_API_KEY
export TRANSFORMERS_CACHE=/fsx/vlialin/cache/transformers_cache
export HF_DATASETS_CACHE=/fsx/vlialin/cache/datasets_cache
# ######################################################

# ######################################################
# Multinode-specific
# Exported from another script
export NCCL_PROTO=simple
export PATH=/opt/amazon/efa/bin:/opt/amazon/openmpi/bin:/opt/slurm/bin:$PATH
module load cuda/12.1
export NCCL_DEBUG=WARNING
export NCCL_TREE_THRESHOLD=0
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export PYTHONFAULTHANDLER=1
export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"
export LD_PRELOAD=/usr/local/cuda-12.1/lib/libnccl.so
# End export

# More multinode-specific:
echo Nodelist: $SLURM_JOB_NODELIST
nodes_array=($(scontrol show hostnames $SLURM_JOB_NODELIST))
echo ${nodes_array[@]}
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
num_nodes=${#nodes_array[@]}

echo Node IP: $head_node_ip
echo Number of nodes: $num_nodes
export LOGLEVEL=WARNING
# End of multinode-specific
# ######################################################

echo "base"

bash bash_scripts/seq2seq_base.sh

echo "3b"

bash bash_scripts/seq2seq_3b.sh

echo "large"

bash bash_scripts/seq2seq_large.sh
