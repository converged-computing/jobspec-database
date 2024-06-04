#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --account=eleuther
#SBATCH --job-name="contriever"
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --output=/fsx/carper/contriever/BEIR/contriever%x_%j.out  # !!SPECIFY THIS 
#SBATCH --exclude=gpu-st-p4d-24xlarge-[1-229]
#SBATCH --comment eleuther 

module load openmpi
source /opt/intel/mpi/latest/env/vars.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib
export NCCL_PROTO=simple
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/aws-ofi-nccl/lib
export PATH=$PATH:/opt/amazon/efa/bin:/opt/amazon/openmpi/bin
export PATH=/opt/amazon/efa/bin:$PATH
export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=info
export NCCL_TREE_THRESHOLD=0
export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=1

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64

export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_pml="^cm"
export OMPI_MCA_btl="tcp,self"
export OMPI_MCA_btl_tcp_if_exclude="lo,docker1"
export OMPI_MCA_btl_base_verbose=30
export OMPI_MCA_plm_rsh_no_tree_spawn=1

export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l)

echo "Node Count: $COUNT_NODE"
echo "Host Names: $HOSTNAMES"

###############################################################################
# Contriever Setup
###############################################################################
export HF_HOME=/fsx/guac/.cache/huggingface
export TRANSFORMERS_CACHE=/fsx/guac/.cache/huggingface/transformers

TRAIN_PATH=/fsx/carper/contriever
cd $TRAIN_PATH
source $TRAIN_PATH/.env/bin/activate

dataset='scidocs' # 'nq' 'quora' 'fever' 'hotpotqa' 'climate-fever' 'arguana' fiqa' 'trec-covid' 'webis-touche2020' 'dbpedia-entity' 'nfcorpus' 'scidocs' 'scifact' 'cqadupstack'
beir_dir=$TRAIN_PATH/BEIR

source $TRAIN_PATH/.env/bin/activate  && srun --comment eleuther --cpu_bind=v --accel-bind=gn python3.8 eval_beir.py \
    --model_name_or_path $model \
    --dataset $dataset \
    --beir_dir $beir_dir \
    --main_port $MASTER_PORT \