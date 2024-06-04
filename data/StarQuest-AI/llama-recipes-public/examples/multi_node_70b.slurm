#!/bin/bash

#SBATCH --job-name=Llama2-70b-sft
#SBATCH --ntasks=2
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8
#SBATCH --partition=debug 
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# Enable for A100
export FI_PROVIDER="efa"

echo Node IP: $head_node_ip
export LOGLEVEL=INFO
# debugging flags (optional)
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="ibs14f0,ibs14f1"
export FI_EFA_USE_DEVICE_RDMA=1

#srun pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes

srun pip uninstall -y llama-recipes
srun pip install -U pip setuptools
srun pip install -U --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118 
srun pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 -e .

srun torchrun --nproc_per_node 8 --rdzv_id $RANDOM --rdzv_backend c10d --rdzv_endpoint $head_node_ip:29500 examples/finetuning.py  \
  --enable_fsdp --use_peft --peft_method lora \
  --batch_size_training 8 \
  --model_name /data0/models/huggingface/meta-llama/Llama-2-70b-hf \
  --output_dir /data0/model_output/Llama-2-70b-hf-sft/ \
  --dist_checkpoint_root_folder /data0/model_output/checkpoint/Llama-2-70b-hf-sft/ \

