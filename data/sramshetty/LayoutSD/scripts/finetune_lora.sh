#!/bin/bash
#SBATCH --partition=g40x
#SBATCH --job-name=LayoutSD
#SBATCH --account=laion
#SBATCH --nodes 4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=8
#SBATCH --output=%x_%j.out
#SBATCH --requeue

module load openmpi
module load cuda/11.7

export NCCL_PROTO=simple

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

export NCCL_DEBUG=info
export PYTHONFAULTHANDLER=1
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

PROJECT=LayoutSD
RUN_NAME=gpt2-xl_lora_local-narratives
LANG_MODEL=gpt2-xl

source /fsx/home-shivr/layoutsd_venv/bin/activate

srun --cpu_bind=v --accel-bind=gn --comment laion python /fsx/home-shivr/LayoutSD/finetune.py \
    --data_path "pipe:aws s3 cp s3://laion-west/flickr-narrative-boxes/output_wds/{00000..01392..2}.tar -" \
    --supp_data_path "pipe:aws s3 cp s3://laion-west/GRIT/wds_no_imgs/coyo_{0..20}_snappy.tar -" \
    --dataset_type narrative_wds \
    --supp_dataset_type grit \
    --num_samples 25000 \
    --supp_num_samples 100000 \
    --batch_size 2 \
    --supp_batch_size 8 \
    --resampled \
    --model ${LANG_MODEL} \
    --lora \
    --workers 8 \
    --lr 5e-6 \
    --lr_schedule cosine \
    --decay 0.1 \
    --epochs 100 \
    --warmup 5000 \
    --logging_interval 500 \
    --run_name ${RUN_NAME} \
    --delete_previous_checkpoint || sbatch /fsx/home-shivr/LayoutSD/scripts/finetune_lora.sh