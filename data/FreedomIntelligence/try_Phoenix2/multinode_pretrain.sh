#!/bin/bash
#SBATCH -N 12 --gres=gpu:4 --qos=gpugpu

# 系统网络架构 这个看服务器待定
export NCCL_ALGO=Ring
# export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mlx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export PYTHONUNBUFFERED=1


experiment_name=tinyllama_test
log_folder="./logs/${SLURM_JOB_ID}"
mkdir -p $log_folder


host=($(scontrol show hostnames))
for hostname in "${host[@]}"; do
	echo $hostname >> ${log_folder}/${SLURM_JOB_ID}.log
done

process_port=29501

# 在 12 个节点上循环运行 srun
for rank in {0..11}
do
	srun -N 1 --gres=gpu:4 -w ${host[${rank}]} \
	lightning run model \
		--node-rank=${rank}  \
		--main-address="${host[0]}" \
		--accelerator=cuda \
		--devices=8 \
		--num-nodes=10 \
		/TinyLlama/pretrain/tinyllama.py --devices 8 --train_data_dir data/slim_star  --val_data_dir data/slim_star
done

# 等待所有后台任务完成
wait