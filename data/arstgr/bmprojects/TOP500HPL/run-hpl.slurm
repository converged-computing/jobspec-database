#!/bin/bash
#SBATCH -t 00:20:00
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH -p top500
#SBATCH -o docker-hpl.%j
#SBATCH --job-name=docker-hpl
#SBATCH --mem=0

echo "Running on hosts: $(echo $(scontrol show hostname))"

echo "Node topology"
lstopo-no-graphics

export BASE_DIR=/workspace
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

export UCX_IB_PCI_RELAXED_ORDERING=on 
export CUDA_DEVICE_ORDER=PCI_BUS_ID 
export NCCL_DEBUG=WARN   #INFO
export NCCL_IB_PCI_RELAXED_ORDERING=1 
export NCCL_TOPO_FILE=/opt/microsoft/ndv5-topo.xml 
export NCCL_SOCKET_IFNAME=eth0 
#export UCX_NET_DEVICES=eth0 
export OMPI_MCA_coll_hcoll_enable=0

export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib,smcuda
export UCX_NET_DEVICES=mlx5_ib0:1,mlx5_ib1:1,mlx5_ib2:1,mlx5_ib3:1,mlx5_ib4:1,mlx5_ib5:1,mlx5_ib6:1,mlx5_ib7:1

export CPU_AFFINITY="0-11:12-23:24-35:36-47:48-59:60-71:72-83:84-95"
export GPU_AFFINITY="0:1:2:3:4:5:6:7"
export MEM_AFFINITY="0:0:0:0:1:1:1:1"
export UCX_AFFINITY="mlx5_ib0:mlx5_ib1:mlx5_ib2:mlx5_ib3:mlx5_ib4:mlx5_ib5:mlx5_ib6:mlx5_ib7"

export PIN_MASK="0,12,24,36,48,60,72,84"
export CONT="nvcr.io/nvidia/hpc-benchmarks:23.5"
export MOUNT="/opt/microsoft:/opt/microsoft,/opt/nccl-tests:/opt/nccl-tests"

export DAT="${BASE_DIR}/hpl-linux-x86_64/sample-dat/HPL-dgx-${SLURM_JOB_NUM_NODES}N.dat"
export CMD="${BASE_DIR}/hpl.sh --cpu-affinity ${CPU_AFFINITY} --gpu-affinity ${GPU_AFFINITY} --mem-affinity ${MEM_AFFINITY} --ucx-affinity ${UCX_AFFINITY} --dat ${DAT}"

cd $SLURM_SUBMIT_DIR
srun --mpi=pmi2 --cpu-bind=none \
	        --ntasks=$SLURM_JOB_NUM_NODES \
		--ntasks-per-node=1           \
		--cpus-per-task=96            \
		--gpus-per-node=8             \
                $SLURM_SUBMIT_DIR/max_gpu_app_clocks.sh 	

echo "****************************************************"
sleep 1
echo "starting ..."

srun --mpi=pmi2 --whole \
	        --cpu-bind=none \
	        --container-image "${CONT}" \
                --container-mounts="${MOUNT}" \
		-N $SLURM_JOB_NUM_NODES \
		--ntasks=$SLURM_NTASKS \
		--ntasks-per-node=8           \
		--cpus-per-task=12 \
		--gpus-per-node=8	\
		/bin/bash -c "${CMD}"

echo "finishing ..."
echo "****************************************************"
sleep 1

srun --mpi=pmi2 --cpu-bind=none \
	        --ntasks=$SLURM_JOB_NUM_NODES \
		--ntasks-per-node=1           \
		--cpus-per-task=96            \
		--gpus-per-node=8             \
                $SLURM_SUBMIT_DIR/max_gpu_app_clocks.sh -r 	

