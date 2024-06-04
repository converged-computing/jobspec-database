#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --account=pls0144
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task 20
##SBATCH --constraint=48core 
#SBATCH -J gnn4itk
#SBATCH -o alazar-%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=alazar@ysu.edu

export NCCL_NET_GDR_LEVEL=PHB
export NCCL_P2P_LEVEL=NVL
export TORCH_DISTRIBUTED_DEBUG=DETAIL

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

module load cuda/11.8.0
module load miniconda3/4.10.3-py37
source activate torch
#module load gcc-compatibility/11.2.0

srun --ntasks-per-node=2 --gpu_cmode=exclusive g4i-train /users/PLS0129/ysu0053/CommonFramework/examples/Example_3/gnn_train_2.yaml
#srun --gpu_cmode=exclusive g4i-train /users/PLS0129/ysu0053/CommonFramework/examples/Example_3/metric_learning_train.yaml
#srun --gpu_cmode=exclusive g4i-infer /users/PLS0129/ysu0053/CommonFramework/examples/Example_3/metric_learning_infer.yaml
#srun --gpu_cmode=exclusive g4i-eval /users/PLS0129/ysu0053/CommonFramework/examples/Example_3/metric_learning_eval.yaml