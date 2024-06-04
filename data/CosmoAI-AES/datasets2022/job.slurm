#!/bin/sh
#SBATCH --partition=GPUQ           # Use a GPU
#SBATCH --account=ie-idi
#SBATCH --time=96:00:00            # Max wall time
#SBATCH --nodes=1
#SBATCH --gres=gpu:8               # Require 8 GPUs - why?
#SBATCH --constraint="A100"        # Require specific GPU model - why?
#SBATCH --job-name="Cosmo-test"    
#SBATCH --output=results.out
#SBATCH --mail-user=hasc@ntnu.no
#SBATCH --mail-type=ALL

 
WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Job Name:          $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Job ID:            $SLURM_JOB_ID"
echo "Nodes used:        $SLURM_JOB_NODELIST"
echo "Number of nodes:   $SLURM_JOB_NUM_NODES"
echo "Cores (per node):  $SLURM_CPUS_ON_NODE"
echo "Total cores:       $SLURM_NTASKS"

module purge
module load torchvision/0.8.2-fosscuda-2020b-PyTorch-1.7.1
module swap NCCL/2.8.3-CUDA-11.1.1 NCCL/2.8.3-GCCcore-10.2.0-CUDA-11.1.1
module swap PyTorch/1.7.1-fosscuda-2020b PyTorch/1.8.1-fosscuda-2020b
module list
python3 Cosmo_Ai/Python/CosmoCNN.py
