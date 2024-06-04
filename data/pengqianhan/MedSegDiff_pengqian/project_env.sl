#!/bin/bash -e
#SBATCH --job-name=diffv2_pengqian      # job name
#SBATCH --account=uoa03829              # Project Account
#SBATCH --time=0-30:00:00               # Walltime
#SBATCH --gpus-per-node=A100:1          # GPU resources required per node
#SBATCH --cpus-per-task 4               # 4 CPUs per GPU
#SBATCH --mem 50G                       # 50G per GPU
#SBATCH --partition gpu                 # Must be run on GPU partition.
#SBATCH --mail-type=ALL
#SBATCH --mail-user=phan635@aucklanduni.ac.nz

# load modules
module purge
module load CUDA/11.6.2
module load Miniconda3/22.11.1-1
source $(conda info --base)/etc/profile.d/conda.sh
export PYTHONNOUSERSITE=1

# display information about the available GPUs
nvidia-smi

# check the value of the CUDA_VISIBLE_DEVICES variable
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# activate conda environment
conda deactivate
conda activate ./venv
which python

# optional, used to peek under NCCL's hood
export NCCL_DEBUG=INFO 

# start training script

srun python /nesi/project/uoa03829/phan635/NeSI-Project-Template/MedSegDiff_pengqian/scripts/segmentation_env.py --inp_pth /nesi/project/uoa03829/phan635/output_sample --out_pth /nesi/project/uoa03829/phan635/output_env