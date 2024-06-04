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

srun python /nesi/project/uoa03829/phan635/NeSI-Project-Template/MedSegDiff_pengqian/scripts/segmentation_sample_brats.py --data_dir /nesi/project/uoa03829/BraTS2023Dataset/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData   --out_dir /nesi/project/uoa03829/phan635/GLI_sample_folder_seg  --model_path /nesi/project/uoa03829/phan635/output/savedmodel015000.pt --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 25  --dpm_solver True 
