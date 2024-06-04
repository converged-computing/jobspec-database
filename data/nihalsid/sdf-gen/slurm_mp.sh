#!/bin/bash

#SBATCH --job-name voxelize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=0
##SBATCH --mem-per-cpu=32G
#SBATCH --exclude=moria

cd /rhome/ysiddiqui/sdf-gen
python process_meshes_matterport.py --mesh_dir /cluster/gondor/ysiddiqui/Matterport3DMeshes/ --df_lowres_dir /cluster/gondor/ysiddiqui/Matterport3DDistanceFields16/complete_lowres --df_highres_dir /cluster/gondor/ysiddiqui/Matterport3DDistanceFields16/complete_highres --chunk_lowres_dir /cluster/gondor/ysiddiqui/Matterport3DDistanceFields16/chunk_lowres --chunk_highres_dir /cluster/gondor/ysiddiqui/Matterport3DDistanceFields16/chunk_highres --num_proc $SLURM_ARRAY_TASK_COUNT --proc $SLURM_ARRAY_TASK_ID
