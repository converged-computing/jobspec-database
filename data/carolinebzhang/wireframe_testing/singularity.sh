#!/bin/bash

#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1

#SBATCH -n 4
#SBATCH --mem=40g
#SBATCH --time=3:00:00
module load cuda

# Print key runtime properties for records
echo Master process running on `hostname`
echo Directory is `pwd`
echo Starting execution at `date`
echo Current PATH is $PATH

cd scripts 
singularity exec -B /oscar/scratch,/oscar/data/ /users/czhan157/wireframe_testing/docker/singularity1.simg python3 test.py  --config-file ../config-files/layout-SRW-S3D.yaml --img-folder ../wireframe_data/wireframe_testing CHECKPOINT ../data/model_proposal_s3d.pth GNN_CHECKPOINT ../data/model_gnn_s3d.pth OUTPUT_DIR ../results