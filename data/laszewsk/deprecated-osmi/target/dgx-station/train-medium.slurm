#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=medium-train-osmi
#SBATCH --output=train-osmi-%u-%j.out
#SBATCH --error=train-osmi-%u-%j.err
#SBATCH --reservation=bi_fox_dgx
#SBATCH --constraint=a100_80gb

## SBATCH -c 4
## SBATCH --mem=32GB

NAME=cloudmesh-rivanna
# NAME=cloudmesh-nvidia
PROJECT_DIR=$PROJECT/osmi
RUN_DIR=$PROJECT_DIR/target/rivanna
MODEL_DIR=$PROJECT_DIR/models
# export CONTAINERDIR=/share/resources/containers/singularity

module purge

module load singularity
# module load tensorflow/2.10.0 gcc openmpi python

nvidia-smi

source $PROJECT_DIR/ENV3/bin/activate

# cms gpu watch --gpu=0 --delay=0.5 --dense > $SLURM_JOBID.gpu.log &

cd $MODEL_DIR

MODEL=medium_cnn 
time singularity exec --nv $RUN_DIR/$NAME.sif python train.py $MODEL

# cd $dir
# time singularity pull docker://tensorflow/serving:latest-gpu 
