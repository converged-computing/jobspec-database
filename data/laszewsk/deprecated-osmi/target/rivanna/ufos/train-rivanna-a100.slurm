#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition=bii-gpu
#SBATCH --account=bii_dsc_community
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=train-osmi
#SBATCH --output=train-osmi-%u-%j.out
#SBATCH --error=train-osmi-%u-%j.err
#SBATCH --reservation=bi_fox_dgx
#SBATCH --constraint=a100_80gb

## SBATCH -c 4
## SBATCH --mem=32GB

NAME=cloudmesh-rivanna
# NAME=cloudmesh-nvidia
PROJECT_DIR="$PROJECT/osmi"
RUN_DIR="$PROJECT_DIR/machine/rivanna"
MODEL_DIR="$PROJECT_DIR/models"
# export CONTAINERDIR=/share/resources/containers/singularity

module purge

module load singularity
# module load tensorflow/2.10.0 gcc openmpi python

nvidia-smi

source $PROJECT_DIR/ENV3/bin/activate

# pip install -r $RUN_DIR/requirements-rivanna.txt

# which python

# cms gpu watch --gpu=0 --delay=0.5 --dense > $SLURM_JOBID.gpu.log &

cd $MODEL_DIR

singularity exec --nv $RUN_DIR/$NAME.sif python train.py small_lstm
# singularity exec --nv ./cloudmesh-docker.sif python train.py small_lstm

# singularity exec --nv $CONTAINERDIR/tensorflow-2.10.0.sif python train.py small_lstm
# singularity run --nv $CONTAINERDIR/tensorflow-2.10.0.sif train.py medium_cnn
# singularity run --nv $CONTAINERDIR/tensorflow-2.10.0.sif train.py large_tcnn

# cd $dir
# time singularity pull docker://tensorflow/serving:latest-gpu 
