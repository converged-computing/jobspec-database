#!/bin/bash

#SBATCH -J PEARLIntelE2E
#SBATCH -o e2e-inference-%j.out
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH --cpus-per-task=2
#SBATCH --gres-flags=enforce-binding
#SBATCH --ntasks-per-socket=4

# SCARF site specific
module load OpenMPI/4.0.0-GCC-8.2.0-2.31.1

# check info
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."

# Name of singularity imag to use
IMAGE=hvd-scarf.simg

# Local working directory
WORKDIR=~/data/scarf688/projects/intel-e2e-benchmark/case2

TRAIN_DATA=/work/data/one-day
TRAIN_DATA_HDF=/work/data/hdf/one-day
TEST_DATA=/work/data/ssts
OUTPUT_DIR=/work/results/run_$SLURM_JOB_ID
SST_FILE=/work/data/ssts/sst_matchups.h5
MODEL_FILE=$OUTPUT_DIR/model.h5

export SINGULARITYENV_PYTHONPATH=/work

# Run training on test data
mpirun -bind-to none -map-by slot \
       -x NCCL_DEBUG=INFO \
       -x LD_LIBRARY_PATH \
       -x PATH \
       -x HOROVOD_TIMELINE=$OUTPUT_DIR/timeline_train.json \
        singularity run --nv -B $WORKDIR:/work $IMAGE \
            python /work/e2e_benchmark/command.py train $TRAIN_DATA_HDF $OUTPUT_DIR --epochs 30 --no-cache

# Run inference on test data
mpirun -bind-to none \
       -map-by slot \
       -x NCCL_DEBUG=INFO \
       -x LD_LIBRARY_PATH \
       -x PATH \
       -x HOROVOD_TIMELINE=$OUTPUT_DIR/timeline_infer.json \
        singularity run --nv -B $WORKDIR:/work $IMAGE \
                python /work/e2e_benchmark/command.py inference $TEST_DATA $OUTPUT_DIR

# Convert training & validation data to HDF
# singularity run --nv -B ~/work/intel-e2e-benchmark:/work e2e.sif \
# 	python /work/e2e_benchmark/command.py convert-hdf $TRAIN_DATA $TRAIN_DATA_HDF

# singularity run --nv -B ~/work/intel-e2e-benchmark:/work e2e.sif \
# 	python /work/e2e_benchmark/command.py convert-hdf $SST_DATA $SST_DATA_HDF

# # Train model on training data
# singularity run --nv -B ~/work/intel-e2e-benchmark:/work e2e.sif \
# 	python /work/e2e_benchmark/command.py train $TRAIN_DATA_HDF $OUTPUT_DIR --epochs 1


# # Run SST comparison
# singularity run --nv -B ~/work/intel-e2e-benchmark:/work e2e.sif \
# 	python /work/e2e_benchmark/command.py sst-comp $SST_FILE $OUTPUT_DIR

