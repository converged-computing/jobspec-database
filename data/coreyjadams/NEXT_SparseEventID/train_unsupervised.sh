#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:45:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:grand


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/NEXT_SparseEventID
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

# NRANKS=1
LOCAL_BATCH_SIZE=512
# let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

# Set up software deps:
module load conda/2022-09-08
conda activate

# Add-ons from conda:
source /home/cadams/Polaris/polaris_conda_2022-09-08-venv/bin/activate

module load cray-hdf5/1.12.1.3

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB

LOSS_BALANCE=even
OPT=adam
NORM=batch

run_id=supervised_ID_mk_mb${GLOBAL_BATCH_SIZE}-${LOSS_BALANCE}-${OPT}-${NORM}-dev
# run_id=supervised_ID_mk_mb${GLOBAL_BATCH_SIZE}-${LOSS_BALANCE}-${OPT}-${NORM}-head-smallerSize

WEIGHTS="/home/cadams/Polaris/NEXT_SparseEventID/output/repr_mb8192-adam-3e-3/checkpoints/epoch=158-step=10300.ckpt"

# mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
--config-name unsupervised_eventID \
mode=train \
framework.oversubscribe=1 \
run.distributed=True \
run.id=${run_id} \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
mode.optimizer.loss_balance_scheme=${LOSS_BALANCE} \
mode.optimizer.lr_schedule.peak_learning_rate=3e-3 \
mode.optimizer.name=${OPT} \
mode.weights_location=${WEIGHTS//=/\\=} \
output_dir=output-test \
run.length=50

