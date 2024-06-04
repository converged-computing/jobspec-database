#!/bin/bash -l
#PBS -l select=3:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug-scaling
#PBS -A datascience
#PBS -l filesystems=home:eagle


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/NEXT_SparseEventID
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`


OVERSUBSCRIBE=4
let NRANKS_PER_NODE=4*${OVERSUBSCRIBE}


# Turn on MPS:
# (on every rank!!)
mpiexec -n ${NNODES} -ppn 1 nvidia-cuda-mps-control -d



let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=256
# let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

# Set up software deps:
# module load conda/2023-10-04
# conda activate

# Add-ons from conda:
# source /home/cadams/Polaris/polaris_conda_2023-10-04-venv/bin/activate

# module load cray-hdf5/1.12.1.3

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB

# For OVERSUBSCRIBE=4:
CPU_AFFINITY="24-25,56-57:26-27,58-59:28-29,60-61:30-31,62-63"
CPU_AFFINITY="${CPU_AFFINITY}:16-17,48-49:18-19,50-51:20-21,52-53:22-23,54-55"
CPU_AFFINITY="${CPU_AFFINITY}:8-9,40-41:10-11,42-43:12-13,44-45:14-15,46-47"
CPU_AFFINITY="${CPU_AFFINITY}:0-1,32-33:2-3,34-35:4-5,36-37:6-7,38-39"
export OMP_NUM_THREADS=4


WEIGHT_PREFIX=/home/cadams/Polaris/NEXT_SparseEventID/output/



i=1
n=1
for OPT in adam lamb novograd;
do 
    for LR in  3e-3;
    # for LR in 3e-1 3e-2 3e-3 3e-4;
    do

        weight_id=repr-mb4096-${OPT}-${LR}
        checkpoint=$(ls ${WEIGHT_PREFIX}/${weight_id}/checkpoints)
        echo "Checkpoint: ${checkpoint}"

        WEIGHTS=${WEIGHT_PREFIX}/${weight_id}/checkpoints/${checkpoint}
        echo "Weights: ${WEIGHTS//=/\\=}"



        let "LOCAL_RANKS=${NRANKS_PER_NODE}"
        # let "LOCAL_RANKS=${n}*${NRANKS_PER_NODE}"
        echo $LOCAL_RANKS
        LOCAL_BATCH_SIZE=256
        let "GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}"
        # let "GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${LOCAL_RANKS}"


        rep_opt=rmsprop

        run_id=repr_class-${rep_opt}_mb${GLOBAL_BATCH_SIZE}-${OPT}-${LR}
        echo $run_id

        echo "Global batch size: ${GLOBAL_BATCH_SIZE}"


        # mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
        # --cpu-bind list:${CPU_AFFINITY} \
        python bin/exec.py \
        --config-name supervised_eventID \
        mode=train \
        encoder=convnet \
        mode.weights_location=${WEIGHTS//=/\\=} \
        mode.optimizer.loss_balance_scheme=none \
        mode.optimizer.name=${rep_opt} \
        mode.optimizer.lr_schedule.peak_learning_rate=0.03 \
        run.distributed=True \
        run.id=${run_id} \
        framework.oversubscribe=4 \
        run.minibatch_size=${GLOBAL_BATCH_SIZE} \
        run.length=25

    done    
done
# 
