#!/bin/bash -l
#PBS -l select=24:system=polaris
#PBS -l place=scatter
#PBS -l walltime=3:00:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=home:eagle


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/NEXT_SparseEventID
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`

HOSTS=`cat $PBS_NODEFILE`

echo "Hosts:" 
echo $HOSTS

OVERSUBSCRIBE=4


let NRANKS_PER_NODE=4*${OVERSUBSCRIBE}


let NRANKS=${NNODES}*${NRANKS_PER_NODE}


# Turn on MPS:
# (on every rank!!)
mpiexec -n ${NNODES} -ppn 1 nvidia-cuda-mps-control -d


source /home/cadams/SS11-flash-build/setup.sh

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB



# # # For OVERSUBSCRIBE=1:
# CPU_AFFINITY=24-31:16-23:8-15:0-7
# export OMP_NUM_THREADS=8

# # For OVERSUBSCRIBE=2:
# CPU_AFFINITY=24-27:28-31:16-19:20-23:8-11:12-15:0-3:4-7
# export OMP_NUM_THREADS=4


# For OVERSUBSCRIBE=4:
CPU_AFFINITY="24-25,56-57:26-27,58-59:28-29,60-61:30-31,62-63"
CPU_AFFINITY="${CPU_AFFINITY}:16-17,48-49:18-19,50-51:20-21,52-53:22-23,54-55"
CPU_AFFINITY="${CPU_AFFINITY}:8-9,40-41:10-11,42-43:12-13,44-45:14-15,46-47"
CPU_AFFINITY="${CPU_AFFINITY}:0-1,32-33:2-3,34-35:4-5,36-37:6-7,38-39"
export OMP_NUM_THREADS=4


# for LR in 3e-1 3e-2 3e-3 3e-4;

i=1
n=2
# for OPT in novograd;
for OPT in lamb adam novograd;
do 
    # for LR in 3e-2 3e-3;
    for LR in 3e-1 3e-2 3e-3 3e-4;
    do


        let "LOCAL_RANKS=${n}*${NRANKS_PER_NODE}"
        echo $LOCAL_RANKS
        LOCAL_BATCH_SIZE=128
        let "GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${LOCAL_RANKS}"

        echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

        hosts=$(cat $PBS_NODEFILE | tail -n +$i | head -n 2)
        hosts=$(echo ${hosts} | tr -s " " ",")
        echo $hosts
        echo ""
        run_id=repr-mb${GLOBAL_BATCH_SIZE}-${OPT}-${LR}
        echo $run_id

        mpiexec -n ${LOCAL_RANKS} -ppn ${NRANKS_PER_NODE} \
        --hosts ${hosts} \
        --cpu-bind list:${CPU_AFFINITY} \
        python bin/exec.py \
        --config-name learn_rep \
        mode=train \
        framework.oversubscribe=${OVERSUBSCRIBE} \
        mode.optimizer.lr_schedule.peak_learning_rate=${LR} \
        mode.optimizer.name=${OPT} \
        run.distributed=True \
        run.profile=True \
        run.id=${run_id} \
        run.minibatch_size=${GLOBAL_BATCH_SIZE} \
        run.length=500  & 

        let "i=i+n"
    done    
done
echo ""

wait
