#!/bin/bash
#SBATCH -J MViTv2_finetune_STAR
#SBATCH -o ../exp/%x/%j_%x.out
#SBATCH -e ../exp/%x/%j_%x.err

#SBATCH --mail-user=yushoubin26@163.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --exclusive
###SBATCH -p sched_system_all

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=slowfast
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited


## Number of total processes
echo " "
echo "Job name" $SLURM_JOB_NAME
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

### distributed run
#We want names of master and slave nodes
MASTER=`/bin/hostname -s`
SLAVES=`scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $MASTER`
#Make sure this node (MASTER) comes first
HOSTLIST="$MASTER $SLAVES"

#Get a random unused port on this host(MASTER) between 2000 and 9999
#First line gets list of unused ports
#2nd line restricts between 2000 and 9999
#3rd line gets single random port from the list
#MPORT=`ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[2-9][0-9]\{3,3\}" | grep -v "[0-9]\{5,5\}" | sort | uniq | shuf|head -n 1`
MPORT=4002
RANK=0


PWD=`pwd`
DATA_ROOT="/nobackup/"
#ENV_VAR="export PYTHONPATH=/nobackup/users/bowu/code/STAR_code/STAR_Action/code/slowfast/slowfast:$PYTHONPATH"
#echo $ENV_VAR

CMD="cd $PWD && python ../code/slowfast/tools/run_net.py --cfg ../code/slowfast/configs/STAR/MVITv2_B_32x3.yaml --opts NUM_SHARDS ${SLURM_JOB_NUM_NODES} NUM_GPUS 4 OUTPUT_DIR ./"${SLURM_JOB_NAME}$"/ "
CMD=${CMD}"TRAIN.CHECKPOINT_EPOCH_RESET True TRAIN.CHECKPOINT_FILE_PATH ../exp/MViTv2_finetune_K400/checkpoints/checkpoint_epoch_00060.pyth MODEL.NUM_CLASSES 111 "
CMD=${CMD}"DATA.PATH_TO_DATA_DIR "${DATA_ROOT}"users/bowu/data/STAR/Situation_Video_Data/ DATA.PATH_PREFIX "${DATA_ROOT}"projects/public/charades/Charades_v1_rgb/ "
echo "Basic commend:"$CMD

#Launch the pytorch processes, first on master (first in $HOSTLIST) then
#on the slaves
echo "master node is $MASTER, master port is $MPORT, slave nodes are $SLAVES"
for node in $HOSTLIST; do
        node_cmd=${CMD}" SHARD_ID $RANK"
        echo $node_cmd
        ssh -q $node $node_cmd &
        if [ $RANK == 0 ]; then
             count=0
             #while ! nc -zvw3 $MASTER $MPORT >/dev/null 2>&1; do
             #     sleep 5
             #     count=$((count+1))
             #     if [ $count == 360 ];then
             #          echo "wait 30 min for master start... exit..."
             #     fi
             #done
        fi
        RANK=$((SLURM_NTASKS_PER_NODE+RANK))
done
wait

echo "Run completed at:- "
date

