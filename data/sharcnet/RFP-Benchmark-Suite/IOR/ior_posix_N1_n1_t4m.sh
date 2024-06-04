#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=8g
#SBATCH -A cc_debug
#SBATCH -t 01:00:00

FS=scratch
CASE="posix" # e.g. posix.odirect-F
API="posix"
API_OPTIONS="" # e.g. --posix.odirect
DATA_TYPE="-l=random" # e.g "-l=incompressible" "-l=random"
MEMORY_PER_NODE="" # e.g. "-M=70%"
FILE_PER_TASK="-F" # -F
TASK_PER_NODE_OFFSET="" # e.g. "-Q=$SLURM_NTASKS_PER_NODE"
BLOCK_SIZE=4m
TRANSFER_SIZE=4m
SEGMENTS=40000
ITERATIONS=3
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
NODES=$SLURM_JOB_NUM_NODES
TASKS_PER_NODE=$SLURM_TASKS_PER_NODE
IOR=/home/hahn/bin/ior
RUN_DIR=/scratch/hahn/wide4
DATA_DIR="$RUN_DIR"/datafiles
if [ -d "$DATA_DIR" ]; then
  rm -f "$DATA_DIR"/*
else
  mkdir "$DATA_DIR"
fi

cmd="mpirun $IOR -o "$DATA_DIR"/ior_test $DATA_TYPE -b=$BLOCK_SIZE -t=$TRANSFER_SIZE -i=$ITERATIONS -g -d=10 -e -C $TASK_PER_NODE_OFFSET -s $SEGMENTS $FILE_PER_TASK"
echo $cmd

{
echo "-------------------STARTING SIMULATION--------------------"
date
echo
echo $cmd
echo
$cmd

echo "-------------------ENDING SIMULATION--------------------"
date 

} &>> run_"${FS}"_FS_"${CASE}"_case_"${NODES}"_nodes_"${TASKS_PER_NODE}"_ppn_"${BLOCK_SIZE}"_block_"${TRANSFER_SIZE}"_transfer_"${TIMESTAMP}".out

