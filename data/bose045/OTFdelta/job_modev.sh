#!/bin/bash

#SBATCH -J NNmd
#SBATCH -p npl-2024
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:6
#SBATCH -o modev_jobfile.%j
#SBATCH --time=00:15:00

if [ -z "$1" ]; then
    echo "Usage: $0 <number>"
    exit 1
fi

iter=$(printf "%04d" $1)
id=$SLURM_ARRAY_TASK_ID
output_dir="${iter}/NNmd/sys${id}"
#scontrol update JobId=${SLURM_JOB_ID}_${id} StdOut="${output_dir}/jobfile.${SLURM_JOB_ID}_${id}"
#scontrol update JobId=$SLURM_ARRAY_JOB_ID StdOut="${output_dir}/jobfile.${SLURM_ARRAY_JOB_ID}"
#cd "${iter}/NNmd/sys${id}" || exit
cd "$output_dir" || exit
curdir=$(pwd)
echo "Current directory: $curdir"
python ../../../pybash/modelDev.py ${id}


