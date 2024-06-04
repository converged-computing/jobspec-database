#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=slurm_run
#SBATCH --partition=long
#SBATCH --time=3-00:00:00
#SBATCH --mem=50GB
#SBATCH --exclude=amdgpu1,amdgpu2,xcna0,xgpd9
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/n/nguyenpk/CS6220/project/NLP_DS_distance/SLURM_SCRIPT/log/%A_%a.log
#SBATCH --error=/home/n/nguyenpk/CS6220/project/NLP_DS_distance/SLURM_SCRIPT/err/err.%A_%a
#SBATCH --array=0-10


ulimit -s 10240
ulimit -u 100000
srun -N 1 --ntasks-per-node=1 ./script.sh ${SLURM_ARRAY_TASK_ID} 
