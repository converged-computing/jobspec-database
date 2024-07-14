#!/bin/bash

#SBATCH -o "./hpc_output_pqd.log"   # Output-File
#SBATCH -D .                        # Working Directory
#SBATCH -J nima-pqd                 # Job Name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:tesla:1          # request one GPU

#SBATCH --time=09:00:00 # expected runtime
#SBATCH --partition=gpu

#Job-Status per Mail:
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mayet@campus.tu-berlin.de

#Using base environment.
source ~/miniconda3/bin/activate base
module load nvidia/cuda/10.1
./train.sh $1 | tee > hpc_job_$1.log

# To run do
# sbatch hpc_gpu_job.sh anv/anf/pqd
