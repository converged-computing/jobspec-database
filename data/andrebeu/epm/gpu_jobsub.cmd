#!/bin/bash

#SBATCH --gpu-accounting

#SBATCH -t 24:00:00			# runs for 48 hours (max)  
#SBATCH -c 8				# number of cores 4
#SBATCH -N 1				# node count 
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks-per-socket=2
#SBATCH --gres=gpu:1		# number of gpus 4

printf "\n\n\n --ntasks-per-node=1 -c=8 ntasks-per-socket=4 \n\n\n"

CELLSIZE=${1}

module load anaconda3/4.4.0
module load cudnn/cuda-9.1/7.1.2
module load openmpi/gcc/2.1.0/64 # srm


printf "\n\n NBack Task \n\n"


srun python -u "/tigress/abeukers/wd/epm/train_and_save.py" ${CELLSIZE} 


printf "\n\nGPU profiling \n\n"
sacct --format="elapsed,CPUTime,TotalCPU"
nvidia-smi --query-accounted-apps=gpu_serial,gpu_utilization,mem_utilization,max_memory_usage --format=csv

# mv slurm*.out slurm_out/