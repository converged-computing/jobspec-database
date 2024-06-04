#!/bin/bash
#----------------------------------------------------

#SBATCH -J NVIDIASMI_GPULab                # Job name
#SBATCH -o runjob/NVIDIASMI_GPULab.o%j     # Name of stdout output file
#SBATCH -e runjob/NVIDIASMI_GPULab.e%j     # Name of stderr error file
#SBATCH -p gpu-a100                        # Queue (partition) name (use gpu-a100 for job >2 hours, use gpu-a100-dev for job <2 hours)
#SBATCH -N 1                               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 01:59:59                        # Run time (hh:mm:ss) !! Adjust the Time !!
#SBATCH --mail-user=                       # Put your email
#SBATCH --mail-type=all                    #
#SBATCH -A                                 # Put your allocation name

# Other commands must follow all #SBATCH directives.
source set_environment

# Example
power/power_measure.sh "nvidia-smi --id=0 --query-gpu=timestamp,temperature.gpu,power.draw --format=csv --filename=testpower.csv --loop=1" "kernel/bin/gemm_cuda_bench -M fp16 -A fp16 -I 100 32768 32768 32768"
