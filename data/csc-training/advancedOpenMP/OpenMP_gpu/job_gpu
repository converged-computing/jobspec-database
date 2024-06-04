#!/bin/bash -l
#SBATCH -J omp_gpu        # Job name
#SBATCH -o %x.out%j       # Name of stdout & stderr file
#SBATCH -p dev-g          # Partition (queue) name
#SBATCH -N 1              # one node
#SBATCH -n 1              # tasks
#SBATCH --cpus-per-task=3 
#SBATCH -t 00:05:00       # Run time (hh:mm:ss)
#SBATCH -A project_462000390  # Project account for billing
#SBATCH --gres=gpu:mi250:1
#

export OMP_NUM_THREADS=3
ml LUMI/23.03
ml partition/G craype-accel-amd-gfx90a craype-x86-trento rocm

export CRAY_ACC_DEBUG=2   # use 1 for less, or 3 for FULL
                          #                 to see everything!
#rocprof ./a.out   #you can uncomment this line to have the profiler run your program and check some kernel execution parameters
./a.out
