#!/bin/bash

#PBS -l walltime=10:00:00,select=1:ncpus=1:ngpus=20:mem=48gb
#PBS -N morphot_parallel_benchmark
#PBS -A pr-kdd-1-gpu
#PBS -m abe
#PBS -M rtkushner@gmail.com
#PBS -o output.txt
#PBS -e output.txt

################################################################################

module load gcc cuda python3 py-pip/19.0.3-py3.7.3 openblas py-scipy parallel

cd $PBS_O_WORKDIR

nvidia-smi >> gpuinfo.txt
lscpu >> cpuinfo.txt

source ../venv/bin/activate

parallel 'python3 simconv.py -fn 25 -it {1} -ds 1 -reg {2} -i1 data/3iyf.mrc -i2 data/3los.mrc --outdir $PBS_O_WORKDIR --cuda' ::: 50 100 200 ::: 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4



