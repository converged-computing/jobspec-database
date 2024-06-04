#!/bin/bash

#PBS -l walltime=10:00:00,select=1:ncpus=1:ngpus=1:mem=16gb
#PBS -J 0-20:1
#PBS -N morphot_25_gamma_vary
#PBS -A pr-kdd-1-gpu
#PBS -m abe
#PBS -M rtkushner@gmail.com
#PBS -o .outputs/output.txt
#PBS -e .outputs/error.txt

################################################################################

module load gcc cuda python3 py-pip/19.0.3-py3.7.3 openblas py-scipy 

cd $PBS_O_WORKDIR

nvidia-smi >> gpuinfo$PBS_ARRAY_INDEX.txt
lscpu >> cpuinfo$PBS_ARRAY_INDEX.txt

source ../venv/bin/activate

python3 simconv.py -fn 25 -it 100 -ds 1 -reg $(($PBS_ARRAY_INDEX/10)) -i1 data/3iyf.mrc -i2 data/3los.mrc --outdir $PBS_O_WORKDIR --cuda



