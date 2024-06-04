#!/bin/bash
#PBS -l walltime=10:30:0,nodes=1:xk:ppn=16,gres=shifter

module load shifter

shifterimg pull luntlab/cs547_project:latest

export OMP_NUM_THREADS=16
aprun -b -n 1 -d 16  -- shifter --image=docker:luntlab/cs547_project:latest --module=mpich,gpu -- python -c "import torch; print(torch.cuda.is_available());"
