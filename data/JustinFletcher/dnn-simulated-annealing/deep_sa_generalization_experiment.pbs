#!/bin/bash
## Walltime in hours:minutes:seconds
#PBS -l walltime=48:00:00
## -o specifies output file
#PBS -o ~/log/evaluate_model_stability.out
## -e specifies error file
#PBS -e ~/log/evaluate_model_stability.error
## Nodes, Processors, CPUs (processors and CPUs should always match)
#PBS -l select=1:mpiprocs=20:ncpus=20
## Enter the proper queue
#PBS -q standard
## MHPCC Account/Project number
#PBS -A MHPCC96670DA1 
module load anaconda2
module load tensorflow
cd ~/dnn-simulated-annealing
python deep_sa_generalization_experiment.py