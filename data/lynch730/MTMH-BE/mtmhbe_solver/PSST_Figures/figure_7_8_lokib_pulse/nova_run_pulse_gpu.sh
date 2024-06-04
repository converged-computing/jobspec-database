#!/bin/bash

#SBATCH --time=00:45:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=8   # 8 processor core(s) per node 
#SBATCH --mem=32G   # maximum memory per node
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=gpu    # gpu node(s)
#SBATCH --job-name="pulse_test"

module load matlab/R2022a
cd /home/lynch/boltzmann_solvers/mtmhbe_solver/performance/pulse_loki_b/
matlab -nodisplay -nosplash -nodesktop -r "time_convergence;exit;"
