#!/bin/bash
#SBATCH -N 1
#SBATCH -t 20:00:00

module load 2019 Anaconda3
source activate my_root

#export R_LIBS=$HOME/rpackages:$R_LIBS

cp -r "$HOME"/ARVARSim "$TMPDIR"
cd "$TMPDIR"/ARVARSim

echo $SLURM_ARRAY_TASK_ID

Rscript --vanilla Sim_LISA.R $SLURM_ARRAY_TASK_ID

cp -r ./*.RDS "$HOME"/ARVARSim/output