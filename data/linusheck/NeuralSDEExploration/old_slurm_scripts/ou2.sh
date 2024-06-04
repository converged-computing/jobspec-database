#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --time=16:00:00
#SBATCH --output=/home/linushe/outputs/plain-%j.log

###
# This script is to submitted via "sbatch" on the cluster.
#
# Set --cpus-per-task above to match the size of your multiprocessing run, if any.
###

echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "Running on nodes: $SLURM_NODELIST"
echo "------------------------------------------------------------"

# Some initial setup
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
module purge

# Load a Julia module, if you're running Julia notebooks
# module load julia/1.8.2


# time dependent OU
srun /home/linushe/julia-1.9.0/bin/julia --project=. -t16 notebooks/sde_train.jl -m ou --batch-size 128 --eta 0.3 --learning-rate 0.04 --latent-dims 3 --stick-landing false --kl-rate 500 --kl-anneal true --lr-cycle false --tspan-start-data 0.0 --tspan-end-data 40.0 --tspan-start-train 0.0 --tspan-end-train 80.0 --tspan-start-model 0.0 --tspan-end-model 40.0 --dt 2.0 --hidden-size 64 --depth 2 --backsolve true --scale 0.01 --decay 1.0
