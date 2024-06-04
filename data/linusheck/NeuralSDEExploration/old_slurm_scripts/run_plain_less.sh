#!/bin/bash

#SBATCH --job-name=train
#SBATCH --account=tipes
#SBATCH --partition=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
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

module add texlive

# Start the script
srun /home/linushe/julia-1.9.0/bin/julia --project=. -t2 notebooks/sde_train.jl -m sun --batch-size 128 --eta 10.0 --learning-rate 0.02 --latent-dims 1 --stick-landing false --dt 0.1 --kl-rate 4000 --noise 0.2

# To stop the script, use 'scancel'
