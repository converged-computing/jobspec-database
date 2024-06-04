#!/bin/bash

#SBATCH --job-name=RT-kinetic-reference

#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=3
#SBATCH --cpus-per-task=1
#SBATCH --array=1-9%3
#SBATCH --mem=20G
#SBATCH --time=96:00:00

#SBATCH --chdir=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/RT_kinetic_reference/
#SBATCH --output=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/RT_kinetic_reference//sims/RT-%a/sim.log
#SBATCH --error=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/RT_kinetic_reference//sims/RT-%a/sim.err

module use ~/modulefiles
module load julia

export OPENBLAS_NUM_THREADS=1

julia --version
julia --project=../.. --startup-file=no main.jl --run $SLURM_ARRAY_TASK_ID
