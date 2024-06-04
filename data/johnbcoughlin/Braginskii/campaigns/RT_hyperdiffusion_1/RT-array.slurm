#!/bin/bash

#SBATCH --job-name=RT-hyperdiffusion-1

#SBATCH --account=amath
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=6
#SBATCH --cpus-per-task=2
#SBATCH --array=1-5%6
#SBATCH --mem=20G
#SBATCH --time=48:00:00

#SBATCH --chdir=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/RT_hyperdiffusion_1/
#SBATCH --output=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/RT_hyperdiffusion_1/sims/RT-%a/sim.log
#SBATCH --error=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/RT_hyperdiffusion_1/sims/RT-%a/sim.err

module use ~/modulefiles
module load julia

export OPENBLAS_NUM_THREADS=1

julia --version
julia --project=../.. --startup-file=no main.jl --run $SLURM_ARRAY_TASK_ID
