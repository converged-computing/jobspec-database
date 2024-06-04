#!/bin/bash

#SBATCH --job-name=vorticity_heat_flux

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --array=4
#SBATCH --mem=20G
#SBATCH --time=24:00:00

#SBATCH --chdir=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/vorticity_heat_flux/
#SBATCH --output=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/vorticity_heat_flux//sims/SF-%a/sim.log
#SBATCH --error=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/vorticity_heat_flux//sims/SF-%a/sim.err

module use ~/modulefiles
module load julia

export OPENBLAS_NUM_THREADS=1

julia --version
julia --project=../.. --startup-file=no main.jl --run $SLURM_ARRAY_TASK_ID

