#!/bin/bash

#SBATCH --job-name=RT-hybrid-vlasov-comparison-arrayjob

#SBATCH --account=amath
#SBATCH --partition=gpu-rtx6k
#SBATCH --nodes=1
#SBATCH --gpus-per-task=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=2
#SBATCH --array=1-2%2
#SBATCH --mem=20G
#SBATCH --time=24:00:00

#SBATCH --chdir=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/RT_hybrid_vlasov_comparison/
#SBATCH --output=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/RT_hybrid_vlasov_comparison//sims/RT-%a/sim.log
#SBATCH --error=/mmfs1/gscratch/aaplasma/johnbc/projects/Braginskii/campaigns/RT_hybrid_vlasov_comparison//sims/RT-%a/sim.err

module use ~/modulefiles
module load julia

export OPENBLAS_NUM_THREADS=1

julia --version
julia --project=../.. --startup-file=no main.jl --run $SLURM_ARRAY_TASK_ID
