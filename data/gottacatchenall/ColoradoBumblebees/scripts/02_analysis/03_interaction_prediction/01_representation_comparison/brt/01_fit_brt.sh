#!/bin/bash
#SBATCH --account=def-gonzalez
#SBATCH --job-name=brt_fit 
#SBATCH --output=slurm-brt_fit-%A.%a.out 
#SBATCH --nodes=1               
#SBATCH --ntasks=1               
#SBATCH --cpus-per-task=1        
#SBATCH --mem-per-cpu=16G      
#SBATCH --time=03:00:00         
#SBATCH --array=1-31

module load cuda
module load julia/1.8.5
module load cudnn 

export JULIA_DEPOT_PATH="/project/def-gonzalez/mcatchen/JuliaEnvironments/COBees"
export CLUSTER="true"

julia fit_brt.jl
