#!/bin/bash                                                                                                                    
#SBATCH --job-name=SBATCH_SCORATES_GRID_optimize
#SBATCH --output=/home/jbenjami/Research_Schneider/CliMa/TrainTau.jl/sbatch_grid_optimize.out
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mail-user=jbenjami@caltech.edu
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1


module load julia/1.9.3
julia --project=/home/jbenjami/Research_Schneider/CliMa/TrainTau.jl/ -e 'include("/home/jbenjami/Research_Schneider/CliMa/TrainTau.jl/src/TrainTau.jl");  TrainTau.grid_search_optimize();'

# we should probably change this to use the depot from TurbulenceConvection.jl or something... it's installing everything in .julia when you call this...

