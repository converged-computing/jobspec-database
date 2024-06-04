#!/bin/bash
#SBATCH --time=00:60:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=16

module load openmpi/gcc/64/4.0.2
module load julia/1.7.3

pwd

# Parallel

mpiexec -np 4 julia --project=. -e '
    include("solution.jl")
    for m in 1:3
        main_run(R=5,N=3200,method=m)
    end
'



# Baseline
julia --project=. -e '
    include("solution.jl")
    main_run(R=5,N=3200,method=0)
'
