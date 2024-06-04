#!/bin/bash -l

#SBATCH -J factorialFunctionSimple                          # Job name
#SBATCH -o logs/factorialFunctionSimple.%j.out              # Name of stdout output file (%j expands to jobId)
#SBATCH -e logs/factorialFunctionSimple.%j.err              # Error File Name 
#SBATCH --mail-type=begin                      # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=lhw150030@utdallas.edu     # Where to send mail	
#SBATCH -N 1                                   # Total number of nodes requested
#SBATCH -n 1                                   # Total number of mpi tasks requested
#SBATCH -t 00:01:00                            # Run time (hh:mm:ss) - 1 minute

echo Running on host: `hostname`
julia ex_02_factorial_function_simple.jl
