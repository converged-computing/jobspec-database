#!/bin/bash
#SBATCH -J helloWorld						  # name of job
#SBATCH -o helloWorld.out				  # name of output file for this submission script
#SBATCH -e helloWorld.err				  # name of error file for this submission script

# load any software environment module required for app (e.g. matlab, gcc, cuda)
module load gcc/12.2
module load R/4.2.2


# run my job (e.g. matlab, python)
Rscript ../NWSimulation.R