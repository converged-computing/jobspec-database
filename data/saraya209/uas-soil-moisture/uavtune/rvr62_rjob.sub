#!/bin/bash
##
#SBATCH --partition=fast.q  ## queue based on wall-clock time limitation
#SBATCH --nodes=1 ## or "-N". Min no. of nodes (eg. 3)
#SBATCH --ntasks-per-node=24 ## Max. tasks per node.
##
#SBATCH --job-name=rvr62 ## Name of Job in queue
#SBATCH --mail-user=saraya@ucmerced.edu
#SBATCH --mail-type=ALL
module load openmpi-2.0/intel
module load anaconda3
source activate my-R
mpirun -np 1 --bind-to none R CMD BATCH --no-save RVR62.R
