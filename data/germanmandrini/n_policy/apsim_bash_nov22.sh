#!/bin/bash
#PBS -l nodes=1:ppn=28 ## Specify how many nodes/cores per node
##PBS -l walltime=00:01:00
#PBS -q secondary ## Specify which queue to run in

cd /projects/aces
module load singularity ## Load the singularity runtime to your environment
# cell_n=$1 #split the cell string for bash work
cell_n=${cell_n} #for qsub


## Code to get the cell_n from qsub or from bash
## if [ -z ${cell+x} ]; then echo "var is unset"; else echo "var is set to '$cell'"; fi
## if [ -z ${cell+x} ]; then cell_n=$1; cell_n=${cell_n};fi

echo "$cell_n"

singularity exec /projects/aces/germanm2/apsim_nov16.simg Rscript /projects/aces/germanm2/n_policy_git/Codes/server_apsim_call.R $cell_n



