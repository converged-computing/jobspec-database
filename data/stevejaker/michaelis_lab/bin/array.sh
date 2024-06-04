#!/bin/bash

#SBATCH -C 'avx2'
#SBATCH --time=00-00:30         # time (DD-HH:MM)
#SBATCH --nodes=1 --ntasks-per-node=16
#SBATCH --mem-per-cpu=400M   # memory per CPU core
#SBATCH --mail-user=jparkman@byu.edu   # email address
#SBATCH --mail-type=FAIL
#SBATCH --requeue
# #SBATCH --qos=test
#SBATCH --array=1-16%1 

LAST_PIECE=16

if [[ "${SLURM_ARRAY_TASK_ID}" -eq "1" ]]; then
        # If this is ARRAY_TASK_ID 1, sander1, we pass our Arguments here so we can run leap corectly

        runamber $SLURM_ARRAY_TASK_ID $@

elif [[ "${SLURM_ARRAY_TASK_ID}" -eq "${LAST_PIECE}" ]]; then
        runamber $SLURM_ARRAY_TASK_ID -L

else
        # If it is not ARRAY_TASK_ID 1, just run the next job in the line
        runamber $SLURM_ARRAY_TASK_ID
fi
