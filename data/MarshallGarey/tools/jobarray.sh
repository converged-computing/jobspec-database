#!/bin/bash
ARRAY_SIZE=${1:-10}
echo "ARRAY_SIZE=$ARRAY_SIZE"
JOB_ID=$(sbatch --begin=now --parsable --array 1-${ARRAY_SIZE} --wrap="srun /bin/true" -J job_array --output=/dev/null)
#JOB_ID=$(sbatch --begin=now --parsable --array 1-${ARRAY_SIZE} --wrap="srun sleep 10" -J job_array --output=/dev/null)
echo "Job id = $JOB_ID"
#sbatch --wrap="srun whereami"
#LAST_JOB_ID=$(sbatch --depend=afterany:${JOB_ID} --wrap="srun whereami" -J wait_barrier -D tmp)
#sbatch --depend=afterok:${JOB_ID} --wrap="whereami" -J wait_barrier -D tmp
