#!/bin/bash

### job parameters
#SBATCH --job-name=GradDist
#SBATCH -o ./logs/gradDistJob-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=1
#SBATCH --array=1-972:1
#SBATCH --requeue
SUBJECT_LIST=./results/CleanSujects4Dist.txt
#####load workbench
module load ConnectomeWorkbench/1.4.2-rh_linux64
#### load python
module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/measureDist-skylake/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
### each subject forms one job of the array job

echo "smoothing kernel is" ${smooth_kernel}

####get file name 
echo the job id is $SLURM_ARRAY_JOB_ID
FILENAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo echo $SLURM_ARRAY_JOB_ID
echo "Processing subject $FILENAME"

# Load a recent python module


python -u CalcGrad2CortDist.py $FILENAME 