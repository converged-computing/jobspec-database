#!/bin/bash
### job parameters
#SBATCH --job-name=Watershed
#SBATCH -o ./logs/Watershed-%j.out
#SBATCH -p short
#SBATCH --constraint="skl-compat"
#SBATCH --cpus-per-task=3
#SBATCH --array=1-912%10
#SBATCH --requeue
SUBJECT_LIST=./subjectsWithParietalPeak.txt


module load ConnectomeWorkbench/1.4.2-rh_linux64
module load Python/3.9.6-GCCcore-11.2.0
source /well/margulies/users/mnk884/python/postHoc-permutations-skl/bin/activate

echo Executing task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on `hostname` as user ${USER} 
### each subject forms one job of the array job

echo "smoothing kernel is" ${smooth_kernel}

####get file name 
echo the job id is $SLURM_ARRAY_JOB_ID
FILENAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $SUBJECT_LIST)
echo echo $SLURM_ARRAY_JOB_ID
echo "Processing subject $FILENAME"

# Load a recent python module


SUBJECT=$(sed -n "${SGE_TASK_ID}p" $SUBJECT_LIST)
echo python3 -u IndividualWatershed.py $SUBJECT
python3 -u IndividualWatershed.py $FILENAME
