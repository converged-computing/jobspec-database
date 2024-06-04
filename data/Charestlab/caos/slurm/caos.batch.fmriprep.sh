#!/bin/bash
#SBATCH --array 1-21
#SBATCH --nodes 1
#SBATCH --ntasks 4
#SBATCH --mem 32G
#SBATCH --time 16:00:00
#SBATCH --qos bbdefault
#SBATCH --mail-type ALL


set -e
# Print basic reference information
echo "${SLURM_JOB_ID}: Job ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_TASK_MAX}"

# Create a temporary working directory at /scratch
JOBTAG="${USER}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
BB_WORKDIR=$(mktemp -d /scratch/${JOBTAG}.XXXXXX)
export TMPDIR=${BB_WORKDIR}

# Clear loaded modules and load modules for singularity
module purge
module load bluebear
module load bear-apps/2018a
module load Singularity/2.5.1-GCC-6.4.0-2.28

# print space on /scratch
df -h /scratch
# sleep 1-320 seconds
sleep $[ ( $RANDOM / 100 )  + 1 ]s
# Run fmriprep for one participant
./caos.fmriprep.sh ${SLURM_ARRAY_TASK_ID}

# Remove temporary data in working directory
ls ${BB_WORKDIR}
test -d ${BB_WORKDIR} && /bin/rm -rf ${BB_WORKDIR}
