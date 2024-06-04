#! /bin/bash

# You need to set the appropriate SBATCH_PARTITION, SBATCH_ACCOUNT, and USE_STEPS4 mentioned in HBP_STEPS/doc/dev/README.md

#SBATCH --array=[1-100%10]
#SBATCH --nodes=32
#SBATCH --time=4:00:00
#SBATCH --exclusive
#SBATCH --mem=0

set -x

module load unstable python-dev python
## load steps by hash
export PYTHONPATH=`spack find --paths /$SPACK_INSTALLED_HASH | tail -n 1 | grep -o "/.*"`:$PYTHONPATH

if [[ -z "${steps_version}" ]]
then
  steps_version=4
fi

nodes=$SLURM_JOB_NUM_NODES
ntasks=$(($nodes * 32))
seed=$(($SLURM_ARRAY_TASK_ID * 1))
#switch these 2 lines once https://github.com/CNS-OIST/HBP_STEPS/issues/1008 is solved
mesh_fls=../../mesh/split_1024/steps3/CNG_segmented_2_split_1024.msh
#mesh_fls=../../mesh/split_1024/steps$steps_version/CNG_segmented_2_split_1024

#dplace is making caburst fail. Readd it once https://github.com/CNS-OIST/HBP_STEPS/issues/1022 is fixed
time srun --nodes=$nodes --ntasks=$ntasks \
python caBurstFullModel.py $seed $mesh_fls $steps_version


