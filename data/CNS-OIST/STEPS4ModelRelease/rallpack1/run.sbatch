#! /bin/bash

# You need to set the appropriate SBATCH_PARTITION, SBATCH_ACCOUNT, and USE_STEPS4 mentioned in HBP_STEPS/doc/dev/README.md

#SBATCH --array=[1]
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --mem=0
#SBATCH --exclusive

set -x

module load unstable python-dev python
# load steps by hash
export PYTHONPATH=`spack find --paths /$SPACK_INSTALLED_HASH | tail -n 1 | grep -o "/.*"`:$PYTHONPATH

nodes=$SLURM_JOB_NUM_NODES
ntasks=$(($nodes * 32))
seed=$(($SLURM_ARRAY_TASK_ID * 1))

if [[ -z "${steps_version}" ]]
then
  steps_version=4
fi

time srun --nodes=$nodes --ntasks=$ntasks dplace \
python rallpack1.py $seed mesh/axon_cube_L1000um_D866nm_1135tets.msh $steps_version
