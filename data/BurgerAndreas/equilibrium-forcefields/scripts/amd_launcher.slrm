#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --job-name=equiformer 
#SBATCH --output=outslurm/slurm-%j.txt 
#SBATCH --error=outslurm/slurm-%j.txt
# errslurm-%j.err

# module load python pytorch
# module load miniforge3 # miniconda3 miniforge3

echo `date`: Job $SLURM_JOB_ID is allocated resources.
echo "Inside slurm_launcher.slrm ($0). received arguments: $@"

HOME_DIR=/home/andreasburger

# export SCRIPTDIR=${HOME_DIR}/equilibrium-forcefields/equilibrium-forcefields/train
export SCRIPTDIR=${HOME_DIR}/equilibrium-forcefields/equiformer
if [[ $1 == *"test"* ]]; then
    echo "Found test in the filename. Changing the scriptdir to equilibrium-forcefields/tests"
    export SCRIPTDIR=${HOME_DIR}/equilibrium-forcefields/tests
elif [[ $1 == *"deq"* ]]; then
    echo "Found deq in the filename. Changing the scriptdir to equilibrium-forcefields/scripts"
    export SCRIPTDIR=${HOME_DIR}/equilibrium-forcefields/scripts
fi

# hand over all arguments to the script
echo "Submitting ${SCRIPTDIR}/$@"

${HOME_DIR}/miniforge3/envs/deq/bin/python ${SCRIPTDIR}/"$@"