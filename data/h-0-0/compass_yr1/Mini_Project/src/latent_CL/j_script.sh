#!/bin/bash
#
#
#SBATCH --partition=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --mem=10G
#SBATCH --array 1-2
#SBATCH --group MATH021322


# Define executable
export EXE=/bin/hostname

# Change into working directory
cd "${SLURM_SUBMIT_DIR}"

# Define list of encoders we want to use
LS=('RN50_clip' 'resnet18')

# Execute code
${EXE}

# Do some stuff
echo JOB ID: ${SLURM_JOBID}
echo SLURM ARRAY ID: ${SLURM_ARRAY_TASK_ID}
echo Working Directory: $(pwd)

echo Start Time: $(date)

# Use this to create virtual env (keeping this here for my reference later on):
# python3 -m venv ./mypyenvb
# We activate the virtual environment
source ~/tmp/mypyenvb/bin/activate
python dataset_encoder.py --pretrained_encoder 1 --regime latent_ER --dataset_name CIFAR100 --dataset_encoder_name ${LS[${SLURM_ARRAY_TASK_ID}]}


echo End Time: $(date)
