#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=def-arguinj
#SBATCH --time=02-00:00         #time limit (DD-HH:MM)
##SBATCH --mem=64G               #memory per node (Beluga)
##SBATCH --cpus-per-task=4       #CPUs threads per node (Beluga)
#SBATCH --gres=gpu:1            #GPUs per node
#SBATCH --job-name=OE_AAE
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0
#---------------------------------------------------------------------

export  SLURM_ID=$SLURM_ARRAY_TASK_ID
export HOST_NAME=$SLURM_SUBMIT_HOST

if [[ $HOST_NAME == *atlas* ]]
then
    # TRAINING ON LPS
    if   [[ -d "/nvme1" ]]
    then
	PATHS=/lcg,/opt,/nvme1
    else
	PATHS=/lcg,/opt
    fi
    SIF=/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif
    singularity shell --nv --bind $PATHS $SIF train.sh $SLURM_ID $HOST_NAME
else
    # TRAINING ON BELUGA
    module load singularity/3.7
    PATHS=/project/def-arguinj
    SIF=/project/def-arguinj/shared/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif
    singularity shell --nv --bind $PATHS $SIF < train.sh $SLURM_ID $HOST_NAME
fi

mkdir -p outputs/log_files
mv *.out outputs/log_files 2>/dev/null
