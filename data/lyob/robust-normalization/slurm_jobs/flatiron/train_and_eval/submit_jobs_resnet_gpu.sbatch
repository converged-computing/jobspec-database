#!/bin/bash
#SBATCH -n 1 # one node
#SBATCH -p gpu --gpus=1 -c 12 --gpus-per-task=1 --constraint=a100

#SBATCH -t 1:00:00
#SBATCH --mem=32G
#SBATCH --job-name=resnorm
#SBATCH --output ./out/resnet/out.%j.%a.%N.out 
#SBATCH -e ./err/resnet/err.%j.%a.%N.err

ARRAY_ID=$1
PARAMETER_FILE=$2

let FILE_LINE=($ARRAY_ID + $SLURM_ARRAY_TASK_ID)

echo "Line ${FILE_LINE}"
# Get the relevant line from the parameters
PARAMETERS=$(sed "${FILE_LINE}q;d" ${PARAMETER_FILE})
# PARAMETERS=${PARAMETER_FILE}
echo ${PARAMETERS}

bash run_code_resnet.sh ${PARAMETERS}
