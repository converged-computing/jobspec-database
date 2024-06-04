#!/bin/bash

#SBATCH -c 8

#SBATCH -p nvidia
#SBATCH --gres=gpu:2

#Max wallTime for the job
#SBATCH -t 48:00:00


#Resource requiremenmt commands end here

# Output and error files
#SBATCH -o Errors/job.%J.out
#SBATCH -e Errors/job.%J.err


module purge 

source /share/apps/NYUAD/miniconda/3-4.11.0/bin/activate

conda activate tf-env2

export TF_CPP_MIN_LOG_LEVEL="2"
#echo $LD_LIBRARY_PATH

#Execute the code
#python test.py

python main.py "../DatabaseV2/TrainSet" "../DatabaseV2/TestSet"