#!/bin/bash

################# Slurm part  ################################

# Write output as following (%A is ARRAY_ID and %a is inner task)
# File is created relative to cwd (unless adding path to these arguments) 
#SBATCH -o cluster_behavior_zebrafish_behavior-%A_%a.out
#SBATCH -e cluster_behavior_zebrafish_behavior-%A_%a.err

# Slurm paramters (memory should be enough by default, -t can be removed (default is higher))
#SBATCH -t 02:00:00

################# Parameters ################################

# External parameters (dataset_patg us /.../Lab-Shared/<your-data>/, fish is 20200720-f3). 
# Note: Event number is calculated from slurm parameters (its numbers)
dataset_path=$1
fish=$2
repository_relative_to_script_path=$3
opencv_conda_env=$4

# allow using the same script on 3 use cases
args=''
if [[ $# -ge 5 ]]; then
  if [[ "$5" == "--full" ]] ; then
     args='--full'
  elif [[ "$5" == "--control"  ]]; then
     args='--control_data'
  else
     echo "error: unknown 3rd argument $5" >&2; exit 1
  fi
fi

echo "Args: $args"

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

EVENT_NUM=$(( $((SLURM_ARRAY_TASK_ID + 1)))) # counting starts from 1 in event numbers

# check if script is started via SLURM or bash (slurm script is a temp copy of original)
# if with SLURM: there variable '$SLURM_JOB_ID' will exist
if [ -n $SLURM_ARRAY_TASK_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | awk -F= '/Command=/{print $2}' | head -n1 | cut -f1 -d " ")
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi

# get script's path to allow running additional relative scripts (if needed) from any folder without errors
path=$(dirname $SCRIPT_PATH)

echo "Array task id: $SLURM_ARRAY_TASK_ID, event num: $EVENT_NUM"
echo "Script path: $SCRIPT_PATH"

################# Main (same code can be used manually) #############################

# activate anaconda installed on your user (Default: /ems/..../<lab>/<user>/anaconda3
source ~/anaconda3/bin/activate
conda init

# Use predefined opencv env (from env file)
conda activate $opencv_conda_env

start_time=$(date)
echo "Start $start_time"

# args: parallel run on control, using calculated event number
m_args="--vid_type ".raw" --event_number $EVENT_NUM --parallel $args"

# doesn't have to be relative (can be cloned, but clone is quite slow) - usually should run main branch
python $path/../../main.py $dataset_path $fish $m_args
end_time=$(date)

echo "Stop $end_time"
conda deactivate

# can use end-start to calc running time if needed (slurm monitoring shows more information)
