#!/bin/bash

############################### Slurm parameters ##################################

# Write output as following (%A is ARRAY_ID and %a is inner task. The path can be absolute. Here it will be created in the folder you run command from)
#SBATCH -o cluster_behavior_full_zebrafish_behavior-%A_%a.out
#SBATCH -e cluster_behavior_full_zebrafish_behavior-%A_%a.err

# limit time for 12h (can be far less, it might make slurm manager be kinder in queue) & ask for larger mem
# Can use sacct and seff commands (can be read online) to validate your data need this (I think it should be good)
#SBATCH -t 12:00:00
#SBATCH --mem 35G

############################### parameters ##################################
dataset_path=$1
fish=$2
SIZE=$3
repository_relative_to_script_path=$4
opencv_conda_env=$5

# additional arguments for run. --fast_run is without tail
fast_run=""
if [[ $# -ge 6 ]]; then
  if [[ "$6" == "--fast_run" ]] ; then
      fast_run="--fast_run"
  fi
fi

input_args=$@ # save if needed
shift $# # remove arguments - this is preventing bug in source usage below

args="--vid_type ".raw" --full --parallel --fish_only $fast_run"
############################### Main (generic code) ##################################

# calculate 
STOP=$(( $((SLURM_ARRAY_TASK_ID + 1))*SIZE))
START="$(($STOP - $(($SIZE - 1))))"

args="$args --start $START --end $STOP"

# check if script is started via SLURM or bash (read command to get script's path)
# this allows having the code and scripts relative to each other, and not using user's abs paths
if [ -n $SLURM_ARRAY_TASK_ID ];  then
    # check the original location through scontrol and $SLURM_JOB_ID
    SCRIPT_PATH=$(scontrol show job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | awk -F= '/Command=/{print $2}' | head -n1 | cut -f1 -d " ")
else
    # otherwise: started with bash. Get the real location.
    SCRIPT_PATH=$(realpath $0)
fi
echo "Array task id: $SLURM_ARRAY_TASK_ID, event num: $EVENT_NUM"
echo "Script path: $SCRIPT_PATH"

# get script's path to allow running from any folder without errors
path=$(dirname $SCRIPT_PATH)

# activate anaconda installed on your user (Default: /ems/..../<lab>/<user>/anaconda3
source ~/anaconda3/bin/activate
conda init

# Use predefined opencv env
conda activate $opencv_conda_env

start_time=$(date)
echo "Start $start_time"
echo "Run fish $fish with args $args"
python $path/../$repository_relative_to_script_path/main.py $dataset_path $fish $args
end_time=$(date)
echo "Stop $end_time"

echo conda deactivate

