#!/bin/bash
#SBATCH -A dune                 # account to use for the job, '--account', '-A'
#SBATCH -J blip_graph_test      # job name, '--job-name', '-J'
#SBATCH -C gpu                  # type of job (constraint can be 'cpu' or 'gpu'), '--constraint', '-C'
#SBATCH -q shared               # Jobs requiring 1 or 2 gpus should use the shared setting, all others use 'regular'
#SBATCH -t 24:00:00             # amount of time requested for the job, '--time', 't'
#SBATCH -N 1                    # number of nodes, '--nodes', '-N'
#SBATCH -n 1                    # number of tasks '--ntasks', -n'
#SBATCH -c 32                   # number of cores per task, '--cpus-per-task', '-c'
#SBATCH --gpus-per-task=1       # number of gpus to be used per task
#SBATCH --gpus-per-node=1       # number of gpus per node.
#SBATCH --gpu-bind=none         # comment this out if you don't want all gpus visible to each task
#SBATCH --dependency=afterok:<optimize_blip_graph_prep_id>      # don't run until prep job is finished
#SBATCH --array=0-9

export LOCAL_SCRATCH=/pscratch/sd/${USER:0:1}/${USER}/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID
mkdir -p $LOCAL_SCRATCH
LOCAL_BLIP=/global/cfs/cdirs/dune/users/${USER}
LOCAL_DATA=/global/cfs/cdirs/dune/users/${USER}

# find the hyper_parameter_file associated with the array id

# use that hyper_parameter_file as input to the optimize_blip_graph.sh script
# along with the task_id

hyper_parameter_file="${LOCAL_BLIP}/hyper_parameter_data.csv"

# read in the hyper_parameter file
if [ ! -e "$hyper_parameter_file" ]; then
    echo "Error: Hyper parameter file '$hyper_parameter_file' not found!"
    exit 1
fi

hyper_parameter_config=$(head -n $((SLURM_ARRAY_TASK_ID+1)) "$hyper_parameter_file" | tail -1 | tr -d '\r')"/hyper_parameter_config.yaml"

setfacl -m u:nobody:x /global/cfs/cdirs/dune/users/${USER}
shifter --image=docker:infophysics/blip:latest \
        --volume="${LOCAL_SCRATCH}:/local_scratch;${LOCAL_BLIP}:/local_blip;${LOCAL_DATA}:/local_data" \
        ./blip_graph_optimize.sh $hyper_parameter_config