#!/bin/bash
#SBATCH --job-name=palm_run
#SBATCH --nodes=2 --ntasks-per-node=24
#SBATCH --output=palm_logs/%j.log
#SBATCH --account=PAS0409

set -x  # for displaying the commands in the log for debugging

run_time=$1
io_config=$2
trial_dir=$3

palm_path=~/palm/current_version
job_name=$(date +"%FT%H%M%S")_$(basename $trial_dir)

julia --project=. run_model.jl $job_name $trial_dir

current_dir=`pwd`

cd $palm_path

PALM_COMMAND=("palmrun" "-r" "$job_name" "-c" "intel" "-X48" "-T24" "-q" "parallel" "-t" "$run_time" "-a" "$io_config" "-v" "-m" 4315)
"${PALM_COMMAND[@]}"

if [ $? -ne 0 ]
then
    # sometimes palm fails to launch, probably because the compiled code is not ready yet is my guess
    sleep 30
    echo "Palm failed initial launch, retrying"
    "${PALM_COMMAND[@]}"
    if [ $? -ne 0 ]
    then
        sleep 30
        echo "Palm failed retry launch, retrying"
        "${PALM_COMMAND[@]}"
    fi
fi

rm -r $palm_path/fastiocatalog/*$job_name*
if [ $io_config = "slurm" ]; then
     # with slurm io, we run in tmp dirctory, but palm still creates input files here, so we clean up
   rm -r $palm_path/JOBS/$job_name
fi

cd $current_dir

julia --project=. fetch_finished_trial.jl $trial_dir

if [ $? -eq 0 ]
then
    echo "Fetched data successfully"
    exit 0
else
    echo "Error Fetching data for $trial_dir and $job_name" >&2
    echo '{"trial_status": "FAILED"}' > $trial_dir/output.json
fi