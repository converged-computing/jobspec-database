#!/usr/bin/env bash
#SBATCH -o /out/%x_%j_%a.out # Standard output
#SBATCH -e /err/%x_%j_%a.err # Standard error
#SBATCH --mem=0
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1-1
#SBATCH --account=def-account_manager

# This is a template file for scheduling experiments on a slurm cluster. Fill
# in all the lines of code marked with "TODO", then schedule this file using
# `sbatch`

USE_GPU=false # Set to true if requesting GPUs
LOCALHOSTS=("todo" "todo") # TODO: set hostnames of local devices

function islocal() {
    host=$(hostname)
    if [[ "${LOCALHOSTS[*]}" =~ "${host}" ]]; then
	return 0 # True: local
    else
	return 1 # False: not local
    fi
}

# TODO: Set module versions
STDENV_VERSION="2023"
GCC_VERSION="11.3.0"
JL_VERSION="1.10.0"
PY_VERSION="3.11" # for Brax environments; ensure PyCall set up for proper venv
CUDA_VERSION="11.8.0"
CUDNN_VERSION="8.6.0.163"

# TODO: Project path on HPC cluster
PROJECT_PATH="${HOME}/scratch/MirrorDescentActorCritic.jl"

# Set the timeout that worker processes wait for the master process to make
# connection
export JULIA_WORKER_TIMEOUT=120

# Ensure PythonCall and PyCall use the same interpreter
export JULIA_PYTHONCALL_EXE="@PyCall"
export JULIA_CONDAPKG_BACKEND="Null"

# Load modules
if islocal; then
    export OMP_NUM_THREADS=1
else
    cd $PROJECT_PATH

    # Make sure Julia is in charge of parallelsim, not OpenMP, by setting
    # OMP_NUM_THREADS=1 (or more generally, to SLURM_CPUS_PER_TASK)
    #
    # See https://pythonspeed.com/articles/concurrency-control/
    # See https://docs.alliancecan.ca/wiki/Running_jobs#Threaded_or_OpenMP_job
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    #
    # Alternatively, you can specifically set OPENBlas's number of threads
    # export OPENBLAS_NUM_THREADS=$(( $SLURM_CPUS_PER_TASK - 1 ))

    module load StdEnv/$STDENV_VERSION
    if $USE_GPU; then
	module load gcc/$GCC_VERSION
	module load cuda/$CUDA_VERSION
	module load cudnn/$CUDNN_VERSION

	# Add libcudnn to the LD_LIBRARY_PATH
	export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$EBROOTCUDNN/lib64
	echo
	echo "Using cudnn library at: $EBROOTCUDNN/lib64" 1>&2
    fi
    module load julia/$JL_VERSION
    module load python/$PY_VERSION
    module load mujoco

    # Send module list to stdout, so we always know which modules are loaded
    module list

    echo 1>&2
    echo "Job info:" 1>&2
    echo -e "\tJob start time: $SLURM_JOB_START_TIME" 1>&2
    echo -e "\tJob projected end time: $SLURM_JOB_END_TIME" 1>&2
    echo -e "\tJob ntasks: $SLURM_NTASKS" 1>&2
    echo -e "\tJob cpus-per-task: $SLURM_CPUS_PER_TASK" 1>&2
fi

# Get config file and save path
if islocal; then
    # TODO: path to save file
    SAVE_PATH="${HOME}/Documents/Code/Julia/MirrorDescentActorCritic.jl/results" # TODO

    # Get algorithm name from current filename alg.sh -> alg
    FILENAME=$(basename $0)
    ALG_NAME="${FILENAME%.*}"

    # Get configuration file path using current directory
    # config/x/y/z/alg.sh -> x/y/z
    PARALLEL_DIR=$(dirname $0)
    CONFIG_DIR=${PARALLEL_DIR/parallel/config}
    CONFIG="${CONFIG_DIR}/${ALG_NAME}.toml"
else
    # TODO: path to save file
    SAVE_PATH="ABSOLUTE_PATH" # TODO

    # Get the path to this jobscript
    JOB_SCRIPT=$(scontrol show job "$SLURM_JOBID" | awk -F= '/Command=/{print $2}')

    # In the jobscript replace "parallel" -> "config" to get the config file
    CONFIG=${JOB_SCRIPT/parallel/config}
    CONFIG="${CONFIG%.*}.toml"

    # In the jobscript replace "parallel" -> "err" and "parallel" -> "out" to
    # set the directories to store standard error and stanard output
    # respectively
    ERR=${JOB_SCRIPT/parallel/err}
    export SBATCH_ERROR="${ERR%.*}.err"
    OUT=${JOB_SCRIPT/parallel/out}
    export SBATCH_OUTPUT="${OUT%.*}.out"
fi

echo
echo "Current working directory: $PWD" 1>&2
echo "Using configuration file at: $CONFIG" 1>&2
echo "Saving data to: $SAVE_PATH" 2>&2

# Julia code to run
julia <(
cat << EOF
    using Pkg
    Pkg.activate(".")
    using Reproduce

    const CONFIG = "$CONFIG"
    const SAVE_PATH = "$SAVE_PATH"

    println("julia working in: ", pwd())

    function main()
	experiment = Reproduce.parse_experiment_from_config(CONFIG, SAVE_PATH)
	pre_experiment(experiment)
	ret = job(experiment)
	post_experiment(experiment, ret)
    end

    main()
EOF
)
