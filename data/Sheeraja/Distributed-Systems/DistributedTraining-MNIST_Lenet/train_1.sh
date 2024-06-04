#!/bin/bash -l
# NOTE the -l flag!

# The name of your job; be descriptive
#SBATCH --job-name=DagTrain1

# Where to save the output and error messages for your job?
# %x will fill in your job name, %j will fill in your job ID
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# How many tasks does your job need?
# This option advises the Slurm controller that job steps run within the
# allocation will launch a maximum of number tasks and to provide for
# sufficient resources. Defaults to one CPU per task.
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2

# How much time does your job need to run?
# Format: Days-Hours:Minutes:Seconds
# Max: 5 days (tier3), 1 day (debug)
#SBATCH --time=0-03:00:00

# How much memory per CPU: MB=m (default), GB=g, TB=t
# SBATCH --mem-per-cpu=200g
#SBATCH  --mem=0

# Your slurm account (created when you fill out the questionnaire)
#SBATCH --account=evidential 

# The partition to run your job on.
# Debug is for troubleshooting your code and getting your job to run.
# When your job runs successfully, switch to tier3.
# We reserve the right to kill jobs running on debug without warning if we need
# to debug with or train a researcher.
# SBATCH --gpus-per-task=1
# SBATCH --gpu-bind=single:1
#SBATCH --partition=debug
# GresTypes=gpu
#SBATCH --gres=gpu:a100:2

# Load your software using spack
# spack load <software>@<version> /<unique_hash>
# or
# Load your software using conda
# /full/path/to/conda activate <environment_name>
#SBATCH --mail-user=sheeraja206@gmail.com
#SBATCH --mail-type=ALL
# SBATCH --mail-type=BEGIN
# SBATCH --mail-type=END
# SBATCH --mail-type=FAIL
# SBATCH --mail-type=REQUEUE

# spack load --first openmpi@4.0.5
# spack load openmpi@4.0.5 /sa66g3e

# The code you actually need to run goes here
export PATH="$PATH:~/julia-1.8.5/bin"
# export JULIA_MPI_BINARY="system"
# export JULIA_MPIEXEC="srun"
# export JULIA_CUDA_MEMORY_POOL=none
# export FLUXMPI_DISABLE_CUDAMPI_SUPPORT=true

# julia --project="/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class" -e 'ENV["JULIA_MPI_BINARY"]="system";ENV["JULIA_MPIEXEC"]="srun"; using Pkg; Pkg.build("MPI"; verbose=true)'
# julia --project="/home/sr8685/distributed_systems/DaggerTraining" -e 'using Pkg; Pkg.build("MPI"; verbose=true)'
# mpiexecjl --project="/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class" -n 1 /home/sr8685/julia-1.8.5/bin/julia --threads=36 /home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/main.jl
# mpiexecjl --project="/home/sr8685/distributed_systems/DaggerTraining" -n 4 /home/sr8685/julia-1.8.5/bin/julia --threads=40 /home/sr8685/distributed_systems/DaggerTraining/trainattempt1.jl
# srun -n 1 /home/sr8685/julia-1.8.5/bin/julia --project="/home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class" --threads=36 /home/sr8685/PersonDetection/scripts/Pascal_VOC2012_ConvMixer_Class_20/main.jl

julia --project="/home/sr8685/distributed_systems/DaggerTraining" --threads=auto /home/sr8685/distributed_systems/DaggerTraining/train_1.jl
# srun hostname
# srun sleep 60