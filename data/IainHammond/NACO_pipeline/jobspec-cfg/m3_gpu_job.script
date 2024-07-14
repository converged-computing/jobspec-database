#!/bin/bash
# Usage: sbatch m3_gpu_job.script
# Prepared By: Kai Xi,  Apr 2015
#              help@massive.org.au
# Modified by: Iain Hammond

# NOTE: To activate a SLURM option, remove the whitespace between the '#' and 'SBATCH'

# To give your job a name, replace "MyJob" with an appropriate name
#SBATCH --job-name=NACO-CQTau

# To set a project account for credit charging,
#SBATCH --account=pd87

# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

# Request for GPU (in this case, Tesla V100)
#SBATCH --gres=gpu:V100:1
#SBATCH --partition=m3g

# Memory usage (MB)
#SBATCH --mem=8000

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=0-01:00:00

# To receive an email when job completes or fails
#SBATCH --mail-user=<iain.hammond@monash.edu>
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# Set the file for output (stdout)
#SBATCH --output=pipeline_output%j.out

# Set the file for error log (stderr)
#SBATCH --error=pipeline_output_error%j.err

# Use reserved node to run job when a node reservation is made for you already
# SBATCH --reservation=reservation_name

# Command to run a gpu job
module load cuda
nvidia-smi
deviceQuery

env # print environmental variables if you wish, good for debugging
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
source /home/ihammond/miniconda3/etc/profile.d/conda.sh # initialise conda shell
conda activate VIPenv # activate personal VIP conda environment
ulimit -s unlimited # recommended by M3 support team to prevent stack size memory error
python run_script.py # runs the script
