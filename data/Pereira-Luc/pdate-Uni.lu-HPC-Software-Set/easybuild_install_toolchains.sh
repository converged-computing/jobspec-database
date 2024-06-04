#!/bin/bash -l
#SBATCH --job-name=install_eb_modules_toolchains
#SBATCH --output=out/install_eb_modules_toolchains_%j.out
#SBATCH --time=20:00:00
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --mem=0

# Purge old modules and set environment variables
module purge
export EASYBUILD_JOB_BACKEND=Slurm

# Function to print an error message and exit
print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
hash parallel 2>/dev/null && test $? -eq 0 || print_error_and_exit "Parallel is not installed on the system"

# Load the EasyBuild module
module load tools/EasyBuild/4.9.1

# Set to "intel-2023a" to install intel toolchain 2023a
# Set to "foss-2023a" to install foss toolchain 2023a
TOOLCHAIN="intel-2023a.eb"

# Create an array of EasyBuild files
EBFILES=($TOOLCHAIN)

# Create a directory for logs
mkdir -p eb_logs_toolchains

# Run command
parallel -j 1 --verbose --joblog eb_logs_toolchains/eb_joblog.log "srun -n1  -c 8 eb {} --robot --job --job-cores=8 --job-max-walltime=5 --job-backend-config=slurm --trace --accept-eula-for=Intel-oneAPI> eb_logs_toolchains/eb_log_{#}.log" ::: "${EBFILES[@]}"

echo 'Tasks are all running now.'
echo 'Use sq to see them. '
