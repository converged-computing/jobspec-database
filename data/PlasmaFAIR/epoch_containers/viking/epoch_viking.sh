#!/bin/bash

# Slurm settings
# --------------

# Set the following to configure your job. You must set 'mail-user' to your Uni of York
# email, and 'account' to the project code you were given when you signed up to Viking.

#SBATCH --job-name=epoch               # Job name
#SBATCH --mail-user=abc123@york.ac.uk  # Where to send mail
#SBATCH --account=ACCOUNT_CODE         # Project account
#SBATCH --mail-type=END,FAIL           # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --nodes=1                      # Number of comput nodes to run on
#SBATCH --ntasks-per-node=2            # Number of MPI processes to spawn per node (max 96)
#SBATCH --cpus-per-task=1              # Number of CPUS per process (leave this as 1!)
#SBATCH --mem-per-cpu=1gb              # Memory per task
#SBATCH --time=00:01:00                # Total time limit hrs:min:sec
#SBATCH --output=%x_%j.log             # Log file for stdout/stderr outputs
#SBATCH --partition=nodes              # 'test' for small test jobs (<1m), 'nodes' otherwise

# User settings
# -------------

# Choose method of running Epoch.
# 'Singularity' if using containers, 'Source' if compiled from source.
method="Singularity"

# Name of directory containing your 'input.deck' file.
# Recommended to use a relative path.
output_dir="."

# Number of dimensions in your run.
# Ignored if running from source.
dims="2"

# Use QED methods
# Set to '--photons' to activate, or just leave as an empty string
# Ignored if running from source.
photons=""

# If running Epoch from containers, set this to the 'run_epoch.py' script
# Ignored if running from source.
# Recommended to use a relative path.
run_epoch="./run_epoch.py"

# If running Epoch from source, set this to the compiled executable
# Ignored if running from containers.
# Recommended to use a relative path, and include './' in front.
epoch_exe="./bin/epoch2d"

# OpenMPI module used to compile Epoch
mpi_module="OpenMPI"

# -------------

module purge
module load ${mpi_module}

if [[ ${method} -eq "Singularity" ]]; then

  module load Python Apptainer

  # Suppress warnings
  export PMIX_MCA_gds=^ds12
  export PMIX_MCA_psec=^munge

  # Fix intra-node communication issue
  # https://ciq.com/blog/workaround-for-communication-issue-with-mpi-apps-apptainer-without-setuid/
  export OMPI_MCA_pml=ucx
  export OMPI_MCA_btl='^vader,tcp,openib,uct'
  export UCX_TLS=^'posix,cma'

  echo "Running Epoch with Apptainer using ${SLURM_NTASKS} processes"

  python ${run_epoch} singularity -d ${dims} -o ${output_dir} ${photons} --srun

  # Alternative in case the above isn't working:
  # srun singularity exec --bind ${output_dir}:/output oras://ghcr.io/plasmafair/epoch.sif:latest run_epoch -d ${dims} -o /output --srun ${photons}

elif [[ ${method} -eq "Source" ]]; then

  echo "Running Epoch from source using ${SLURM_NTASKS} processes"

  echo ${output_dir} | srun ${epoch_exe}

else

  echo "Set method to one of 'Singularity' or 'Source'" 1>&2
  exit 1

fi


